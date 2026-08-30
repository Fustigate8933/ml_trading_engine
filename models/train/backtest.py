"""
Backtest: simulate trading on FI-2010 test set using the LSTM model.

Measures:
  - Prediction accuracy (overall + per-class)
  - Simulated PnL from directional bets
  - Win rate, max drawdown, Sharpe ratio
"""

import numpy as np
import torch
from dataloader import load_fi2010
from model import LSTMModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = LSTMModel().to(device)
model.load_state_dict(torch.load("models/lstm_best.pt", map_location=device))
model.eval()

# Load data
_, _, test_loader = load_fi2010()

# Collect all predictions and labels
all_preds = []
all_labels = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().numpy()
        labels = y.numpy()
        all_preds.append(preds)
        all_labels.append(labels)

all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

# === Prediction Accuracy ===
total = len(all_preds)
correct = (all_preds == all_labels).sum()
print(f"{'='*50}")
print(f"Prediction Accuracy: {correct}/{total} = {correct/total:.4f}")
print()

# Per-class accuracy
class_names = ["Down (Sell)", "Stable (Hold)", "Up (Buy)"]
for c in range(3):
    mask = all_labels == c
    if mask.sum() == 0:
        continue
    acc = (all_preds[mask] == c).sum() / mask.sum()
    print(f"  {class_names[c]:15s}: {acc:.4f} ({mask.sum()} samples)")

# === Simulated Trading ===
# Strategy: when model predicts 0 (down) → short, predicts 2 (up) → long
# PnL: +1 if prediction matches actual direction, -1 if wrong
# Skip "stable" predictions (hold)

position = 0       # 0=flat, 1=long, -1=short
trades = []         # list of PnL per trade
cumulative_pnl = []
pnl = 0.0
peak_pnl = 0.0
max_drawdown = 0.0

for i in range(len(all_preds)):
    pred = all_preds[i]
    label = all_labels[i]

    if pred == 1:  # hold
        continue

    # Direction bet
    if pred == 2:  # predicted up → long
        reward = 1.0 if label == 2 else (-1.0 if label == 0 else 0.0)
    elif pred == 0:  # predicted down → short
        reward = 1.0 if label == 0 else (-1.0 if label == 2 else 0.0)
    else:
        continue

    pnl += reward
    trades.append(reward)
    cumulative_pnl.append(pnl)

    # Drawdown tracking
    if pnl > peak_pnl:
        peak_pnl = pnl
    dd = peak_pnl - pnl
    if dd > max_drawdown:
        max_drawdown = dd

trades = np.array(trades)

print(f"\n{'='*50}")
print(f"Trading Simulation (directional bet)")
print(f"  Total trades:    {len(trades)}")
print(f"  Winning trades:  {(trades > 0).sum()} ({(trades > 0).mean()*100:.1f}%)")
print(f"  Losing trades:   {(trades < 0).sum()} ({(trades < 0).mean()*100:.1f}%)")
print(f"  Neutral trades:  {(trades == 0).sum()} ({(trades == 0).mean()*100:.1f}%)")
print(f"  Total PnL:       {pnl:+.1f} units")
print(f"  Max Drawdown:    {max_drawdown:.1f} units")

# Sharpe ratio: mean(returns) / std(returns) * sqrt(N)
# Annualized assuming ~252 trading days, ~1000 trades/day
if len(trades) > 1 and trades.std() > 0:
    daily_sharpe = trades.mean() / trades.std()
    annualized_sharpe = daily_sharpe * np.sqrt(252 * 1000)
    print(f"  Sharpe Ratio:    {annualized_sharpe:.2f} (annualized, rough estimate)")
else:
    print(f"  Sharpe Ratio:    N/A")

print(f"\n{'='*50}")
if pnl > 0:
    print(f"Result: PROFITABLE (+{pnl:.0f} units over {len(trades)} trades)")
elif pnl < 0:
    print(f"Result: UNPROFITABLE ({pnl:.0f} units over {len(trades)} trades)")
else:
    print(f"Result: BREAKEVEN")
