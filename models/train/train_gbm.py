"""
LightGBM model for LOB mid-price prediction.

Unlike the LSTM which sees a window of 50 snapshots, GBT operates on a
single feature vector. We flatten the window into one row: 50 * 40 = 2000 features.

This is the same input the MLP baseline used, but GBTs handle tabular data
much better than MLPs — no normalization needed, handles feature interactions
natively, and inference is sub-microsecond on CPU.
"""

import numpy as np
import lightgbm as lgb
from dataloader import load_fi2010
from sklearn.metrics import accuracy_score, classification_report
import time

# Load data
train_loader, val_loader, test_loader = load_fi2010(batch_size=4096)


def loader_to_flat(loader, window=50, features=40):
    """Convert DataLoader of (batch, window, features) into flat (N, window*features) arrays."""
    X_list, y_list = [], []
    for x, y in loader:
        # x: (batch, 50, 40) → flatten to (batch, 2000)
        X_list.append(x.numpy().reshape(x.shape[0], -1))
        y_list.append(y.numpy())
    return np.concatenate(X_list), np.concatenate(y_list)


print("Flattening datasets...")
X_train, y_train = loader_to_flat(train_loader)
X_val, y_val = loader_to_flat(val_loader)
X_test, y_test = loader_to_flat(test_loader)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

params = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 127,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,      # use 80% of features per tree (regularization)
    "bagging_fraction": 0.8,      # use 80% of samples per tree
    "bagging_freq": 5,
    "verbose": 1,
    "num_threads": 16,            # use P-cores
}

# Train
print("\nTraining LightGBM...")
callbacks = [
    lgb.early_stopping(stopping_rounds=20),
    lgb.log_evaluation(period=10),
]

model = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    valid_sets=[val_data],
    valid_names=["val"],
    callbacks=callbacks,
)

# Evaluate
y_pred_val = model.predict(X_val).argmax(axis=1)
y_pred_test = model.predict(X_test).argmax(axis=1)

val_acc = accuracy_score(y_val, y_pred_val)
test_acc = accuracy_score(y_test, y_pred_test)

print(f"\n{'='*50}")
print(f"Val  Accuracy: {val_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"\nTest Classification Report:")
print(classification_report(y_test, y_pred_test, target_names=["Down", "Stable", "Up"]))

# Benchmark inference latency (single sample, CPU)
x_single = X_test[0:1]
# Warmup
for _ in range(100):
    model.predict(x_single)

N = 10000
start = time.perf_counter_ns()
for _ in range(N):
    model.predict(x_single)
end = time.perf_counter_ns()

avg_ns = (end - start) / N
print(f"{'='*50}")
print(f"Single-sample inference: {avg_ns:.0f} ns ({avg_ns/1000:.1f} µs)")
print(f"vs LSTM TensorRT: ~348,000 ns (348 µs)")
print(f"Speedup: {348000/avg_ns:.0f}x")

# Save model
model.save_model("models/lgbm_model.txt")
print(f"\nModel saved to models/lgbm_model.txt")

# Feature importance (top 20)
importance = model.feature_importance(importance_type="gain")
top_idx = np.argsort(importance)[-20:][::-1]
print(f"\nTop 20 features by gain:")
for i in top_idx:
    # Decode: feature i = timestep (i // 40), field (i % 40)
    timestep = i // 40
    field = i % 40
    field_names = ["ask_p", "ask_v", "bid_p", "bid_v"] * 10
    level = field // 4 + 1
    ftype = field_names[field]
    print(f"  Feature {i:4d}: timestep={timestep:2d}, level={level:2d}, {ftype}  (gain={importance[i]:.0f})")
