from model import Baseline, LSTMModel
from dataloader import load_fi2010
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = LSTMModel().to(device)
model_name = "lstm"

train_loader, val_loader, test_loader = load_fi2010()

epochs = 30
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
# Reduce LR by 50% if val loss doesn't improve for 5 epochs
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)

best_val_acc = 0.0


def evaluate(loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)

            total_loss += loss.item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += len(y)

    return total_loss / total, correct / total


for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()

        train_loss += loss.item() * len(y)
        train_correct += (logits.argmax(dim=1) == y).sum().item()
        train_total += len(y)

    train_loss /= train_total
    train_acc = train_correct / train_total

    val_loss, val_acc = evaluate(val_loader)
    scheduler.step(val_loss)

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), f"models/{model_name}_best.pt")

    lr = opt.param_groups[0]['lr']
    print(f"Epoch {epoch+1:2d}/{epochs} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
          f"LR: {lr:.6f}")

# Load best model for test evaluation
model.load_state_dict(torch.load(f"models/{model_name}_best.pt", map_location=device))
test_loss, test_acc = evaluate(test_loader)
print(f"\n{'='*50}")
print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
print(f"Best Val Accuracy: {best_val_acc:.4f}")

# Export to ONNX
model.eval()
dummy_input = torch.randn(1, 50, 40).to(device)
torch.onnx.export(
    model,
    dummy_input,
    f"models/{model_name}.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
)
print(f"Exported to models/{model_name}.onnx")
