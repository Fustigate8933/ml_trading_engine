from model import Baseline
from dataloader import load_fi2010
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = Baseline().to(device)
train_loader, val_loader, test_loader = load_fi2010()

epochs = 20
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)


def evaluate(loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # disable gradient computation (faster, less memory)
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

        opt.zero_grad()          # clear gradients from previous step
        logits = model(x)        # forward pass: (256, 3)
        loss = loss_fn(logits, y)  # cross-entropy needs both predictions AND labels
        loss.backward()          # compute gradients
        opt.step()               # update weights

        train_loss += loss.item() * len(y)
        train_correct += (logits.argmax(dim=1) == y).sum().item()
        train_total += len(y)

    train_loss /= train_total
    train_acc = train_correct / train_total

    val_loss, val_acc = evaluate(val_loader)

    print(f"Epoch {epoch+1:2d}/{epochs} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

test_loss, test_acc = evaluate(test_loader)
print(f"\n{'='*50}")
print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

torch.save(model.state_dict(), "models/baseline.pt")
print(f"Model saved to models/baseline.pt")
