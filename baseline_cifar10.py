# baseline_cifar10.py
import os
import time
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import CNN
from utils import train_one_epoch, evaluate, save_csv

def main(
    epochs=50,
    batch_size=64,
    lr=1e-3,
    seed=42,
    out_dir="reports/baseline/cifar10",
):
    os.makedirs(out_dir, exist_ok=True)

    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Centralized baseline CIFAR-10 | device={device} | epochs={epochs} | lr={lr}")

    stats = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(*stats)
    ])

    train_ds = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR10("./data", train=False, download=True, transform=transform)
    
    pin = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=pin)

    model = CNN(input_channels=3, num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # NEW: Add a scheduler to lower LR when accuracy plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)

    rows = []
    start_time = time.time()
    best_acc = 0.0

    # Epoch 0 evaluation
    metrics0 = evaluate(model, test_loader, device)
    rows.append({
        "epoch": 0, "train_loss": None, "test_loss": float(metrics0["loss"]),
        "test_accuracy": float(metrics0["accuracy"]), "test_f1": float(metrics0["f1_score"]),
        "test_auc": float(metrics0["auc"]), "elapsed_sec": float(time.time() - start_time),
    })

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate(model, test_loader, device)
        
        # Step the scheduler based on test accuracy
        current_acc = metrics['accuracy']
        scheduler.step(current_acc)

        rows.append({
            "epoch": int(epoch), "train_loss": float(train_loss),
            "test_loss": float(metrics["loss"]), "test_accuracy": float(current_acc),
            "test_f1": float(metrics["f1_score"]), "test_auc": float(metrics["auc"]),
            "elapsed_sec": float(time.time() - start_time),
        })

        if current_acc > best_acc:
            best_acc = current_acc
            # Save best model checkpoint
            torch.save(model.state_dict(), os.path.join(out_dir, "best_baseline_model.pt"))

        print(f"[Epoch {epoch}] train_loss={train_loss:.4f} test_acc={current_acc:.4f} (Best: {best_acc:.4f})")

    csv_path = os.path.join(out_dir, "baseline_cifar10_log.csv")
    save_csv(rows, csv_path)
    print(f"Final Best Baseline Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main(epochs=50, lr=1e-3)