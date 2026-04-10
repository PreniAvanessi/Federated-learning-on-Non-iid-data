import torch
import torch.nn.functional as F
import numpy as np
import csv
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad() 
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)

def evaluate(model, loader, device): 
    
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    
    # Containers for sklearn metrics
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for x, y in loader:   
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits = model(x)  
            
            # Loss calculation
            loss = F.cross_entropy(logits, y)
            loss_sum += loss.item() * x.size(0)

            # Predictions
            probs = F.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)
            
            # Stats
            correct += (pred == y).sum().item()
            total += y.size(0)
            
            # Store for advanced metrics
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    if total == 0:
        return {"accuracy": 0, "loss": 0, "f1_score": 0, "auc": 0}

    # 1. Basic Metrics
    acc = correct / total
    avg_loss = loss_sum / total
    
    # 2. Advanced Metrics (F1 & AUC)
    f1 = f1_score(all_labels, all_preds, average="macro")
    
    # AUC requires concatenated probabilities
    y_prob_concat = np.concatenate(all_probs)
    try:
        # Multi-class AUC (One-vs-Rest)
        auc = roc_auc_score(all_labels, y_prob_concat, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")

    return {
        "accuracy": acc,
        "loss": avg_loss,
        "f1_score": f1,
        "auc": auc
    }

# CSV

def save_csv(rows, filepath):
    """Save a list of dictionaries to a CSV file."""
    if not rows:
        print("WARNING: No rows to save for", filepath)
        return

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("Saved CSV:", filepath)

def parameter_stats(ndarrays):
    """Compute statistics for model parameters."""
    flat = np.concatenate([p.flatten() for p in ndarrays])
    return {
        "param_mean": float(flat.mean()),
        "param_std": float(flat.std()),
        "param_min": float(flat.min()),
        "param_max": float(flat.max()),
        "num_params": int(flat.size),
    }
