import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import glob
import re
import os

from model import CNN 

def get_latest_checkpoint(checkpoint_dir):
  
    if not checkpoint_dir.exists():
        print(f"❌ Error: Folder not found: {checkpoint_dir}")
        return None, 0

    # 1. Look for all .npz files
    files = list(checkpoint_dir.glob("*.npz"))
    
    if not files:
        print(f"⚠️  No .npz files found in {checkpoint_dir}")
        return None, 0
    
    latest_file = None
    max_round = -1
    
  
    for f in files:
        match = re.search(r"(\d+)", f.name)
        if match:
            r = int(match.group(1))
            if r > max_round:
                max_round = r
                latest_file = f
                
    return latest_file, max_round

def plot_cm(dataset_name, num_clients):
    print(f"\n--- Generating Confusion Matrix for {dataset_name} ({num_clients} clients) ---")

    # 1. SETUP PATHS
    # We check both common naming conventions just to be safe
    path_options = [
        Path(f"reports/params_{dataset_name}_{num_clients}clients"),
        Path(f"reports/param_{dataset_name}_{num_clients}clients")
    ]
    
    checkpoint_dir = None
    for p in path_options:
        if p.exists():
            checkpoint_dir = p
            break
    
    if not checkpoint_dir:
        print(f"❌ Error: Could not find checkpoint folder for {dataset_name}.")
        return

    # 2. FIND LATEST CHECKPOINT
    checkpoint_path, round_num = get_latest_checkpoint(checkpoint_dir)
    
    if not checkpoint_path:
        print("❌ Error: No valid checkpoint file found.")
        return

    print(f"✅ Loading checkpoint: {checkpoint_path.name} (Round {round_num})")

    # 3. OUTPUT DIRECTORY
    save_dir = Path(f"reports/{dataset_name}/figures")
    save_dir.mkdir(parents=True, exist_ok=True)

    # 4. DATASET SETUP
    if dataset_name == "mnist":
        stats = ((0.1307,), (0.3081,))
        test_dataset = datasets.MNIST("./data", train=False, download=True, 
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize(*stats)
                                      ]))
        class_names = [str(i) for i in range(10)]
        
    elif dataset_name == "fashion_mnist":
        stats = ((0.2860,), (0.3530,))
        test_dataset = datasets.FashionMNIST("./data", train=False, download=True, 
                                             transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(*stats)
                                             ]))
        class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", 
                       "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]
    else:
        raise ValueError("Unknown dataset")

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # 5. MODEL & WEIGHTS
    model = CNN(input_channels=1, num_classes=10) 
    model.eval()
    
    try:
        # Load parameters
        params = np.load(checkpoint_path)
        
        # Handle different .npz structures (Flower vs Standard)
        if isinstance(params, np.lib.npyio.NpzFile):
            files = sorted(params.files)
            param_list = [params[f] for f in files]
        else:
            param_list = params
            
        with torch.no_grad():
            for p, arr in zip(model.parameters(), param_list):
                p.copy_(torch.tensor(arr))
    except Exception as e:
        print(f"❌ Failed to load model parameters: {e}")
        return

    # 6. INFERENCE
    y_true = []
    y_pred = []

    print("🚀 Running inference...")
    with torch.no_grad():
        for x, y in test_loader:
            outputs = model(x)
            preds = outputs.argmax(dim=1)
            y_true.extend(y.numpy())
            y_pred.extend(preds.numpy())

    # 7. PLOT
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, 
                yticklabels=class_names)
    
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix: {dataset_name} (Round {round_num})")
    
    output_filename = f"confusion_matrix_{dataset_name}_{num_clients}clients.png"
    output_file = save_dir / output_filename
    
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"🎉 Success! Plot saved to: {output_file}")

if __name__ == "__main__":
    # Run for Fashion-MNIST
    plot_cm(dataset_name="fashion_mnist", num_clients=15)
    
    # Run for MNIST
    plot_cm(dataset_name="mnist", num_clients=15)