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
import sys

# Import your model
from model import CNN 

def get_latest_checkpoint_smart(checkpoint_dir):
    """
    Scans the folder for ANY .npz file, prints them, 
    and picks the one with the highest number.
    """
    # 1. Check if folder exists
    if not checkpoint_dir.exists():
        print(f"❌ ERROR: Folder not found: {checkpoint_dir}")
        print(f"   Current working directory: {os.getcwd()}")
        return None, 0

    # 2. List ALL files in that folder
    all_files = list(checkpoint_dir.glob("*.npz"))
    
    print(f"📂 Scanning '{checkpoint_dir}'...")
    if not all_files:
        print("   ⚠️  Folder exists but contains NO .npz files!")
        # List *any* file to see what's inside
        any_files = os.listdir(checkpoint_dir)
        print(f"   Files actually found in folder: {any_files[:5]}")
        return None, 0
    
    print(f"   ✅ Found {len(all_files)} .npz files. Example: {all_files[0].name}")

    # 3. Find the one with the highest number
    latest_file = None
    max_round = -1
    
    for f in all_files:
        # Look for ANY number in the filename
        # This matches "round_5", "rounds_5", "5", "global_5", etc.
        match = re.search(r"(\d+)", f.name)
        if match:
            r = int(match.group(1))
            if r > max_round:
                max_round = r
                latest_file = f
    
    if latest_file:
        print(f"   🎯 Selected Best File: '{latest_file.name}' (Round {max_round})")
        
    return latest_file, max_round

def plot_cm(dataset_name, num_clients):
    print(f"\n--- Processing {dataset_name} ({num_clients} clients) ---")

    # 1. SETUP PATHS
    # Try to handle both common naming conventions (plural vs singular)
    # We will check both "params_..." and "param_..." just in case
    possible_paths = [
        Path(f"reports/params_{dataset_name}_{num_clients}clients"),
        Path(f"reports/param_{dataset_name}_{num_clients}clients")
    ]
    
    checkpoint_dir = None
    for p in possible_paths:
        if p.exists():
            checkpoint_dir = p
            break
            
    if not checkpoint_dir:
        print(f"❌ ERROR: Could not find a params folder. Checked:")
        for p in possible_paths:
            print(f"   - {p}")
        return

    # Output Path
    save_dir = Path(f"reports/{dataset_name}/figures")
    save_dir.mkdir(parents=True, exist_ok=True)

    # 2. Find the checkpoint
    checkpoint_path, round_num = get_latest_checkpoint_smart(checkpoint_dir)
    
    if not checkpoint_path:
        print("❌ Stopping because no checkpoint file could be selected.")
        return

    # 3. DATASET SETUP
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

    # 4. MODEL LOADING
    model = CNN(input_channels=1, num_classes=10) 
    model.eval()
    
    try:
        # Load params (handles both Dict and NpzFile)
        params = np.load(checkpoint_path)
        if isinstance(params, np.lib.npyio.NpzFile):
            # Sort files to ensure order (e.g. arr_0, arr_1...)
            files = sorted(params.files)
            param_list = [params[f] for f in files]
        else:
            param_list = params
            
        with torch.no_grad():
            for p, arr in zip(model.parameters(), param_list):
                p.copy_(torch.tensor(arr))
        print("   ✅ Model weights loaded successfully.")
    except Exception as e:
        print(f"   ❌ Failed to load model parameters: {e}")
        return

    # 5. INFERENCE
    y_true = []
    y_pred = []

    print("   🚀 Running inference on test set...")
    with torch.no_grad():
        for x, y in test_loader:
            outputs = model(x)
            preds = outputs.argmax(dim=1)
            y_true.extend(y.numpy())
            y_pred.extend(preds.numpy())

    # 6. PLOT
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
    
    print(f"   🎉 Success! Saved plot to: {output_file}")

if __name__ == "__main__":
    # Run for Fashion-MNIST
    plot_cm(dataset_name="fashion_mnist", num_clients=15)
    
    # Run for MNIST
    plot_cm(dataset_name="mnist", num_clients=15)