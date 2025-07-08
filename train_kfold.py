import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import numpy as np
from tqdm import tqdm
from model import AudioCNN
import os
import wandb

# Training configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_FOLDS = 10
WEIGHT_DECAY = 1e-4 

def load_all_data():
    """Load all data with fold information"""
    with open("urban_all_with_folds.pkl", "rb") as f:
        all_data = pickle.load(f)
    
    # Convert to tensors
    features = torch.stack([item["input_tensor"] for item in all_data])
    labels = torch.tensor([item["label"] for item in all_data], dtype=torch.long)
    folds = torch.tensor([item["fold"] for item in all_data], dtype=torch.long)
    
    return features, labels, folds

def create_kfold_split(features, labels, folds, test_fold):
    """Create train/val/test split for a specific fold"""
    # Test set: one fold
    test_mask = (folds == test_fold)
    test_features = features[test_mask]
    test_labels = labels[test_mask]
    
    # Train set: remaining folds
    train_mask = (folds != test_fold)
    train_features = features[train_mask]
    train_labels = labels[train_mask]
    
    # Split train into train and validation (80/20)
    num_train = len(train_features)
    num_val = int(num_train * 0.2)
    
    # Random shuffle for train/val split
    indices = torch.randperm(num_train)
    val_indices = indices[:num_val]
    train_indices = indices[num_val:]
    
    val_features = train_features[val_indices]
    val_labels = train_labels[val_indices]
    train_features = train_features[train_indices]
    train_labels = train_labels[train_indices]
    
    return train_features, train_labels, val_features, val_labels, test_features, test_labels

def create_data_loader(features, labels, batch_size, shuffle=True):
    """Create a data loader"""
    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_epoch(model, train_loader, criterion, optimizer, device, fold_num, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_features, batch_labels in tqdm(train_loader, desc="Training"):
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    # Log training metrics
    wandb.log({
        f"fold_{fold_num}/train_loss": epoch_loss,
        f"fold_{fold_num}/train_acc": epoch_acc,
        f"fold_{fold_num}/epoch": epoch
    })
    
    return epoch_loss, epoch_acc

def evaluate(model, data_loader, criterion, device, split_name="Validation", fold_num=None, epoch=None):
    """Evaluate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_features, batch_labels in tqdm(data_loader, desc=split_name):
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
    
    avg_loss = running_loss / len(data_loader)
    accuracy = 100 * correct / total
    
    # Log evaluation metrics if fold_num is provided
    if fold_num is not None and epoch is not None:
        wandb.log({
            f"fold_{fold_num}/{split_name.lower()}_loss": avg_loss,
            f"fold_{fold_num}/{split_name.lower()}_acc": accuracy,
            f"fold_{fold_num}/epoch": epoch
        })
    
    return avg_loss, accuracy

def train_fold(features, labels, folds, fold_num):
    """Train model for a specific fold"""
    print(f"\n{'='*50}")
    print(f"Training Fold {fold_num}")
    print(f"{'='*50}")
    
    # Create split for this fold
    train_features, train_labels, val_features, val_labels, test_features, test_labels = create_kfold_split(
        features, labels, folds, fold_num
    )
    
    print(f"Train: {len(train_features)} samples")
    print(f"Validation: {len(val_features)} samples")
    print(f"Test: {len(test_features)} samples")
    
    # Create data loaders
    train_loader = create_data_loader(train_features, train_labels, BATCH_SIZE, shuffle=True)
    val_loader = create_data_loader(val_features, val_labels, BATCH_SIZE, shuffle=False)
    test_loader = create_data_loader(test_features, test_labels, BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = AudioCNN(num_classes=10).to(DEVICE)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    best_val_acc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE, fold_num, epoch)
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE, "Validation", fold_num, epoch)
        scheduler.step(val_acc)
        # Save best model for this fold
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'models/best_model_fold{fold_num}.pth')
            # Log best validation accuracy
            wandb.log({f"fold_{fold_num}/best_val_acc": best_val_acc})
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
    
    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(f'models/best_model_fold{fold_num}.pth'))
    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE, "Test", fold_num, None)
    
    # Log final test results for this fold
    wandb.log({
        f"fold_{fold_num}/final_test_acc": test_acc,
        f"fold_{fold_num}/final_test_loss": test_loss
    })
    
    print(f"Fold {fold_num} Results:")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    return {
        "fold": fold_num,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc
    }

def main():
    # Initialize wandb
    wandb.init(
        project="mlx8-audio-cnn",
        config={
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
            "k_folds": K_FOLDS,
            "device": str(DEVICE),
            "model": "AudioCNN"
        }
    )
    
    print(f"Using device: {DEVICE}")
    print(f"Starting {K_FOLDS}-fold cross-validation...")
    
    # Load all data
    features, labels, folds = load_all_data()
    print(f"Loaded {len(features)} samples with fold information")
    
    # Train all folds
    fold_results = []
    for fold in range(1, K_FOLDS + 1):
        result = train_fold(features, labels, folds, fold)
        fold_results.append(result)
    
    # Calculate average results
    avg_val_acc = np.mean([r["best_val_acc"] for r in fold_results])
    avg_test_acc = np.mean([r["test_acc"] for r in fold_results])
    std_test_acc = np.std([r["test_acc"] for r in fold_results])
    
    # Log final results
    wandb.log({
        "final/avg_val_acc": avg_val_acc,
        "final/avg_test_acc": avg_test_acc,
        "final/std_test_acc": std_test_acc
    })
    
    print(f"\n{'='*50}")
    print("K-FOLD CROSS-VALIDATION RESULTS")
    print(f"{'='*50}")
    print(f"Average Validation Accuracy: {avg_val_acc:.2f}% ± {np.std([r['best_val_acc'] for r in fold_results]):.2f}%")
    print(f"Average Test Accuracy: {avg_test_acc:.2f}% ± {std_test_acc:.2f}%")
    
    # Print individual fold results
    print("\nIndividual Fold Results:")
    for result in fold_results:
        print(f"Fold {result['fold']}: Val={result['best_val_acc']:.2f}%, Test={result['test_acc']:.2f}%")
    
    # Save results
    with open('kfold_results.pkl', 'wb') as f:
        pickle.dump(fold_results, f)
    
    # Finish wandb run
    wandb.finish()
    
    print("Training completed!")

if __name__ == "__main__":
    main() 