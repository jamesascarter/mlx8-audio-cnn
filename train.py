import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import numpy as np
from tqdm import tqdm
from model import AudioCNN
import os

# Training configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data_split(file_path):
    """Load a specific data split"""
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    
    features = torch.stack([item["input_tensor"] for item in data])
    labels = torch.tensor([item["label"] for item in data], dtype=torch.long)
    
    return features, labels

def create_data_loader(features, labels, batch_size, shuffle=True):
    """Create a data loader"""
    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_features, batch_labels in tqdm(train_loader, desc="Training"):
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def evaluate(model, data_loader, criterion, device, split_name="Validation"):
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
    
    return avg_loss, accuracy

def plot_training_history(train_losses, train_accs, val_losses, val_accs):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracies
    ax2.plot(train_accs, label='Train Accuracy', color='blue')
    ax2.plot(val_accs, label='Validation Accuracy', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    print(f"Using device: {DEVICE}")
    
    # Load data splits
    print("Loading data splits...")
    train_features, train_labels = load_data_split("urban_train.pkl")
    val_features, val_labels = load_data_split("urban_val.pkl")
    test_features, test_labels = load_data_split("urban_test.pkl")
    
    print(f"Train: {len(train_features)} samples")
    print(f"Validation: {len(val_features)} samples")
    print(f"Test: {len(test_features)} samples")
    
    # Create data loaders
    train_loader = create_data_loader(train_features, train_labels, BATCH_SIZE, shuffle=True)
    val_loader = create_data_loader(val_features, val_labels, BATCH_SIZE, shuffle=False)
    test_loader = create_data_loader(test_features, test_labels, BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = AudioCNN(num_classes=10).to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler (optional)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Training history
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    best_val_acc = 0.0
    patience_counter = 0
    patience = 10  # Early stopping patience
    
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    print(f"Early stopping patience: {patience} epochs")
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE, "Validating")
        
        # Update learning rate scheduler
        scheduler.step(val_acc)
        
        # Store history
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"New best model saved! Validation accuracy: {val_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Progress: {epoch+1}/{NUM_EPOCHS} epochs completed")
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Load best model and evaluate on test set
    print("\nEvaluating best model on test set...")
    model.load_state_dict(torch.load('best_model.pth'))
    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE, "Testing")
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(train_losses, train_accs, val_losses, val_accs)
    
    # Save final model and results
    torch.save(model.state_dict(), 'final_model.pth')
    
    # Save training history
    training_history = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'final_epoch': len(train_losses)
    }
    
    with open('training_history.pkl', 'wb') as f:
        pickle.dump(training_history, f)
    
    print("\nResults Summary:")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Training completed in {len(train_losses)} epochs")
    print("\nFiles saved:")
    print("- best_model.pth (best model based on validation)")
    print("- final_model.pth (model after final epoch)")
    print("- training_history.png (training curves)")
    print("- training_history.pkl (training metrics)")

if __name__ == "__main__":
    main() 