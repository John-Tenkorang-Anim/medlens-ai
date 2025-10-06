"""
MedLens AI - Training Script
Trains ResNet-18 for pneumonia detection with transfer learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==================== CONFIGURATION ====================
class Config:
    IMG_SIZE = 224
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    EPOCHS = 10
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"🚀 Using device: {Config.DEVICE}")

# ==================== DATASET CLASS ====================
class ChestXRayDataset(Dataset):
    """Custom dataset for chest X-ray images"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load image paths and labels
        for label_idx, label_name in enumerate(['NORMAL', 'PNEUMONIA']):
            label_dir = os.path.join(data_dir, label_name)
            if not os.path.exists(label_dir):
                print(f"⚠️  Warning: {label_dir} not found!")
                continue
            
            for img_name in os.listdir(label_dir):
                if img_name.endswith(('.jpeg', '.jpg', '.png')):
                    self.images.append(os.path.join(label_dir, img_name))
                    self.labels.append(label_idx)
        
        print(f"📊 Loaded {len(self.images)} images from {data_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ==================== MODEL DEFINITION ====================
class MedLensNet(nn.Module):
    """Modified ResNet-18 for binary classification"""
    
    def __init__(self):
        super(MedLensNet, self).__init__()
        # Load pre-trained ResNet-18
        self.model = models.resnet18(pretrained=True)
        
        # Replace final layer for binary classification
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2)
        
        # Store reference to last conv layer for Grad-CAM
        self.target_layer = self.model.layer4[1].conv2
    
    def forward(self, x):
        return self.model(x)

# ==================== TRAINING FUNCTION ====================
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    """Train the model and track metrics"""
    
    history = {
        'train_loss': [], 
        'train_acc': [], 
        'val_loss': [], 
        'val_acc': []
    }
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # ========== Training Phase ==========
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # ========== Validation Phase ==========
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'\n📊 Epoch {epoch+1}: Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/best_model.pth')
            print(f'✅ Saved best model (Val Acc: {val_acc:.2f}%)')
    
    return model, history

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    print("\n📂 Loading datasets...")
    train_dataset = ChestXRayDataset('data/train', transform=train_transform)
    val_dataset = ChestXRayDataset('data/test', transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=2
    )
    
    # Initialize model
    print("\n🧠 Initializing model...")
    model = MedLensNet().to(Config.DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Train
    print("\n🚀 Starting training...")
    model, history = train_model(
        model, train_loader, val_loader, 
        criterion, optimizer, Config.EPOCHS
    )
    
    # Save final model
    torch.save(model.state_dict(), 'models/final_model.pth')
    print("\n✅ Training complete! Model saved to models/final_model.pth")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('outputs/training_history.png')
    print("📊 Training plots saved to outputs/training_history.png")