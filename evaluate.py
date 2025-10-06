"""
Model Evaluation Script
Generates comprehensive performance metrics and visualizations
"""

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    roc_curve
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms

from train import MedLensNet, ChestXRayDataset, Config

# Load model
print("🔧 Loading model...")
model = MedLensNet().to(Config.DEVICE)
model.load_state_dict(torch.load('models/best_model.pth', map_location=Config.DEVICE))
model.eval()

# Load test data
test_transform = transforms.Compose([
    transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("📂 Loading test dataset...")
test_dataset = ChestXRayDataset('data/test', transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# Evaluate model
print("🔍 Evaluating model...")
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(Config.DEVICE)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

# Classification report
print("\n" + "="*60)
print(" CLASSIFICATION REPORT")
print("="*60)
print(classification_report(
    all_labels, 
    all_preds, 
    target_names=['Normal', 'Pneumonia'],
    digits=4
))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=['Normal', 'Pneumonia'],
    yticklabels=['Normal', 'Pneumonia'],
    cbar_kws={'label': 'Count'}
)
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('outputs/confusion_matrix.png', dpi=300)
print("\n Confusion matrix saved to outputs/confusion_matrix.png")

# ROC Curve
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
auc = roc_auc_score(all_labels, all_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve', fontsize=16, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/roc_curve.png', dpi=300)
print(" ROC curve saved to outputs/roc_curve.png")

# Summary statistics
print("\n" + "="*60)
print(" SUMMARY STATISTICS")
print("="*60)
print(f"AUC-ROC Score:        {auc:.4f}")
print(f"Total Test Samples:   {len(all_labels)}")
print(f"Correct Predictions:  {sum([1 for i in range(len(all_preds)) if all_preds[i] == all_labels[i]])}")
print(f"Accuracy:             {100 * sum([1 for i in range(len(all_preds)) if all_preds[i] == all_labels[i]]) / len(all_labels):.2f}%")
print("="*60)

print("\n Evaluation complete!")
