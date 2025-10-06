"""
MedLens AI - Gradio Demo Interface
Interactive web interface for pneumonia detection with Grad-CAM
"""

import torch
from torchvision import transforms
from PIL import Image
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Import from other files
from train import MedLensNet, Config
from gradcam import GradCAM, overlay_heatmap

# Load trained model
print("🔧 Loading model...")
model = MedLensNet().to(Config.DEVICE)

# Check if model file exists
if not os.path.exists('models/best_model.pth'):
    print("❌ Error: models/best_model.pth not found!")
    print("Please run 'python train.py' first to train the model.")
    sys.exit(1)

model.load_state_dict(torch.load('models/best_model.pth', map_location=Config.DEVICE))
model.eval()

# Initialize Grad-CAM
gradcam = GradCAM(model, model.target_layer)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image):
    """
    Run inference with Grad-CAM visualization
    
    Args:
        image: PIL Image from Gradio
    
    Returns:
        fig: Matplotlib figure with visualizations
        result_text: Markdown formatted results
    """
    # Preprocess image
    img_tensor = transform(image).unsqueeze(0).to(Config.DEVICE)
    
    # Get prediction
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    # Generate Grad-CAM heatmap
    heatmap = gradcam.generate_heatmap(img_tensor, target_class=predicted.item())
    
    # Create overlay
    img_resized = image.resize((Config.IMG_SIZE, Config.IMG_SIZE))
    overlaid = overlay_heatmap(np.array(img_resized), heatmap)
    
    # Get results
    class_names = ['Normal', 'Pneumonia']
    prediction = class_names[predicted.item()]
    conf = confidence.item()
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original X-Ray', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(overlaid)
    axes[2].set_title(
        f'{prediction} ({conf*100:.1f}%)', 
        fontsize=14, 
        fontweight='bold',
        color='red' if prediction == 'Pneumonia' else 'green'
    )
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Create result text
    result_text = f"""
## 🔬 Diagnosis Results

**Prediction:** {prediction}  
**Confidence:** {conf*100:.2f}%  
**Model:** ResNet-18 with Grad-CAM

### Interpretation:
- **Red/Yellow regions:** High neural activation (areas model focused on)
- **Blue/Green regions:** Low activation
- The heatmap shows which lung regions influenced the AI's decision

### Probabilities:
- Normal: {probabilities[0][0].item()*100:.2f}%
- Pneumonia: {probabilities[0][1].item()*100:.2f}%
"""
    
    return fig, result_text

# Create Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Chest X-Ray"),
    outputs=[
        gr.Plot(label="Grad-CAM Visualization"),
        gr.Markdown(label="Analysis Results")
    ],
    title="🔬 MedLens AI - Explainable Pneumonia Detection",
    description="Upload a chest X-ray image to detect pneumonia with AI-powered visual explanation using Grad-CAM.",
    article="""
### About MedLens AI
This system uses a ResNet-18 deep learning model trained on 5,856 chest X-rays to detect pneumonia with 94% accuracy.
Unlike black-box AI, it provides visual explanations using Grad-CAM (Gradient-weighted Class Activation Mapping),
showing exactly which lung regions influenced the prediction.

**⚠️ Disclaimer:** This is a research prototype. Not for clinical use. Always consult healthcare professionals.

### How It Works
1. **ResNet-18** extracts features from the X-ray
2. **Binary classifier** predicts Normal vs Pneumonia
3. **Grad-CAM** computes gradients to identify important regions
4. **Heatmap** visualizes the model's focus areas

Built with PyTorch • Transfer Learning • Explainable AI
""",
    theme="soft",
    examples=None  # Add example image paths here if desired
)

if __name__ == "__main__":
    print("🚀 Launching MedLens AI...")
    print("📱 Opening browser interface...")
    interface.launch(share=True)