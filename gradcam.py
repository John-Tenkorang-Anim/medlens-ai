"""
Grad-CAM Implementation
Gradient-weighted Class Activation Mapping for visual explanations
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    """
    Grad-CAM: Visual explanations from deep networks
    
    How it works:
    1. Forward pass through network
    2. Backward pass to compute gradients
    3. Global average pooling of gradients = importance weights
    4. Weighted combination of feature maps = heatmap
    """
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks to capture gradients and activations
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Hook to save forward pass activations"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_heatmap(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_tensor: Preprocessed input (1, 3, 224, 224)
            target_class: Class to visualize (None = predicted class)
        
        Returns:
            heatmap: Numpy array of activation heatmap
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass for target class
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()
        
        # Global average pooling of gradients
        # Shape: [batch, channels, height, width] -> [channels]
        weights = torch.mean(self.gradients, dim=(2, 3))[0]
        
        # Weighted combination of activation maps
        heatmap = torch.zeros(self.activations.shape[2:]).to(input_tensor.device)
        for i, weight in enumerate(weights):
            heatmap += weight * self.activations[0, i, :, :]
        
        # Apply ReLU and normalize
        heatmap = F.relu(heatmap)
        heatmap = heatmap / (torch.max(heatmap) + 1e-8)
        
        return heatmap.cpu().numpy()


def overlay_heatmap(image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay Grad-CAM heatmap on original image
    
    Args:
        image: Original image (numpy array or tensor)
        heatmap: Grad-CAM heatmap
        alpha: Transparency of overlay
        colormap: OpenCV colormap
    
    Returns:
        Overlaid image
    """
    # Convert tensor to numpy if needed
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
        # Denormalize
        image = (image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
        image = np.uint8(np.clip(image, 0, 255))
    
    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_resized, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlaid = cv2.addWeighted(image, 1-alpha, heatmap_colored, alpha, 0)
    
    return overlaid