import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import cv2
import numpy as np
from tqdm import tqdm
from scipy.special import softmax
from scipy.special import expit
import hydra
from torchvision import transforms
import datetime
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_image, class_idx=None):
        # Forward pass
        output = self.model(input_image)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[:, class_idx].squeeze()
        class_score.backward(retain_graph=True)
        
        # Generate CAM
        gradients = self.gradients
        activations = self.activations
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3))
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[2:], dtype=torch.float32)
        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i, :, :]
        
        # Apply ReLU
        cam = F.relu(cam)
        cam = cam / torch.max(cam)  # Normalize
        
        return cam.detach().numpy()
    
def visualize_cam(args) -> None:
    model = torch.load(args.model_path, weights_only=False)
    target_layer = args.target_layer
    input_image = args.input_image
    class_idx = args.class_idx

    grad_cam = GradCAM(model, target_layer)
    cam = grad_cam.generate_cam(input_image, class_idx)

    # Visualize the CAM
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    from pathlib import Path

    model_path = Path("model.pth")
    model = 