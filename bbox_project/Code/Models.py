import torch
import torch.nn as nn

class FlexibleTinyDetector(nn.Module):
    def __init__(self, max_boxes, n_classes):
        super(FlexibleTinyDetector, self).__init__()
        self.max_boxes = max_boxes
        self.n_classes = n_classes

        # Encoder: Works on any image size
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(2), 
            
            # Layer 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            
            # Layer 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )
        
        # The "Bridge": Squashes any spatial size to a fixed 1x4 representation
        # self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 4))
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 16))
        
        # Regression Head: 6 outputs per box [x, y, w, h, class, conf]
        self.fc = nn.Sequential(
            # nn.Linear(64 * 1 * 4, 128),
            nn.Linear(64 * 8 * 16, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, max_boxes * (5 + self.n_classes)) # max_boxes * (x, y, w, h, class_1, ..., class_n, conf)
        )

    def forward(self, x):
        # x shape: [Batch, 1, H, W]
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1) # Shape: [Batch, 256]
        x = self.fc(x)
        
        # Reshape to [Batch, Max_Boxes, 5 + n_classes] (x, y, w, h, class_1, ..., class_n, conf)
        return x.view(-1, self.max_boxes, 5 + self.n_classes)
    
class GlobalDetectorLongerTime(nn.Module):
    def __init__(self, max_boxes, n_classes, pooling_size=(4, 40)):
        super().__init__()
        self.max_boxes = max_boxes
        self.n_classes = n_classes
        self.pooling_size = pooling_size

        # 5 layers to handle the longer temporal context
        self.features = nn.Sequential(
            self._block(1, 32),    # Layer 1
            self._block(32, 64),   # Layer 2
            self._block(64, 128),  # Layer 3
            self._block(128, 256), # Layer 4
            self._block(256, 512), # Layer 5: New high-level feature extractor
        )
        
        self.pool = nn.AdaptiveAvgPool2d(pooling_size) 
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            # Input dim: 512 filters * 4 freq_bins * 80 time_bins = 163,840
            nn.Linear(512 * pooling_size[0] * pooling_size[1], 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3), # Increased dropout for the larger layer
            nn.Linear(512, max_boxes * (5 + n_classes))
        )

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.fc(x)
        # Reshape to [Batch, Max_Boxes, coords+class+conf]
        return x.view(-1, self.max_boxes, 5 + self.n_classes)
    
    def init_weights(self, neg_bias=-1.5):
        """Custom initialization logic for this specific architecture"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming is best for ReLU/LeakyReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        # Apply the "Goldilocks" Confidence Bias to the very last layer
        last_layer = self.fc[-1]
        n_out = 5 + self.n_classes
        with torch.no_grad():
            for i in range(self.max_boxes):
                # The confidence score is the last index in each box's block
                conf_idx = (i * n_out) + (n_out - 1)
                last_layer.bias[conf_idx] = neg_bias
                
                stride = i * 7
                nn.init.constant_(self.fc[-1].bias[stride + 2], neg_bias) # Set conf bias for all boxes at once
                nn.init.constant_(self.fc[-1].bias[stride + 3], neg_bias) # Set class bias to 0 for all boxes at once
        
        print(f"Model initialized: Kaiming/Xavier with conf_bias={neg_bias}")
    
class DeepSpectrogramDetector(nn.Module):
    def __init__(self, max_boxes, n_classes, pooling_size=(4, 8)):
        super().__init__()
        self.max_boxes = max_boxes
        self.n_classes = n_classes

        self.features = nn.Sequential(
            self._block(1, 32),    # L1
            self._block(32, 64),   # L2
            self._block(64, 128),  # L3
            self._block(128, 256), # L4
        )
        
        # Increased resolution to 4x8 (32 spatial "zones")
        self.pool = nn.AdaptiveAvgPool2d(pooling_size)
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * pooling_size[0] * pooling_size[1], 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(512, max_boxes * (5 + n_classes))
        )

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.fc(x)
        # Reshape to [Batch, Max_Boxes, coords+class+conf]
        return x.view(-1, self.max_boxes, 5 + self.n_classes)
    
    def init_weights(self, neg_bias=-1.5):
        """Custom initialization logic for this specific architecture"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming is best for ReLU/LeakyReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        # Apply the "Goldilocks" Confidence Bias to the very last layer
        last_layer = self.fc[-1]
        n_out = 5 + self.n_classes
        with torch.no_grad():
            for i in range(self.max_boxes):
                # The confidence score is the last index in each box's block
                conf_idx = (i * n_out) + (n_out - 1)
                last_layer.bias[conf_idx] = neg_bias
        
        print(f"Model initialized: Kaiming/Xavier with conf_bias={neg_bias}")

def initialize_detector(model, neg_bias=-2.0):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    # Target the very last Linear layer
    last_layer = model.fc[-1]
    with torch.no_grad():
        # Every (5 + n_classes)-th element in the bias is the 'conf' score
        stride = 5 + model.n_classes
        for i in range(model.max_boxes):
            # The confidence is the last index in the group
            conf_index = (i * stride) + (stride - 1)
            last_layer.bias[conf_index] = -4.0

if __name__ == "__main__":
    # Example usage:
    # model = FlexibleTinyDetector(max_boxes=3, n_classes=3)
    # initialize_detector(model)
    model = DeepSpectrogramDetector(max_boxes=3, n_classes=3)
    model.init_weights(neg_bias=-2.0)
    img = torch.randn(32, 1, 21, 65) # Batch of 32, Grayscale, 21 height, 65 width
    output = model(img) # Output shape: [32, 3, 6]
    print(output.shape)
    print(output[0])