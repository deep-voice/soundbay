import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoModel, AutoConfig, AutoProcessor

from dclde_2026 import config


class BEATSDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        print(f"Initializing BEATS model: {config.TRANSFORMER_MODEL_NAME}")
        print(f"WARNING: BEATS typically expects 16kHz audio. Current sample rate: {config.SAMPLE_RATE}")
        print("Audio will be resampled to 16kHz for BEATS processing.")
        
        try:
            self.processor = AutoProcessor.from_pretrained(config.TRANSFORMER_MODEL_NAME)
            self.beats_model = AutoModel.from_pretrained(config.TRANSFORMER_MODEL_NAME)
        except Exception as e:
            print(f"Warning: Could not load processor, using model only. Error: {e}")
            self.processor = None
            self.beats_model = AutoModel.from_pretrained(config.TRANSFORMER_MODEL_NAME)
        
        self.beats_target_sr = 16000
        beats_config = AutoConfig.from_pretrained(config.TRANSFORMER_MODEL_NAME)
        embed_dim = beats_config.hidden_size
        
        self.num_boxes_to_predict = 50
        self.num_outputs_per_box = 5 + config.NUM_CLASSES
        
        self.detection_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, self.num_boxes_to_predict * self.num_outputs_per_box)
        )
        
    def forward(self, x):
        if config.SAMPLE_RATE != self.beats_target_sr:
            x = torch.stack([
                torchaudio.functional.resample(x[i:i+1].unsqueeze(0), config.SAMPLE_RATE, self.beats_target_sr)
                .squeeze(0).squeeze(0) for i in range(x.shape[0])
            ])
        
        if self.processor is not None:
            input_values = self.processor(x.cpu().numpy(), sampling_rate=self.beats_target_sr, return_tensors="pt")["input_values"].to(x.device)
        else:
            input_values = x.unsqueeze(1)
        
        outputs = self.beats_model(input_values=input_values, return_dict=True)
        pooled = outputs.pooler_output if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None else outputs.last_hidden_state.mean(dim=1)
        
        preds = self.detection_head(pooled)
        out = preds.view(x.shape[0], self.num_boxes_to_predict, self.num_outputs_per_box)
        
        return torch.cat([
            torch.sigmoid(out[..., :4]),
            torch.sigmoid(out[..., 4:5]),
            torch.sigmoid(out[..., 5:])
        ], dim=-1)


class YOLOModelWrapper(nn.Module):
    """
    Wrapper for ultralytics YOLO model to integrate with PyTorch training loop.
    YOLO model uses its own training API, but this wrapper allows using it in our training script.
    """
    def __init__(self, model_size='n', num_classes=None):
        super().__init__()
        try:
            from ultralytics import YOLO
            from ultralytics.nn.modules.head import Detect
            
            self.num_classes = num_classes or config.NUM_CLASSES
            
            # Load pretrained YOLO model
            self._yolo = [YOLO(f'yolov8{model_size}.pt')]
            print(f"Initialized YOLO model: yolov8{model_size}.pt")
            
            # Spectrogram transform (Moved from Dataset to GPU)
            self.spec_transform = torchaudio.transforms.Spectrogram(
                n_fft=config.N_FFT_LIST[-1], # 2048
                hop_length=config.HOP_LENGTH,
                power=2.0
            )
            
            # Adjust the number of classes in the model if necessary
            if self.yolo_model.model.nc != self.num_classes:
                print(f"Adjusting YOLO head from {self.yolo_model.model.nc} to {self.num_classes} classes")
                
                # Find the Detect module and rebuild cv3 layers
                for name, m in self.yolo_model.model.named_modules():
                    if isinstance(m, Detect):
                        # Update class count
                        old_nc = m.nc
                        m.nc = self.num_classes
                        m.no = m.reg_max * 4 + self.num_classes
                        
                        # Rebuild cv3 (class prediction) layers with correct number of classes
                        # cv3 structure: Sequential(Conv(in, nc), Conv(nc, nc), Conv2d(nc, nc))
                        from ultralytics.nn.modules.conv import Conv
                        
                        for i in range(len(m.cv3)):
                            # Get input channels from first layer
                            in_ch = m.cv3[i][0].conv.in_channels
                            
                            # Rebuild cv3[i] with new nc
                            m.cv3[i] = nn.Sequential(
                                Conv(in_ch, self.num_classes, 3),
                                Conv(self.num_classes, self.num_classes, 3),
                                nn.Conv2d(self.num_classes, self.num_classes, 1)
                            )
                        
                        # Update model-level nc
                        self.yolo_model.model.nc = self.num_classes
                        
                        print(f"Rebuilt Detect head: nc={m.nc}, no={m.no}")
                        break
            
            # Register the inner model as a submodule so parameters are found
            # and .train()/.eval() work on the actual network
            if hasattr(self.yolo_model, 'model'):
                self.core_model = self.yolo_model.model
                # FORCE gradients on for all parameters
                for param in self.core_model.parameters():
                    param.requires_grad = True
            
            self.core_model.train()

        except ImportError:
            raise ImportError("ultralytics not available. Install with: pip install ultralytics")
    
    @property
    def yolo_model(self):
        return self._yolo[0]
    
    def train(self, mode=True):
        """Override train to ensure parameters require gradients after switching modes.
        
        Ultralytics YOLO's predict() and __call__() methods can freeze parameters
        when running in inference mode. We need to re-enable gradients when
        switching back to training mode.
        """
        super().train(mode)
        if mode and hasattr(self, 'core_model'):
            # Re-enable gradients on all parameters after switching to train mode
            for param in self.core_model.parameters():
                param.requires_grad = True
        return self
    
    def get_loss_criterion(self, use_soft_negative=True):
        """Get the loss criterion for this model"""
        from dclde_2026.losses import SoftNegativeYOLOLoss
        try:
            from ultralytics.utils.loss import v8DetectionLoss
        except ImportError:
            try:
                from ultralytics.yolo.utils.loss import v8DetectionLoss
            except ImportError:
                raise ImportError("Could not import v8DetectionLoss from ultralytics")
        
        if hasattr(self.yolo_model.model, 'args') and isinstance(self.yolo_model.model.args, dict):
            class AttributeDict(dict):
                def __getattr__(self, attr):
                    return self.get(attr)
                def __setattr__(self, attr, value):
                    self[attr] = value
            
            args_dict = self.yolo_model.model.args
            if 'box' not in args_dict: args_dict['box'] = 7.5
            if 'cls' not in args_dict: args_dict['cls'] = 0.5
            if 'dfl' not in args_dict: args_dict['dfl'] = 1.5
            self.yolo_model.model.args = AttributeDict(args_dict)

        base_loss = v8DetectionLoss(self.yolo_model.model)
        if use_soft_negative:
            return SoftNegativeYOLOLoss(base_loss, cls_loss_scale=0.7)
        return base_loss

    def forward(self, x):
        """
        Forward pass for YOLO.
        Input: Raw Audio [B, 1, T] or [B, T]
        Output: Model predictions
        """
        # 1. Compute Spectrogram on GPU
        # Check input shape
        if x.dim() == 2: # [B, T]
            x = x.unsqueeze(1) # [B, 1, T]
        
        # x is now [B, 1, T]. Spectrogram expects [..., T].
        # If we pass [B, 1, T], output is [B, 1, F, T]
        spec = self.spec_transform(x) 
        spec = 10 * torch.log10(spec + 1e-10)
        # Per-sample normalization (spec is [B, 1, F, T])
        spec_min = spec.amin(dim=(1, 2, 3), keepdim=True)
        spec_max = spec.amax(dim=(1, 2, 3), keepdim=True)
        spec = (spec - spec_min) / (spec_max - spec_min + 1e-6)
        
        # Resize if needed
        if spec.shape[2] != config.TARGET_FREQ_BINS:
             # spec is [B, 1, F, T]
             # F.interpolate expects [B, C, H, W]
             # We want to resize dim 2 (F)
             spec = F.interpolate(spec, size=(config.TARGET_FREQ_BINS, spec.shape[3]), mode='bilinear', align_corners=False)

        # Flip vertically (Low freq at bottom)
        spec = torch.flip(spec, [2])
        
        # Replicate to 3 channels [B, 3, F, T]
        x = spec.repeat(1, 3, 1, 1)
        
        # Pad/resize to ensure dimensions are divisible by 32 (YOLO stride requirement)
        _, _, h, w = x.shape
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)

        # 2. YOLO Forward
        # Always return raw tensors from forward() for loss computation
        # Use predict() method for inference with Results objects
        
        # Force Detect head to return raw output (not post-processed)
        detect_module = None
        for m in self.core_model.modules():
            if m.__class__.__name__ == 'Detect':
                detect_module = m
                break
        
        if detect_module:
            prev_training = detect_module.training
            detect_module.training = True  # Force raw output mode
            
            out = self.core_model(x)
            
            detect_module.training = prev_training
            return out
        
        return self.core_model(x)
    
    def predict(self, x, **kwargs):
        """
        Run inference and return Results objects (for visualization/export).
        Use this instead of forward() when you need post-processed predictions.
        """
        # Prepare input (same as forward)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        spec = self.spec_transform(x)
        spec = 10 * torch.log10(spec + 1e-10)
        # Per-sample normalization (spec is [B, 1, F, T])
        spec_min = spec.amin(dim=(1, 2, 3), keepdim=True)
        spec_max = spec.amax(dim=(1, 2, 3), keepdim=True)
        spec = (spec - spec_min) / (spec_max - spec_min + 1e-6)
        
        if spec.shape[2] != config.TARGET_FREQ_BINS:
            spec = F.interpolate(spec, size=(config.TARGET_FREQ_BINS, spec.shape[3]), mode='bilinear', align_corners=False)
        
        spec = torch.flip(spec, [2])
        x = spec.repeat(1, 3, 1, 1)
        
        # Pad for stride 32
        _, _, h, w = x.shape
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
        
        return self.yolo_model(x, verbose=False, **kwargs)
    
    def train_yolo(self, data, epochs, imgsz, batch, **kwargs):
        return self.yolo_model.train(
            data=data,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            **kwargs
        )


def get_model(model_type=None):
    """
    Get model based on config or specified type.
    
    Args:
        model_type: 'beats' or 'yolo'. If None, uses config.MODEL_TYPE
    
    Returns:
        Model instance
    """
    model_type = model_type or config.MODEL_TYPE.lower()
    
    if model_type == 'beats':
        return BEATSDetectionModel()
    elif model_type == 'yolo':
        return YOLOModelWrapper(model_size=config.YOLO_MODEL_SIZE, num_classes=config.NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'beats' or 'yolo'.")
