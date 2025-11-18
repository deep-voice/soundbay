import torch
import torch.nn as nn
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
            self.yolo_model = YOLO(f'yolov8{model_size}.pt')
            self.num_classes = num_classes or config.NUM_CLASSES
            print(f"Initialized YOLO model: yolov8{model_size}.pt")
            print(f"Note: YOLO has its own training API. For best results, use: yolo train data=dclde_2026/data.yaml model=yolov8{model_size}.pt")
        except ImportError:
            raise ImportError("ultralytics not available. Install with: pip install ultralytics")
    
    def forward(self, x):
        """
        Forward pass for YOLO.
        Note: YOLO expects images in [B, C, H, W] format with values in [0, 1] or [0, 255]
        """
        # YOLO expects images in specific format
        # x is [B, 3, H, W] spectrogram
        # Convert to [0, 255] range if needed
        if x.max() <= 1.0:
            x = x * 255.0
        
        # YOLO model expects numpy or PIL, but we can use it directly
        # For training, YOLO uses its own training loop, so this is mainly for inference
        results = self.yolo_model(x, verbose=False)
        return results
    
    def train_yolo(self, data, epochs, imgsz, batch, **kwargs):
        """
        Train YOLO using its native training API.
        This is the recommended way to train YOLO.
        """
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
