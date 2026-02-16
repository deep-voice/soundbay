import torch
import onnxruntime as ort
import torchaudio
from transformers import AutoModel, AutoProcessor, AutoConfig


class PerchONNX(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device(config.device)
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(config.onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def forward(self, x: torch.Tensor):
        x_np = x.detach().cpu().numpy()
        embeddings = self.session.run(None, {self.input_name: x_np})[1]
        return torch.from_numpy(embeddings).to(self.device)


class BEATSEncoder(torch.nn.Module):
    """BEATS encoder with automatic resampling to 16kHz."""
    
    def __init__(self, config):
        super().__init__()
        self.device = torch.device(config.device)
        self.beats_model_name = config.beats_model_name
        self.beats_target_sr = 16000
        self.source_sr = config.target_sr
        
        print(f"Initializing BEATS model: {self.beats_model_name}")
        print(f"WARNING: BEATS requires 16kHz audio. Current sample rate: {self.source_sr}")
        print("Audio will be resampled to 16kHz for BEATS processing.")
        
        # Load BEATS model and processor
        self.processor = AutoProcessor.from_pretrained(self.beats_model_name)
        
        self.beats_model = AutoModel.from_pretrained(self.beats_model_name)
        self.beats_model.to(self.device)
        # Note: Model is frozen by default in train.py, but can be unfrozen if needed
        
        # Get embedding dimension from BEATS config
        beats_config = AutoConfig.from_pretrained(self.beats_model_name)
        self.embed_dim = beats_config.hidden_size
        
        # Resampler for converting input audio to 16kHz
        if self.source_sr != self.beats_target_sr:
            self.resampler = torchaudio.transforms.Resample(
                orig_freq=self.source_sr,
                new_freq=self.beats_target_sr
            ).to(self.device)
        else:
            self.resampler = None
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, samples) or (batch, 1, samples) - audio waveform at source_sr
        
        Returns:
            embeddings: (batch, seq_len, embed_dim) - BEATS sequence embeddings
        """
        # Squeeze channel dimension if present: (batch, 1, samples) -> (batch, samples)
        if x.dim() == 3:
            x = x.squeeze(1)
        
        # Ensure x is on the correct device
        x = x.to(self.device)
        
        # Resample to 16kHz if needed
        if self.resampler is not None:
            x = self.resampler(x)
        
        # Process with BEATS processor
        # Note: BEATS processor returns processed waveform values (input_values), NOT a spectrogram
        # BEATS uses a learned convolutional frontend that processes raw waveforms internally
        x_np = x.detach().cpu().numpy()
        inputs = self.processor(
                x_np, 
                sampling_rate=self.beats_target_sr, 
                return_tensors="pt",
                padding=True
            )
        input_values = inputs["input_values"].to(self.device)  # (batch, seq_len) - normalized waveform, not spectrogram

        # Get BEATS embeddings (gradients enabled/disabled based on model.training)
        outputs = self.beats_model(input_values=input_values, return_dict=True)
        
        # BEATS outputs last_hidden_state: (batch, seq_len, embed_dim)
        # This is a sequence of embeddings, similar to Perch's temporal-spectral structure
        embeddings = outputs.last_hidden_state
        
        return embeddings


class BioacousticDetector(torch.nn.Module):
    def __init__(self, config, num_temporal=16, num_spectral=4):
        super().__init__()
        self.encoder = PerchONNX(config)
        self.num_temporal = num_temporal
        self.num_spectral = num_spectral
        self.num_classes = config.num_classes
        embed_dim = config.embed_dim
        hidden_dim = config.hidden_dim

        # Learnable positional embeddings: separate for temporal and spectral dimensions
        self.temporal_pos_embed = torch.nn.Parameter(
            torch.randn(num_temporal, embed_dim) * 0.02
        )
        self.spectral_pos_embed = torch.nn.Parameter(
            torch.randn(num_spectral, embed_dim) * 0.02
        )

        # Deeper head: two attention blocks with residual + LayerNorm
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=config.num_heads, batch_first=True
        )
        self.attention2 = torch.nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=config.num_heads, batch_first=True
        )
        self.norm1 = torch.nn.LayerNorm(embed_dim)
        self.norm2 = torch.nn.LayerNorm(embed_dim)

        # Shared projection then per-class towers
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(config.dropout),
        )
        self.class_towers = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(config.dropout),
                torch.nn.Linear(hidden_dim, 1),
            )
            for _ in range(config.num_classes)
        ])

    def _build_positional_encoding(self, T, C, device):
        """Build 2D positional encoding by combining temporal and spectral embeddings."""
        temp_pos = self.temporal_pos_embed[:T].unsqueeze(1).expand(T, C, -1)
        spec_pos = self.spectral_pos_embed[:C].unsqueeze(0).expand(T, C, -1)
        pos_encoding = (temp_pos + spec_pos).reshape(T * C, -1)
        return pos_encoding

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        emb = self.encoder(x)
        if emb.dim() == 4:
            batch, T, C, D = emb.shape
            emb = emb.reshape(batch, T * C, D)
            pos_encoding = self._build_positional_encoding(T, C, emb.device)
            x = emb + pos_encoding.unsqueeze(0)
        else:
            x = emb

        # Block 1: attention + residual + norm
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        # Block 2: attention + residual + norm
        attn_out, _ = self.attention2(x, x, x)
        x = self.norm2(x + attn_out)

        # Shared MLP then per-class towers
        h = self.shared(x)  # (batch, seq, hidden_dim)
        logits = torch.cat([tower(h) for tower in self.class_towers], dim=-1)  # (batch, seq, num_classes)
        return logits


class BioacousticDetectorBEATS(torch.nn.Module):
    """Bioacoustic detector using BEATS encoder instead of Perch."""

    def __init__(self, config):
        super().__init__()
        self.encoder = BEATSEncoder(config)
        embed_dim = self.encoder.embed_dim
        hidden_dim = config.hidden_dim
        self.target_frames = config.num_output_frames

        max_seq_len = int(config.window_sec * config.target_sr / 16000 * 50)
        self.temporal_pos_embed = torch.nn.Parameter(
            torch.randn(max_seq_len, embed_dim) * 0.02
        )

        # Deeper head: two attention blocks with residual + LayerNorm
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=config.num_heads, batch_first=True
        )
        self.attention2 = torch.nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=config.num_heads, batch_first=True
        )
        self.norm1 = torch.nn.LayerNorm(embed_dim)
        self.norm2 = torch.nn.LayerNorm(embed_dim)

        # Shared projection then per-class towers
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(config.dropout),
        )
        self.class_towers = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(config.dropout),
                torch.nn.Linear(hidden_dim, 1),
            )
            for _ in range(config.num_classes)
        ])

    def forward(self, x):
        """
        Args:
            x: (batch, samples) or (batch, 1, samples) - audio waveform

        Returns:
            output: (batch, target_frames, num_classes) - frame-level predictions
        """
        emb = self.encoder(x)
        batch, seq_len, embed_dim = emb.shape
        pos_encoding = self.temporal_pos_embed[:seq_len].unsqueeze(0)
        x = emb + pos_encoding

        # Block 1: attention + residual + norm
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        # Block 2: attention + residual + norm
        attn_out, _ = self.attention2(x, x, x)
        x = self.norm2(x + attn_out)

        # Project to target frame count if needed
        if seq_len != self.target_frames:
            x = x.transpose(1, 2)
            x = torch.nn.functional.interpolate(
                x, size=self.target_frames, mode='linear', align_corners=False
            )
            x = x.transpose(1, 2)

        # Shared MLP then per-class towers
        h = self.shared(x)
        output = torch.cat([tower(h) for tower in self.class_towers], dim=-1)
        return output
