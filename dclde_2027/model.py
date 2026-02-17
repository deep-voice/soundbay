import torch
import onnxruntime as ort
import torchaudio
from transformers import AutoModel, AutoProcessor, AutoConfig


def _normalize_state_dict_keys(state_dict):
    """
    Strip leading 'module.' and 'model.' prefixes (e.g. DataParallel or wrapper).
    Applied repeatedly so 'model.module.classifier.0.weight' -> 'classifier.0.weight'.
    """
    if not state_dict:
        return state_dict
    out = dict(state_dict)
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "model."):
            if any(k.startswith(prefix) for k in out.keys()):
                out = {k.replace(prefix, "", 1) if k.startswith(prefix) else k: v for k, v in out.items()}
                changed = True
                break
    return out


def _old_classifier_prefix(state_dict):
    """
    Return the prefix for old single-classifier head keys, or None.
    Supports 'classifier' and 'head' (e.g. from different codebases).
    """
    if "shared.0.weight" in state_dict:
        return None
    for prefix in ("classifier", "head"):
        if f"{prefix}.3.weight" in state_dict or f"{prefix}.0.weight" in state_dict:
            return prefix
    return None


def _is_old_classifier_format(state_dict):
    """True if checkpoint uses single 'classifier' or 'head' (no shared + class_towers)."""
    return _old_classifier_prefix(state_dict) is not None


def load_state_dict_compat(state_dict, model, strict=True):
    """
    Load checkpoint state_dict into model, remapping old single classifier to
    shared + per-class towers if needed. Use for training resume or inference.

    Old format: classifier/head = Sequential(Linear, ReLU, Dropout, Linear(hidden, num_classes))
    New format: shared + class_towers[i] = ... Linear(hidden, 1)

    Returns:
        (state_dict_to_load, strict_flag) so caller can do:
        model.load_state_dict(state_dict_to_load, strict=strict_flag)
    """
    state_dict = _normalize_state_dict_keys(state_dict)
    new_sd = model.state_dict()
    # Model has old architecture (single classifier): load checkpoint as-is, no remap
    if "classifier.0.weight" in new_sd:
        return state_dict, strict
    prefix = _old_classifier_prefix(state_dict)
    if prefix is None:
        return state_dict, strict

    num_classes = sum(1 for k in new_sd if k.startswith("class_towers.") and k.endswith(".3.weight"))

    for key in list(state_dict.keys()):
        if key.startswith(prefix + "."):
            continue
        if key in new_sd and new_sd[key].shape == state_dict[key].shape:
            new_sd[key] = state_dict[key].clone()

    if f"{prefix}.0.weight" in state_dict:
        new_sd["shared.0.weight"] = state_dict[f"{prefix}.0.weight"].clone()
    if f"{prefix}.0.bias" in state_dict:
        new_sd["shared.0.bias"] = state_dict[f"{prefix}.0.bias"].clone()

    if f"{prefix}.3.weight" in state_dict:
        w = state_dict[f"{prefix}.3.weight"]  # (num_classes, hidden_dim)
        n = min(num_classes, w.shape[0])
        for i in range(n):
            new_sd[f"class_towers.{i}.3.weight"] = w[i : i + 1].clone()
    if f"{prefix}.3.bias" in state_dict:
        b = state_dict[f"{prefix}.3.bias"]
        n = min(num_classes, b.shape[0])
        for i in range(n):
            new_sd[f"class_towers.{i}.3.bias"] = b[i : i + 1].clone()

    # Old classifier was shared -> ReLU -> Dropout -> single Linear(512, 4). New towers are
    # shared -> tower_i = Linear(512,512) -> ReLU -> Dropout -> Linear(512,1). We only load
    # the final layer (class_towers.i.3) from the old classifier. Set the first layer of each
    # tower (class_towers.i.0) to identity so tower_i(h) = class_towers.i.3(ReLU(Dropout(h))),
    # matching the old shared -> ReLU -> Dropout -> output.
    for i in range(num_classes):
        wkey = f"class_towers.{i}.0.weight"
        bkey = f"class_towers.{i}.0.bias"
        if wkey in new_sd and wkey not in state_dict:
            torch.nn.init.eye_(new_sd[wkey])
        if bkey in new_sd and bkey not in state_dict:
            new_sd[bkey].zero_()

    # Old checkpoints (e.g. fwwvq6zy/best.pt) had a single "attention" block and no norm1/norm2.
    # Copy attention -> attention2 and set norms to identity so the model behaves like the old one.
    if "attention.in_proj_weight" in state_dict and "attention2.in_proj_weight" in new_sd:
        if "attention2.in_proj_weight" not in state_dict:
            for key in list(new_sd.keys()):
                if key.startswith("attention2."):
                    old_key = "attention." + key.split(".", 1)[1]
                    if old_key in state_dict and state_dict[old_key].shape == new_sd[key].shape:
                        new_sd[key] = state_dict[old_key].clone()
    if "norm1.weight" in new_sd and "norm1.weight" not in state_dict:
        new_sd["norm1.weight"] = torch.ones_like(new_sd["norm1.weight"])
        new_sd["norm1.bias"] = torch.zeros_like(new_sd["norm1.bias"])
    if "norm2.weight" in new_sd and "norm2.weight" not in state_dict:
        new_sd["norm2.weight"] = torch.ones_like(new_sd["norm2.weight"])
        new_sd["norm2.bias"] = torch.zeros_like(new_sd["norm2.bias"])

    return new_sd, True


def get_state_dict_from_checkpoint(checkpoint):
    """
    Extract model state_dict from a checkpoint dict.
    Handles older or alternate save formats: 'model_state_dict', 'state_dict',
    or a nested 'model' object with .state_dict().
    """
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a dict")
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if "model" in checkpoint:
        obj = checkpoint["model"]
        if hasattr(obj, "state_dict"):
            return obj.state_dict()
    raise KeyError(
        "Checkpoint has no 'model_state_dict', 'state_dict', or 'model' key. "
        f"Keys: {list(checkpoint.keys())}"
    )


def inspect_checkpoint_state_dict(checkpoint):
    """
    Inspect checkpoint state dict keys (after normalization) for diagnostics.
    Returns dict with used_old_compat, sample_keys, has_classifier, has_shared.
    """
    raw = get_state_dict_from_checkpoint(checkpoint)
    normalized = _normalize_state_dict_keys(raw)
    prefix = _old_classifier_prefix(normalized)
    used_compat = prefix is not None
    sample_keys = list(normalized.keys())[:30]
    return {
        "used_old_compat": used_compat,
        "old_head_prefix": prefix,
        "sample_keys": sample_keys,
        "has_classifier": "classifier.0.weight" in normalized or "classifier.3.weight" in normalized,
        "has_head": "head.0.weight" in normalized or "head.3.weight" in normalized,
        "has_shared": "shared.0.weight" in normalized,
        "num_keys": len(normalized),
    }


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

        # Learnable positional embeddings: separate for temporal and spectral dimensions
        self.temporal_pos_embed = torch.nn.Parameter(
            torch.randn(num_temporal, config.embed_dim) * 0.02
        )
        self.spectral_pos_embed = torch.nn.Parameter(
            torch.randn(num_spectral, config.embed_dim) * 0.02
        )

        self.attention = torch.nn.MultiheadAttention(
            embed_dim=config.embed_dim, num_heads=config.num_heads, batch_first=True
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(config.embed_dim, config.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(config.dropout),
            torch.nn.Linear(config.hidden_dim, config.num_classes)
        )

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
            emb = emb + pos_encoding.unsqueeze(0)
        attn_output, _ = self.attention(emb, emb, emb)
        return self.classifier(attn_output)


class BioacousticDetectorBEATS(torch.nn.Module):
    """Bioacoustic detector using BEATS encoder instead of Perch."""

    def __init__(self, config):
        super().__init__()
        self.encoder = BEATSEncoder(config)
        embed_dim = self.encoder.embed_dim
        self.target_frames = config.num_output_frames

        max_seq_len = int(config.window_sec * config.target_sr / 16000 * 50)
        self.temporal_pos_embed = torch.nn.Parameter(
            torch.randn(max_seq_len, embed_dim) * 0.02
        )
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=config.num_heads, batch_first=True
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, config.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(config.dropout),
            torch.nn.Linear(config.hidden_dim, config.num_classes)
        )

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
        emb = emb + pos_encoding
        attn_output, _ = self.attention(emb, emb, emb)
        if seq_len != self.target_frames:
            attn_output = attn_output.transpose(1, 2)
            attn_output = torch.nn.functional.interpolate(
                attn_output, size=self.target_frames, mode='linear', align_corners=False
            )
            attn_output = attn_output.transpose(1, 2)
        output = self.classifier(attn_output)
        return output
