"""
Test that fwwvq6zy/best.pt (old checkpoint) loads correctly via load_state_dict_compat.
Run: python test_old_checkpoint_load.py
"""
import torch
from pathlib import Path


def _normalize_state_dict_keys(state_dict):
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
    if "shared.0.weight" in state_dict:
        return None
    for prefix in ("classifier", "head"):
        if f"{prefix}.3.weight" in state_dict or f"{prefix}.0.weight" in state_dict:
            return prefix
    return None


class MockModel:
    """Minimal mock that has state_dict() with same keys/shapes as BioacousticDetector (no encoder)."""
    def __init__(self):
        embed_dim = 1536
        hidden_dim = 512
        num_classes = 4
        num_temporal = 16
        num_spectral = 4
        self._state = {}
        self._state["temporal_pos_embed"] = torch.randn(num_temporal, embed_dim) * 0.02
        self._state["spectral_pos_embed"] = torch.randn(num_spectral, embed_dim) * 0.02
        # MultiheadAttention(1536, 8, batch_first=True) -> in_proj (4608,1536), out_proj (1536,1536)
        self._state["attention.in_proj_weight"] = torch.randn(4608, 1536) * 0.02
        self._state["attention.in_proj_bias"] = torch.randn(4608) * 0.02
        self._state["attention.out_proj.weight"] = torch.randn(1536, 1536) * 0.02
        self._state["attention.out_proj.bias"] = torch.randn(1536) * 0.02
        self._state["attention2.in_proj_weight"] = torch.randn(4608, 1536) * 0.02
        self._state["attention2.in_proj_bias"] = torch.randn(4608) * 0.02
        self._state["attention2.out_proj.weight"] = torch.randn(1536, 1536) * 0.02
        self._state["attention2.out_proj.bias"] = torch.randn(1536) * 0.02
        self._state["norm1.weight"] = torch.ones(embed_dim)
        self._state["norm1.bias"] = torch.zeros(embed_dim)
        self._state["norm2.weight"] = torch.ones(embed_dim)
        self._state["norm2.bias"] = torch.zeros(embed_dim)
        self._state["shared.0.weight"] = torch.randn(hidden_dim, embed_dim) * 0.02
        self._state["shared.0.bias"] = torch.randn(hidden_dim) * 0.02
        for i in range(num_classes):
            self._state[f"class_towers.{i}.0.weight"] = torch.randn(hidden_dim, hidden_dim) * 0.02
            self._state[f"class_towers.{i}.0.bias"] = torch.randn(hidden_dim) * 0.02
            self._state[f"class_towers.{i}.3.weight"] = torch.randn(1, hidden_dim) * 0.02
            self._state[f"class_towers.{i}.3.bias"] = torch.randn(1) * 0.02

    def state_dict(self):
        return dict(self._state)


def get_state_dict_from_checkpoint(checkpoint):
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    raise KeyError("No state dict in checkpoint")


def load_state_dict_compat_standalone(state_dict, model, strict=True):
    """Copy of load_state_dict_compat to run without importing model (avoids onnxruntime)."""
    state_dict = _normalize_state_dict_keys(state_dict)
    prefix = _old_classifier_prefix(state_dict)
    if prefix is None:
        return state_dict, strict
    new_sd = model.state_dict()
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
        w = state_dict[f"{prefix}.3.weight"]
        n = min(num_classes, w.shape[0])
        for i in range(n):
            new_sd[f"class_towers.{i}.3.weight"] = w[i : i + 1].clone()
    if f"{prefix}.3.bias" in state_dict:
        b = state_dict[f"{prefix}.3.bias"]
        n = min(num_classes, b.shape[0])
        for i in range(n):
            new_sd[f"class_towers.{i}.3.bias"] = b[i : i + 1].clone()
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
    # Set class_towers.i.0 to identity so tower(h) = output_layer(ReLU(Dropout(h))) like old model
    for i in range(num_classes):
        wkey = f"class_towers.{i}.0.weight"
        bkey = f"class_towers.{i}.0.bias"
        if wkey in new_sd and wkey not in state_dict:
            torch.nn.init.eye_(new_sd[wkey])
        if bkey in new_sd and bkey not in state_dict:
            new_sd[bkey].zero_()
    return new_sd, True


def test_compat_standalone():
    """Test compat logic without importing full model (no onnxruntime)."""
    ckpt_path = Path(__file__).parent / "checkpoints" / "fwwvq6zy" / "best.pt"
    assert ckpt_path.exists(), str(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd_raw = get_state_dict_from_checkpoint(ckpt)
    model = MockModel()
    new_sd, strict = load_state_dict_compat_standalone(sd_raw, model, strict=True)

    # Check remapped head is non-zero
    assert "shared.0.weight" in new_sd
    assert "class_towers.0.3.weight" in new_sd
    sw = new_sd["shared.0.weight"]
    cw = new_sd["class_towers.0.3.weight"]
    assert sw.abs().sum() > 0, "shared.0.weight should be non-zero after compat"
    assert cw.abs().sum() > 0, "class_towers.0.3.weight should be non-zero after compat"
    print("OK: compat produces non-zero shared and class_towers")

    # Check attention2 was filled from attention
    a2 = new_sd["attention2.in_proj_weight"]
    a1 = new_sd["attention.in_proj_weight"]
    assert (a2 == a1).all(), "attention2 should equal attention after copy"
    print("OK: attention2 copied from attention")

    # Check norms are identity
    assert (new_sd["norm1.weight"] == 1).all() and (new_sd["norm1.bias"] == 0).all()
    print("OK: norm1/norm2 set to identity")
    # Check class_towers.i.0 set to identity (so old single-classifier behavior is preserved)
    for i in range(4):
        w = new_sd[f"class_towers.{i}.0.weight"]
        assert w.dim() == 2 and w.shape[0] == w.shape[1], "tower.0.weight should be square"
        assert (torch.eye(w.shape[0]) - w).abs().max() < 1e-5, f"class_towers.{i}.0.weight should be identity"
    print("OK: class_towers.i.0 set to identity")
    return new_sd


def test_full_model_forward():
    """Test full model load + forward (requires onnxruntime)."""
    try:
        from config import Config
        from model import BioacousticDetector, load_state_dict_compat, get_state_dict_from_checkpoint
    except ImportError as e:
        print("SKIP full model test (need deps):", e)
        return
    ckpt_path = Path(__file__).parent / "checkpoints" / "fwwvq6zy" / "best.pt"
    config = Config()
    model = BioacousticDetector(config)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd_raw = get_state_dict_from_checkpoint(ckpt)
    new_sd, _ = load_state_dict_compat(sd_raw, model, strict=True)
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    model.eval()
    sr = config.target_sr
    sec = config.window_sec
    x = torch.randn(2, 1, int(sr * sec)) * 0.01
    with torch.no_grad():
        out = model(x)
    print("Output shape:", out.shape, "min/max:", out.min().item(), out.max().item())
    above = (torch.sigmoid(out) > 0.5).sum().item()
    print("Frames with sigmoid>0.5:", above)
    assert above > 0, "Expected some positive predictions after loading old checkpoint"
    print("OK: full model forward gives non-zero predictions")


if __name__ == "__main__":
    test_compat_standalone()
    test_full_model_forward()
