"""
Streamlit app for comparing multiple checkpoints.

Features:
- Load 2+ checkpoints and compare saved metadata (epoch, val loss, F1, per-class metrics)
- Run inference on the same validation samples with each model and compare metrics
- View spectrograms side-by-side for the same sample across checkpoints
- Audio playback for the selected sample
"""

import io
import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from collections import defaultdict

from config import Config
from model import (
    BioacousticDetector,
    BioacousticDetectorBEATS,
    load_state_dict_compat,
    get_state_dict_from_checkpoint,
    inspect_checkpoint_state_dict,
)
from local_dataset import get_local_dataloaders, download_dataset
from callbacks import SpectrogramVisualizer, CLASS_COLORS


def load_model_from_checkpoint(checkpoint_path, config):
    """Load model from checkpoint. Returns (model, device, checkpoint, load_warning)."""
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    if config.model_type == "beats":
        model = BioacousticDetectorBEATS(config).to(device)
    elif config.model_type == "perch":
        model = BioacousticDetector(config).to(device)
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict_raw = get_state_dict_from_checkpoint(checkpoint)
    state_dict, strict = load_state_dict_compat(state_dict_raw, model)
    load_warning = None
    try:
        model.load_state_dict(state_dict, strict=strict)
    except Exception as e:
        model_keys = set(model.state_dict().keys())
        state_keys = set(state_dict.keys())
        missing = sorted(model_keys - state_keys)
        unexpected = sorted(state_keys - model_keys)
        model.load_state_dict(state_dict, strict=False)
        load_warning = (
            f"Strict load failed ({e}). Loaded with strict=False. "
            f"Missing ({len(missing)}): {missing[:15]}{'...' if len(missing) > 15 else ''}. "
            f"Unexpected ({len(unexpected)}): {unexpected[:15]}{'...' if len(unexpected) > 15 else ''}."
        )
    model.eval()
    return model, device, checkpoint, load_warning


@st.cache_resource
def load_dataset_cached(config):
    """Load validation dataset once."""
    download_dataset()
    _, val_loader = get_local_dataloaders(config)
    return val_loader


def compute_sample_metrics(pred_frames, target_frames, threshold=0.5):
    """Compute TP, FP, FN per sample and per class."""
    pred_bool = (torch.sigmoid(pred_frames) > threshold).float()
    target_bool = (target_frames > 0.5).float()
    frame_mask = target_frames.sum(dim=-1) > 0
    tp = ((pred_bool * target_bool) * frame_mask.unsqueeze(-1)).sum(dim=(0, 1))
    fp = ((pred_bool * (1 - target_bool)) * frame_mask.unsqueeze(-1)).sum(dim=(0, 1))
    fn = (((1 - pred_bool) * target_bool) * frame_mask.unsqueeze(-1)).sum(dim=(0, 1))
    tp_np = tp.cpu().numpy().flatten()
    fp_np = fp.cpu().numpy().flatten()
    fn_np = fn.cpu().numpy().flatten()
    return {
        "tp": tp_np,
        "fp": fp_np,
        "fn": fn_np,
        "has_fp": fp_np.sum() > 0,
        "has_fn": fn_np.sum() > 0,
        "has_correct": tp_np.sum() > 0 and fp_np.sum() == 0 and fn_np.sum() == 0,
    }


def get_classes_present(target_frames):
    """Get list of class indices that have annotations."""
    class_sums = target_frames.sum(dim=0)
    return [i for i, count in enumerate(class_sums) if count > 0]


def get_fixed_samples(dataloader, num_samples):
    """Collect a fixed set of validation samples (audio + targets, no predictions)."""
    samples = []
    for batch in dataloader:
        if len(samples) >= num_samples:
            break
        batch_size = batch["audio"].shape[0]
        window_starts = batch["window_start"]
        window_ends = batch["window_end"]
        if isinstance(window_starts, torch.Tensor):
            window_starts = window_starts.cpu().tolist()
        if isinstance(window_ends, torch.Tensor):
            window_ends = window_ends.cpu().tolist()
        for i in range(batch_size):
            if len(samples) >= num_samples:
                break
            target_frames = batch["target_frames"][i]
            samples.append({
                "audio": batch["audio"][i],
                "target_frames": target_frames,
                "window_start": window_starts[i] if not isinstance(window_starts[i], (int, float)) else float(window_starts[i]),
                "window_end": window_ends[i] if not isinstance(window_ends[i], (int, float)) else float(window_ends[i]),
                "gcs_url": batch["gcs_url"][i],
                "annotations": batch["annotations"][i],
                "classes_present": get_classes_present(target_frames),
            })
    return samples


def run_models_on_samples(models_devices, base_samples, threshold, class_names):
    """
    Run each model on the same base samples. Return dict: checkpoint_key -> list of result dicts.
    Each result dict has pred_frames, tp, fp, fn, has_fp, has_fn, has_correct.
    """
    results = {}
    for ckpt_key, (model, device) in models_devices.items():
        preds_list = []
        with torch.no_grad():
            for sample in base_samples:
                audio = sample["audio"].unsqueeze(0).to(device)
                targets = sample["target_frames"].unsqueeze(0).to(device)
                out = model(audio)
                metrics = compute_sample_metrics(out, targets, threshold)
                preds_list.append({
                    "pred_frames": out.squeeze(0).cpu(),
                    **metrics,
                })
        results[ckpt_key] = preds_list

    return results


def aggregate_metrics(results_per_checkpoint, class_names):
    """Aggregate TP/FP/FN across samples per checkpoint, then compute P/R/F1."""
    summary = {}
    for ckpt_key, result_list in results_per_checkpoint.items():
        n_classes = len(class_names)
        tp_sum = np.zeros(n_classes)
        fp_sum = np.zeros(n_classes)
        fn_sum = np.zeros(n_classes)
        for r in result_list:
            tp_sum += r["tp"]
            fp_sum += r["fp"]
            fn_sum += r["fn"]
        precision = tp_sum / (tp_sum + fp_sum + 1e-8)
        recall = tp_sum / (tp_sum + fn_sum + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        macro_f1 = float(np.mean(f1))
        summary[ckpt_key] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "macro_f1": macro_f1,
            "tp": tp_sum,
            "fp": fp_sum,
            "fn": fn_sum,
        }
    return summary


def checkpoint_metadata_table(checkpoints_info, class_names):
    """Build a DataFrame from saved checkpoint metadata (epoch, val_metrics, best_f1)."""
    rows = []
    for label, ckpt in checkpoints_info.items():
        if ckpt is None:
            rows.append({"Checkpoint": label, "epoch": None, "train_loss": None, "val_loss": None, "val_macro_f1": None, "best_f1": None})
            continue
        epoch = ckpt.get("epoch")
        train_loss = ckpt.get("train_loss")
        val_metrics = ckpt.get("val_metrics") or {}
        val_loss = val_metrics.get("loss")
        val_macro_f1 = val_metrics.get("macro_f1")
        best_f1 = ckpt.get("best_f1")
        row = {
            "Checkpoint": label,
            "epoch": epoch,
            "train_loss": float(train_loss) if train_loss is not None else None,
            "val_loss": float(val_loss) if val_loss is not None else None,
            "val_macro_f1": float(val_macro_f1) if val_macro_f1 is not None else None,
            "best_f1": float(best_f1) if best_f1 is not None else None,
        }
        for i, name in enumerate(class_names):
            p = val_metrics.get("precision")
            r = val_metrics.get("recall")
            f = val_metrics.get("f1")
            if p is not None and i < len(p):
                row[f"val_P_{name}"] = float(p[i].item() if torch.is_tensor(p[i]) else p[i])
            if r is not None and i < len(r):
                row[f"val_R_{name}"] = float(r[i].item() if torch.is_tensor(r[i]) else r[i])
            if f is not None and i < len(f):
                row[f"val_F1_{name}"] = float(f[i].item() if torch.is_tensor(f[i]) else f[i])
        rows.append(row)
    return pd.DataFrame(rows)


def audio_to_wav_bytes(audio_tensor, sample_rate):
    """Convert audio tensor to WAV bytes for Streamlit."""
    try:
        import torchaudio
        if audio_tensor.dim() == 2:
            audio_tensor = audio_tensor.mean(dim=0, keepdim=True)
        elif audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        audio_np = audio_tensor.squeeze().numpy()
        max_abs = np.abs(audio_np).max()
        if max_abs > 1e-8:
            audio_np = audio_np / max_abs
        buffer = io.BytesIO()
        torchaudio.save(buffer, torch.from_numpy(audio_np).unsqueeze(0), sample_rate, format="wav")
        buffer.seek(0)
        return buffer.read()
    except ImportError:
        audio_np = audio_tensor.numpy() if hasattr(audio_tensor, "numpy") else np.array(audio_tensor)
        if audio_np.ndim == 2:
            audio_np = audio_np.mean(axis=0)
        max_abs = np.abs(audio_np).max()
        if max_abs > 1e-8:
            audio_np = audio_np / max_abs
        return audio_np


def main():
    st.set_page_config(page_title="Checkpoint Compare", layout="wide")
    st.title("Checkpoint comparison")
    st.markdown("Compare 2+ checkpoints: saved metrics, inference on same samples, and spectrograms.")

    config = Config()
    class_names = list(config.label_map.keys())

    # Sidebar: checkpoint paths and labels
    st.sidebar.header("Checkpoints")
    num_ckpts = st.sidebar.number_input("Number of checkpoints", min_value=2, max_value=6, value=2, step=1)
    checkpoint_paths = []
    checkpoint_labels = []
    for i in range(num_ckpts):
        path = st.sidebar.text_input(
            f"Checkpoint {i+1} path",
            value="/home/ubuntu/soundbay/dclde_2027/checkpoints/fwwvq6zy/best.pt" if i == 0 else "",
            key=f"ckpt_path_{i}",
        )
        label = st.sidebar.text_input(f"Label (optional) {i+1}", value=f"ckpt_{i+1}", key=f"ckpt_label_{i}")
        checkpoint_paths.append(path)
        checkpoint_labels.append(label or f"ckpt_{i+1}")

    threshold = st.sidebar.slider("Classification threshold", 0.0, 1.0, 0.5, 0.05)
    num_samples = st.sidebar.number_input("Number of samples for inference", 50, 1000, 200, 50)

    if st.sidebar.button("Load checkpoints & dataset", type="primary"):
        with st.spinner("Loading..."):
            try:
                download_dataset()
                val_loader = load_dataset_cached(config)
                models_devices = {}
                checkpoints_info = {}
                load_warnings = {}
                load_diagnostics = {}
                for path, label in zip(checkpoint_paths, checkpoint_labels):
                    if not path or not Path(path).exists():
                        st.sidebar.warning(f"Missing or invalid path: {path}")
                        continue
                    model, device, ckpt, load_warning = load_model_from_checkpoint(path, config)
                    key = label
                    models_devices[key] = (model, device)
                    checkpoints_info[key] = ckpt
                    load_warnings[key] = load_warning
                    load_diagnostics[key] = inspect_checkpoint_state_dict(ckpt)
                if len(models_devices) < 2:
                    st.sidebar.error("Need at least 2 valid checkpoints to compare.")
                else:
                    st.session_state["models_devices"] = models_devices
                    st.session_state["checkpoints_info"] = checkpoints_info
                    st.session_state["load_warnings"] = load_warnings
                    st.session_state["load_diagnostics"] = load_diagnostics
                    st.session_state["val_loader"] = val_loader
                    st.session_state["config"] = config
                    st.session_state["visualizer"] = SpectrogramVisualizer(config)
                    st.sidebar.success(f"Loaded {len(models_devices)} checkpoints.")
            except Exception as e:
                st.sidebar.error(str(e))
                st.exception(e)

    if "models_devices" not in st.session_state or "checkpoints_info" not in st.session_state:
        st.info("Add 2+ checkpoint paths in the sidebar and click **Load checkpoints & dataset**.")
        return

    models_devices = st.session_state["models_devices"]
    checkpoints_info = st.session_state["checkpoints_info"]
    val_loader = st.session_state["val_loader"]
    config = st.session_state["config"]
    visualizer = st.session_state["visualizer"]
    ckpt_labels = list(checkpoints_info.keys())
    load_warnings = st.session_state.get("load_warnings", {})
    load_diagnostics = st.session_state.get("load_diagnostics", {})

    with st.expander("Load diagnostics (why a checkpoint might show zeros)"):
        for label in ckpt_labels:
            st.markdown(f"**{label}**")
            diag = load_diagnostics.get(label, {})
            if diag:
                st.caption(f"Old-classifier compat: {diag.get('used_old_compat', '?')} (prefix: {diag.get('old_head_prefix', 'N/A')}) | classifier: {diag.get('has_classifier')} | head: {diag.get('has_head')} | shared: {diag.get('has_shared')} | Keys: {diag.get('num_keys', 0)}")
                st.code(" ".join(diag.get("sample_keys", [])[:20]) + (" ..." if len(diag.get("sample_keys", [])) > 20 else ""), language=None)
            if load_warnings.get(label):
                st.warning(load_warnings[label])
            st.divider()

    # --- Tab: Metadata | Inference metrics | Spectrograms
    tab_meta, tab_inference, tab_spec = st.tabs(["Checkpoint metadata", "Inference metrics (same samples)", "Spectrogram comparison"])

    with tab_meta:
        st.subheader("Saved checkpoint metrics")
        df_meta = checkpoint_metadata_table(checkpoints_info, class_names)
        st.dataframe(df_meta, use_container_width=True, hide_index=True)
        st.caption("Values from the last validation run saved in each checkpoint (epoch, train_loss, val_metrics, best_f1).")

    with tab_inference:
        if st.button("Run inference on same samples", type="primary", key="run_inf"):
            with st.spinner("Collecting samples and running all models..."):
                base_samples = get_fixed_samples(val_loader, num_samples)
                if len(base_samples) < num_samples:
                    st.warning(f"Only {len(base_samples)} samples available.")
                results_per_ckpt = run_models_on_samples(models_devices, base_samples, threshold, class_names)
                summary = aggregate_metrics(results_per_ckpt, class_names)
                st.session_state["base_samples"] = base_samples
                st.session_state["results_per_ckpt"] = results_per_ckpt
                st.session_state["inference_summary"] = summary
                st.session_state["inference_done"] = True
                st.success(f"Inference done on {len(base_samples)} samples.")

        if st.session_state.get("inference_done"):
            summary = st.session_state["inference_summary"]
            rows = []
            for ckpt_key in summary:
                s = summary[ckpt_key]
                row = {"Checkpoint": ckpt_key, "macro_f1": s["macro_f1"]}
                for i, name in enumerate(class_names):
                    row[f"P_{name}"] = s["precision"][i]
                    row[f"R_{name}"] = s["recall"][i]
                    row[f"F1_{name}"] = s["f1"][i]
                rows.append(row)
            df_inf = pd.DataFrame(rows)
            st.dataframe(df_inf, use_container_width=True, hide_index=True)
            st.caption("Metrics aggregated over the same validation samples for each checkpoint.")
        else:
            st.info("Click **Run inference on same samples** to compare metrics on the same data.")

    with tab_spec:
        base_samples = st.session_state.get("base_samples", [])
        results_per_ckpt = st.session_state.get("results_per_ckpt", {})

        if not base_samples or not results_per_ckpt:
            st.info("Run inference in the **Inference metrics** tab first to enable spectrogram comparison.")
        else:
            sample_idx = st.selectbox(
                "Sample index",
                range(len(base_samples)),
                format_func=lambda i: f"{i}: {Path(base_samples[i]['gcs_url']).name} ({base_samples[i]['window_start']:.2f}s–{base_samples[i]['window_end']:.2f}s)",
            )
            sample = base_samples[sample_idx]

            st.subheader("Ground truth + predictions per checkpoint")
            st.audio(audio_to_wav_bytes(sample["audio"], config.target_sr), format="audio/wav")

            # One row per checkpoint: spectrogram with GT and that checkpoint's predictions
            for ckpt_key in ckpt_labels:
                if ckpt_key not in results_per_ckpt:
                    continue
                pred_result = results_per_ckpt[ckpt_key][sample_idx]
                fig = visualizer.plot_spectrogram_with_predictions(
                    waveform=sample["audio"],
                    target_frames=sample["target_frames"],
                    pred_frames=pred_result["pred_frames"],
                    window_start=sample["window_start"],
                    window_end=sample["window_end"],
                    gcs_url=sample["gcs_url"],
                )
                st.markdown(f"**{ckpt_key}**")
                st.pyplot(fig)
                plt.close(fig)

                # Mini metrics for this sample for this checkpoint
                with st.expander(f"Sample metrics — {ckpt_key}"):
                    for class_idx, name in enumerate(class_names):
                        tp = float(pred_result["tp"][class_idx])
                        fp = float(pred_result["fp"][class_idx])
                        fn = float(pred_result["fn"][class_idx])
                        if tp + fp + fn > 0:
                            p = tp / (tp + fp + 1e-8)
                            r = tp / (tp + fn + 1e-8)
                            f1 = 2 * p * r / (p + r + 1e-8)
                            st.caption(f"{name}: P={p:.2f} R={r:.2f} F1={f1:.2f} (TP={int(tp)} FP={int(fp)} FN={int(fn)})")
                    if pred_result["has_correct"]:
                        st.success("All correct")
                    if pred_result["has_fp"]:
                        st.error("Has false positives")
                    if pred_result["has_fn"]:
                        st.error("Has false negatives")


if __name__ == "__main__":
    main()
