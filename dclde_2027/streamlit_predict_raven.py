"""
Streamlit app: run the bioacoustic detector on WAV files, download Raven selection tables.

Run on the SSH machine (repo root or dclde_2027):
  streamlit run dclde_2027/streamlit_predict_raven.py

Recommended: Copy WAVs to the server first (avoids slow browser upload), then enter
the path in the app. From your computer:
  scp /Users/danielle/Downloads/*.wav user@server:/home/ubuntu/soundbay/dclde_2027/wavs/
Then in the app enter path: /home/ubuntu/soundbay/dclde_2027/wavs
"""

import sys
from pathlib import Path

import streamlit as st

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import Config
from predict_to_raven import (
    load_model,
    load_wav_from_bytes,
    run_inference_on_waveform,
    run_inference_on_file,
    get_class_names,
)
from utils import raven_table_to_tsv


def ensure_config_paths(config):
    """Use paths relative to script dir so the app works on any machine."""
    if not Path(config.onnx_path).is_file():
        config.onnx_path = str(SCRIPT_DIR / "perch_v2.onnx")


@st.cache_resource
def get_model(checkpoint_path, config):
    """Load and cache the model."""
    return load_model(checkpoint_path, config)


st.set_page_config(page_title="WAV → Raven", page_icon="🎵", layout="centered")
st.title("WAV → Raven selection table")
st.markdown("Run the bioacoustic detector and download Raven-compatible `.txt` annotation files.")

# Sidebar: checkpoint and options
with st.sidebar:
    st.subheader("Model & options")
    default_ckpt = SCRIPT_DIR / "checkpoints" / "fwwvq6zy" / "best.pt"
    checkpoint_path = st.text_input(
        "Checkpoint path",
        value=str(default_ckpt),
        help="Path to your trained .pt checkpoint.",
    )
    threshold = st.slider("Detection threshold", 0.1, 0.9, 0.5, 0.05)
    overlap_ratio = st.slider(
        "Window overlap",
        0.0,
        0.9,
        0.5,
        0.1,
        help="50% overlap = each 5s window steps by 2.5s. Improves accuracy; takes longer.",
    )
    max_call_sec = st.number_input(
        "Max call duration (s)",
        min_value=0.0,
        value=1.75,
        step=0.25,
        help="Split longer detections into segments of this length. 0 = no split.",
    )
    class_options = list(Config().label_map.keys())
    classes_to_include = st.multiselect(
        "Classes in Raven table",
        class_options,
        default=class_options,
        help="Only selected classes appear in the downloaded table (e.g. only KW).",
    )
    with st.expander("Min duration (s) per class", expanded=False):
        st.caption("Drop detections shorter than this (e.g. KW &lt; 0.4s). 0 = no filter.")
        min_dur_cols = st.columns(len(class_options))
        min_duration_by_class = {}
        for i, c in enumerate(class_options):
            with min_dur_cols[i]:
                val = st.number_input(
                    c,
                    min_value=0.0,
                    value=0.4 if c == "KW" else 0.0,
                    step=0.1,
                    key=f"min_dur_{c}",
                )
                if val > 0:
                    min_duration_by_class[c] = val
    st.caption("Audio is resampled to 32 kHz (e.g. 96 kHz → 32 kHz) for the model.")

# How to provide files: server path (fast) or browser upload (slow)
st.subheader("Input files")
server_path = st.text_input(
    "Path on this server (recommended — no browser upload)",
    placeholder="e.g. /home/ubuntu/wavs or /home/ubuntu/wavs/file.wav",
    help="Copy your WAVs to the server first with scp/rsync, then enter the path here. Much faster than uploading in the browser.",
)

# Resolve server path to list of WAV files (if any)
server_wavs = []
path_error = None
path_hint = None
if server_path and server_path.strip():
    p = Path(server_path.strip()).expanduser().resolve()
    if p.is_file() and p.suffix.lower() in (".wav", ".wave"):
        server_wavs = [p]
    elif p.is_dir():
        server_wavs = sorted(p.glob("*.wav")) + sorted(p.glob("*.WAV"))
        if not server_wavs:
            path_error = f"No WAV files in folder: {p}"
            path_hint = "Copy your WAVs there first, e.g. from your computer: `scp /path/to/*.wav user@server:/home/ubuntu/soundbay/dclde_2027/wavs/`"
    elif not p.exists():
        path_error = f"Path not found: {p}"
        path_hint = "Create the folder on the server first, then copy WAVs: `mkdir -p /home/ubuntu/soundbay/dclde_2027/wavs` then scp your files."
    else:
        path_error = f"Not a file or folder, or no WAV files: {p}"

use_server_files = len(server_wavs) > 0
if path_error:
    st.warning(path_error)
    if path_hint:
        st.caption(path_hint)
if use_server_files:
    st.success(f"Found **{len(server_wavs)}** WAV file(s). Click **Run detection** below.")
    # Show file list so they see something happened
    with st.expander("Files that will be processed", expanded=False):
        for w in server_wavs[:20]:
            st.text(w.name)
        if len(server_wavs) > 20:
            st.caption(f"... and {len(server_wavs) - 20} more")

# Always show upload as alternative (don't hide it when server path is used)
st.caption("Or upload in the browser (can be slow for large files):")
uploaded = st.file_uploader(
    "Choose WAV file(s)",
    type=["wav", "wave"],
    accept_multiple_files=True,
    help="Upload from your computer. For large files, copy to the server first and use the path above instead.",
)

# We need at least one source: server path with files, or uploaded files
if not use_server_files and not uploaded:
    st.info("👆 Enter a path above (after copying WAVs with scp) or upload files here. Then the **Run detection** button will appear below.")
    st.stop()

# Resolve config and checkpoint
config = Config()
ensure_config_paths(config)
ckpt = Path(checkpoint_path).expanduser().resolve() if checkpoint_path else None

if not ckpt or not ckpt.is_file():
    st.error(f"Checkpoint not found: {checkpoint_path or '(empty)'}. Please set a valid path in the sidebar.")
    st.stop()

with st.spinner("Loading model..."):
    try:
        model, device = get_model(str(ckpt), config)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

st.sidebar.success("Model loaded.")

st.divider()
st.subheader("Run detection")
if st.button("Run detection", type="primary", use_container_width=True):
    results = []

    if use_server_files:
        for wav_path in server_wavs:
            name = wav_path.name
            try:
                detections = run_inference_on_file(
                    wav_path, model, device, config,
                    threshold=threshold,
                    overlap_ratio=overlap_ratio,
                    max_call_sec=max_call_sec if max_call_sec and max_call_sec > 0 else None,
                    min_duration_by_class=min_duration_by_class or None,
                )
                tsv = raven_table_to_tsv(detections, file_path=name)
                results.append((name, detections, tsv))
            except Exception as e:
                st.error(f"Error processing **{name}**: {e}")
                continue
    else:
        for f in uploaded:
            name = f.name
            data = f.read()
            try:
                waveform, _ = load_wav_from_bytes(data, config.target_sr)
                detections = run_inference_on_waveform(
                    waveform, model, device, config,
                    threshold=threshold,
                    overlap_ratio=overlap_ratio,
                    max_call_sec=max_call_sec if max_call_sec and max_call_sec > 0 else None,
                    min_duration_by_class=min_duration_by_class or None,
                )
                tsv = raven_table_to_tsv(detections, file_path=name)
                results.append((name, detections, tsv))
            except Exception as e:
                st.error(f"Error processing **{name}**: {e}")
                continue

    if not results:
        st.warning("No files could be processed.")
        if "predict_raven_results" in st.session_state:
            del st.session_state["predict_raven_results"]
    else:
        st.session_state["predict_raven_results"] = results
        st.success(f"Processed {len(results)} file(s). Download any file below — results stay until you clear them.")
        st.rerun()

# Show persisted results (so downloading one file doesn't clear the others)
if "predict_raven_results" in st.session_state:
    results = st.session_state["predict_raven_results"]
    st.divider()
    st.subheader("Results — download Raven tables")
    if st.button("Clear results", help="Remove results so you can run detection again with different files."):
        del st.session_state["predict_raven_results"]
        st.rerun()

    # Apply class filter from sidebar (filter is applied to what we show and download)
    for i, (name, detections, _tsv_full) in enumerate(results):
        filtered = [d for d in detections if d["Class"] in classes_to_include]
        tsv = raven_table_to_tsv(filtered, file_path=name)
        base = Path(name).stem
        out_filename = f"{base}.Table.1.selections.txt"
        n_total, n_filtered = len(detections), len(filtered)

        with st.expander(f"**{name}** — {n_filtered} selections" + (f" (of {n_total})" if n_filtered != n_total else ""), expanded=True):
            st.download_button(
                label=f"Download {out_filename}",
                data=tsv,
                file_name=out_filename,
                mime="text/plain",
                key=f"dl_{i}_{base}",
            )
            if filtered:
                import pandas as pd
                from io import StringIO
                df = pd.read_csv(StringIO(tsv), sep="\t")
                st.dataframe(df.head(20), use_container_width=True)
                if len(df) > 20:
                    st.caption(f"… and {len(df) - 20} more rows.")
            else:
                st.caption("No detections in selected classes." if n_total else "No detections above threshold.")
