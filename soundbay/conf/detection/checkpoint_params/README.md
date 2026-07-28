# Spectrogram params per checkpoint

Inference must slice spectrograms exactly the way the checkpoint was trained, or
boxes get decoded against the wrong time-frequency grid (see the recall collapse
investigated on 2026-07-28).

`soundbay/detection/inference.py` reads these automatically when the JSON is copied
next to the weights as `<weights-stem>.spectrogram.json`, e.g.

    cp soundbay/conf/detection/checkpoint_params/cr_finetune.spectrogram.json \
       runs/detect/humpback_v2/cr_finetune/weights/best.spectrogram.json

`prepare_dataset.py` writes a `spectrogram_params.json` into every dataset it builds;
that file is the authoritative source for models trained on that dataset.

| Checkpoint | chunk_duration | freq range | Notes |
|-----------|----------------|-----------|-------|
| `humpback_v2/cr_finetune` | 5.0s | 50–3000 Hz | Recovered by decoding training labels; dataset `/data/humpback_yolo_cr` |
| `humpback_detection/mozambique_yolo11m` | unverified | unverified | Assumed the 15s/50–4000Hz defaults, NOT confirmed |
| `humpback_v2/combined_3000hz` | unverified | likely 50–3000 Hz | Name suggests 3000 Hz; confirm before use |
| `humpback_v2/yolo11m_5s_1000hz` | likely 5.0s | likely 50–1000 Hz | Name suggests both; confirm before use |
