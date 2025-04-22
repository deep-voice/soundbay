python inference.py \
    --config-name runs/inference_single_audio \
        data.test_dataset.file_path="/home/ubuntu/amber_data/" \
        data.test_dataset.data_sample_rate=96000\
        data.num_workers=6 \
        experiment.save_raven=True\
        experiment.checkpoint.path=/home/ubuntu/ambers_ckpts/${1}/last.pth\
        experiment.threshold=${2} \
