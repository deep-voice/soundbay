from fvcore.common.config import CfgNode

def load_config_from_yaml(file_path):
    cfg = CfgNode()
    cfg = cfg._load_cfg_from_file(file_obj=open(file_path))

    # # add essential params that are related to the data preprocessing - they affect the density of the pooling layers
    # cfg.AUDIO_DATA.NUM_FRAMES = num_frames
    # cfg.AUDIO_DATA.NUM_FREQUENCIES = num_freqs

    return cfg