import os
import shutil
from soundbay.inference import inference_main
from hydra import initialize, compose
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from tqdm import tqdm
import subprocess




# python inference.py -cn runs/inference_single_audio data.test_dataset.file_path="../../../../opt/dlami/nvme/FlumeL_m_23.04.24-23.05.09_wav_shaye_tudor/" experiment.checkpoint.path=/home/ubuntu/soundbay/checkpoints/wsc271eu/best.pth data.data_sample_rate=16000 experiment.save_raven=yes
def main():

    list_of_folders = [
        '2024-01-31 22.33.14 - FlumeL_m_23.05.09-23.05.23_wav shaye tudor/',
                           '2024-02-03 01.32.35 - FlumeL_m_23.05.24-23.06.06_wav shaye tudor/',
                           '2024-08-13 17.29.17 - FlumeL_mixed_23.02.06-23.02.20 Kelsie Murchy/',
                           '2024-08-15 17.54.36 - FlumeL_mixed_23.02.21-23.03.09 Kelsie/',
                           'FlumeL_m_23.03.26-23.04.11_wav shaye tudor/',
                           'FlumeL_m_23.04.11-23.04.23_wav shaye tudor/',
                           'FlumeL_m_23.04.24-23.05.09_wav shaye tudor/',
                           'FlumeL_m_23.05.24-23.06.06_wav shaye tudor/',
                           'FlumeL_m_23.07.03-23.07.16.wav shaye tudor/',
                           'FlumeL_m_23.07.17-23.07.31.wav shaye tudor',
                           'LOOT_23.03.10-23.03.28_wav shaye tudor/',
                           'LOOT_23.03.29-23.04.12_wav shaye tudor/']


    
    for dir in tqdm(list_of_folders):

        dir_name = f'/opt/dlami/nvme/{dir}'
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
            aws_cmd = f'aws s3 cp "S3_FOLDER/{dir}" "{dir_name}" --recursive'

            # cmd = ["aws", "s3", "cp", f"S3_FOLDER/{dir}/", local_path, "--recursive"]
            result = subprocess.run(aws_cmd,shell=True, capture_output=True)
            # os.system(aws_cmd)

            if result.returncode == 0:
                print("Download successful")
                print(result.stdout)   # captured standard output
            else:
                print("Error occurred:", result.stderr)

        else:
            print(f'{dir_name} exists!')

        with initialize(config_path="soundbay/conf", job_name="programmatic", version_base="1.2"):
            cfg = compose(
                config_name="runs/inference_single_audio.yaml",
                overrides=[
                    f"experiment.checkpoint.path=/home/ubuntu/soundbay/checkpoints/p4phc42h/best.pth",
                    f'data.test_dataset.file_path={dir_name}',
                    "data.data_sample_rate=16000",
                    "experiment.save_raven=yes"
                ],
                return_hydra_config=True,   # <- important
            )

            HydraConfig.instance().set_config(cfg)
            concat_dataset = inference_main.__wrapped__(cfg)
            shutil.rmtree(dir_name)



        
        # python inference.py -cn runs/inference_single_audio data.test_dataset.file_path="../../../../opt/dlami/nvme/FlumeL_m_23.04.24-23.05.09_wav_shaye_tudor/" experiment.checkpoint.path=/home/ubuntu/soundbay/checkpoints/wsc271eu/best.pth data.data_sample_rate=16000 experiment.save_raven=yes



    

    return








if __name__ == "__main__":
    main()