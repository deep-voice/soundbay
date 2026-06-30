"""Download humpback whale recordings from Google Drive to local/S3."""
import argparse
import os
import gdown


MOZAMBIQUE_FOLDER_ID = "1245QCyv2twFnVOsHmBkcz0TvDcNUGPpo"
COSTA_RICA_FOLDER_ID = "1PFJuSEC3fQC0uAcv4bMQgdPa5YACS0HP"


def download_folder(folder_id: str, output_dir: str):
    """Download entire Google Drive folder."""
    os.makedirs(output_dir, exist_ok=True)
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    gdown.download_folder(url, output=output_dir, quiet=False)


def main():
    parser = argparse.ArgumentParser(description="Download humpback data from Google Drive")
    parser.add_argument("--dataset", choices=["mozambique", "costa_rica", "both"], default="both")
    parser.add_argument("--output-dir", default="./data/raw")
    args = parser.parse_args()

    if args.dataset in ("mozambique", "both"):
        print("Downloading Mozambique 2021...")
        download_folder(MOZAMBIQUE_FOLDER_ID, os.path.join(args.output_dir, "mozambique_2021"))

    if args.dataset in ("costa_rica", "both"):
        print("Downloading Costa Rica 2022...")
        download_folder(COSTA_RICA_FOLDER_ID, os.path.join(args.output_dir, "costa_rica_2022"))

    print("Done.")


if __name__ == "__main__":
    main()
