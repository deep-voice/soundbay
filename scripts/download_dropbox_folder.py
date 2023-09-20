import os
import requests
import re

import boto3
import click
import dropbox
from tqdm import tqdm

ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID")
SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
BUCKET_NAME = "deepvoice-user-uploads"
region_name = "us-east-1"
table_name = "user_uploads"


def upload_to_s3(url, s3_client, user_name, folder_name):
    """
    Upload a file to S3 directly from a URL
    """
    print(f"Downloading url: {url}")

    # Get File name
    head_response = requests.head(url)
    file_name = re.findall("filename=\"(.+)\"", head_response.headers.get('content-disposition'))[0]

    print(f"Downloading file: {file_name}")
    r = requests.get(url, stream=True)

    bucket = s3_client.Bucket(BUCKET_NAME)
    response_s3 = bucket.upload_fileobj(r.raw, f"{user_name}/dropbox/{folder_name}/{file_name}")

    print("Upload to S3 completed")


@click.command()
@click.option('--dropbox-token', '-d', required=True, type=str,
              help='refer https://app.clickup.com/1861158/v/dc/1rth6-33/1rth6-1282 for details about token generation')
@click.option('--dropbox-path', '-p', required=True, type=str, help='Dropbox folder path')
@click.option('--user-name', '-u', required=True, type=str,
              help='The user name from the dynamodb table, usually the email.')
@click.option('--folder-name', '-f', required=True, type=str,
              help='The folder name in the user uploads bucket')
def download_dropbox_folder_to_s3(dropbox_token: str, dropbox_path: str, user_name: str, folder_name: str):
    # Initialize Dropbox and AWS S3 clients
    dbx = dropbox.Dropbox(dropbox_token)
    s3 = boto3.resource('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)

    # Get file metadata from Dropbox folder
    entries = dbx.files_list_folder(dropbox_path).entries
    # Iterate through files in the Dropbox folder
    for i, entry in tqdm(enumerate(entries)):
        if isinstance(entry, dropbox.files.FileMetadata):
            # Generate temporary link for the file
            print(str(entry) + "\n")
            link = ""
            # wrap with try catch to avoid error when there is a download issue
            try:
                link = dbx.files_get_temporary_link(entry.path_display).link
            except Exception as e:
                print(f"Error {e}")
            if link != "":
                # Upload the file to S3 directly from the link
                upload_to_s3(link, s3, user_name, folder_name)
    print(f"Total of {i} files")

if __name__ == '__main__':
    download_dropbox_folder_to_s3()
