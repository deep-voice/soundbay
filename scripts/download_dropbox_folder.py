import os
import requests

import boto3
import click
import dropbox
from tqdm import tqdm

ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID")
SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
BUCKET_NAME = "deepvoice-user-uploads"
region_name = "us-east-1"
table_name = "user_uploads"


def upload_to_s3(url, s3_client, user_name, folder_name, file_name):
    """
    Upload a file to S3 directly from a URL
    """
    print(f"Downloading file: {file_name}")
    r = requests.get(url, stream=True)

    bucket = s3_client.Bucket(BUCKET_NAME)
    full_path = f"{user_name}/dropbox/{folder_name}/{file_name}"

    # file does not exist
    print("Uploading to S3")
    if 'text' in r.headers.get('Content-Type'): # different treatment to make sure txt files are not compressed
        _ = bucket.put_object(Body=r.content, Key=full_path)
    else:  # better treatment for big files which don't suffer from the compression issues
        _ = bucket.upload_fileobj(r.raw, full_path)
    print("Upload to S3 completed")


def file_exists_in_s3(s3_handler, user_name, folder_name, file_name, file_size):
    bucket = s3_handler.Bucket(BUCKET_NAME)
    full_path = f"{user_name}/dropbox/{folder_name}/{file_name}"
    objs = list(bucket.objects.filter(Prefix=full_path))
    return (len(objs) == 1) and (objs[0].key == full_path) and (objs[0].size == file_size)


def get_dropbox_folder_size(dropbox_handler, dropbox_path):
    total_size = 0
    entries = get_all_entries_from_folder_dropbox(dropbox_handler, dropbox_path)
    for entry in entries:
        if isinstance(entry, dropbox.files.FileMetadata):
            total_size += entry.size
        elif isinstance(entry, dropbox.files.FolderMetadata):
            total_size += get_dropbox_folder_size(dropbox_handler, entry.path_display)
    return total_size


def get_all_entries_from_folder_dropbox(dbx, dropbox_path):
    all_files = []  # collects all files here
    has_more_files = True  # because we haven't queried yet
    cursor = None  # because we haven't queried yet

    while has_more_files:
        if cursor is None:  # if it is our first time querying
            result = dbx.files_list_folder(dropbox_path)
        else:
            result = dbx.files_list_folder_continue(cursor)
        all_files.extend(result.entries)
        cursor = result.cursor
        has_more_files = result.has_more
    return all_files


def get_s3_folder_size(s3_handler, user_name, folder_name):
    bucket = s3_handler.Bucket(BUCKET_NAME)
    full_path = f"{user_name}/dropbox/{folder_name}/"
    objs = list(bucket.objects.filter(Prefix=full_path))
    total_size = 0
    for obj in objs:
        total_size += obj.size
    return total_size


def recursive_dowload_dropbox_folder_to_s3(dropbox_path, dropbox_handler, s3_handler, user_name, folder_name,
                                           path_heirarchy=1):
    """
    Download a file from Dropbox and upload it to S3
    """
    # Iterate through files in the Dropbox folder
    entries = get_all_entries_from_folder_dropbox(dropbox_handler, dropbox_path)
    i = 0
    for i, entry in tqdm(enumerate(entries), desc=f"Downloading {dropbox_path}"):
        if isinstance(entry, dropbox.files.FileMetadata):
            # Generate temporary link for the file
            tqdm.write(str(entry) + "\n")
            link = ""
            # wrap with try catch to avoid error when there is a download issue
            if file_exists_in_s3(s3_handler, user_name, folder_name, entry.name, entry.size):
                tqdm.write(f"File {entry.name} already exists in S3, skipping")
                continue
            else:
                try:
                    link = dropbox_handler.files_get_temporary_link(entry.path_display).link
                except Exception as e:
                    print(f"Error {e}")
                if link != "":
                    # Upload the file to S3 directly from the link
                    upload_to_s3(link, s3_handler, user_name, folder_name, entry.name)
        # In case the entry is a folder, recursively call the function
        elif isinstance(entry, dropbox.files.FolderMetadata):
            path_parts = entry.path_display.split("/")[1:]  # remove the first empty string
            path_parts_to_use = '/'.join(path_parts[path_heirarchy:])
            s3_folder = f'{folder_name}/{path_parts_to_use}'
            try:
                dropbox_folder_size = get_dropbox_folder_size(dropbox_handler, entry.path_display)
                s3_folder_size = get_s3_folder_size(s3_handler, user_name, s3_folder)
            except Exception as e:
                print(f"Error {e}")
                dropbox_folder_size = 1
                s3_folder_size = 0
            if dropbox_folder_size != s3_folder_size:
                recursive_dowload_dropbox_folder_to_s3(entry.path_display, dropbox_handler, s3_handler, user_name,
                                                       s3_folder, path_heirarchy)
            else:
                print(f"Folder {entry.path_display} already exists in S3, skipping")
    if i > 0:
        print(f"Total of {i} files")


@click.command()
@click.option('--dropbox-token', '-d', required=True, type=str,
              help='refer https://app.clickup.com/1861158/v/dc/1rth6-33/1rth6-1282 for details about token generation')
@click.option('--dropbox-path', '-p', required=True, type=str, help='Dropbox folder path')
@click.option('--user-name', '-u', required=True, type=str,
              help='The user name from the dynamodb table, usually the email.')
@click.option('--folder-name', '-f', required=True, type=str,
              help='The folder name in the user uploads bucket')
@click.option('--path-heirarchy', '-h', required=False, type=int, default=1, help='In case the folder name '
                                                                                  'is not the same as the dropbox path '
                                                                                  'but a subfolder of it, specify the '
                                                                                  'path heirarchy level from which we '
                                                                                  'want to keep the naming in the s3 bucket')
def download_dropbox_folder_to_s3(dropbox_token: str, dropbox_path: str, user_name: str, folder_name: str,
                                  path_heirarchy: int = 1):
    # Initialize Dropbox and AWS S3 clients
    dbx = dropbox.Dropbox(dropbox_token)
    s3 = boto3.resource('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)

    # Do the magic
    recursive_dowload_dropbox_folder_to_s3(dropbox_path=dropbox_path,
                                           dropbox_handler=dbx,
                                           s3_handler=s3,
                                           user_name=user_name,
                                           folder_name=folder_name,
                                           path_heirarchy=path_heirarchy)


if __name__ == '__main__':
    download_dropbox_folder_to_s3()
