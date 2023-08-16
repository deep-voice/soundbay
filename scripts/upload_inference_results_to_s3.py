from botocore.exceptions import NoCredentialsError
from boto3.dynamodb.conditions import Key
import boto3
import os
import click
from tqdm import tqdm
import sys
import json

# Please configure your own keys on aws cli
ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID")
SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
bucket_name = "deepvoice-user-uploads"
region_name = "us-east-1"
table_name = "user_uploads"

try:
    sts = boto3.client(
        "sts", aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY
    )
    sts.get_caller_identity()
except Exception as e:
    print(f"Error connecting to aws, {e}")
    sys.exit()

dynamo_db = boto3.resource("dynamodb", region_name=region_name)
user_uploads_table = dynamo_db.Table(table_name)
upload_options = ["app", "dropbox"]


def verify_file_exists(name, user_email, upload_source):
    key = f"{user_email}/{upload_source}/{name}"

    dynamo_db_response = user_uploads_table.query(
        KeyConditionExpression=Key("file_key").eq(key)
    )

    if len(dynamo_db_response["Items"]) == 0:
        return False
    return True


def upload_to_aws(local_file, bucket, s3_file, overwrite):
    s3 = boto3.client(
        "s3", aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY
    )
    print(f"Uploading to path {s3_file}")
    files_exists = False
    results = s3.list_objects(Bucket=bucket, Prefix=s3_file)
    if "Contents" in results:
        if len(results["Contents"]) > 0:
            for file in results["Contents"]:
                if file["Key"] == s3_file:
                    files_exists = True

    if files_exists and not overwrite:
        print(f"File {s3_file} already exists, use --overwrite to upload anyway")
        return
    try:
        with tqdm(
            total=os.stat(local_file).st_size,
            unit="B",
            unit_scale=True,
            desc=local_file,
        ) as pbar:
            s3.upload_file(
                Filename=local_file,
                Bucket=bucket,
                Key=s3_file,
                Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
            )

        print("Upload Successful")
        return
    except FileNotFoundError:
        print(f"File {local_file} not found")
        return
    except NoCredentialsError:
        print("AWS credentials not available")
        return


def update_db(path, user_email, upload_source, s3_file_name):
    file_key = f"{user_email}/{upload_source}/{path}"
    lambda_client = boto3.client('lambda') 

    response = user_uploads_table.update_item(
        Key={"file_key": file_key},
        UpdateExpression="set response_file_key=:s",
        ExpressionAttributeValues={":s": s3_file_name},
        ReturnValues="UPDATED_NEW",
    )

    if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
        print(f"Error in updating DB, response: {response}")


    update_status_params = {"file_key":file_key, "new_status": "Completed"}

    lambda_client.invoke(
            FunctionName = 'arn:aws:lambda:us-east-1:890799267054:function:UpdateStatus',
            InvocationType = 'Event', 
            Payload = json.dumps(update_status_params)
        ) 

    if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
        print(f"Error in updating DB, response: {response}")


@click.command()
@click.option("--path", type=str, default="", help="Path to results file")
@click.option("--user-email", type=str, help="Uploading user(client/biologist) email")
@click.option(
    "--upload-source",
    type=click.Choice(upload_options),
    help="Original file upload source(dropbox/web(app))",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Upload the results file without validating that the original file exists",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Results file already exists on S3, use this flag to overwrite existing file",
)
def main(path, user_email, upload_source, force, overwrite):
    """
    This scripts uploads results file to S3.
    It firsts check that the original file (from the specified user and upload source) exists in the DB.
    Than uploads your file name to S3 and results_ prefix
    You can use force in order to upload results for non-existing file
    """

    if not force:
        if not verify_file_exists(os.path.basename(path), user_email, upload_source):
            print(
                f"File with params {path, user_email,upload_source} doens't exists, use --force to upload response for non-existing input file"
            )
            return

    s3_file_name = os.path.join(user_email, "results_" + os.path.basename(path))

    update_db(os.path.basename(path), user_email, upload_source, s3_file_name)

    upload_to_aws(path, bucket_name, s3_file_name, overwrite)


if __name__ == "__main__":
    main()
