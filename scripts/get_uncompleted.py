from botocore.exceptions import NoCredentialsError
from boto3.dynamodb.conditions import Key
import boto3
import os
import click
from pprint import pprint
from boto3.dynamodb.conditions import Attr    


# Please configure your own keys on aws cli
ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID")
SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
bucket_name = "deepvoice-user-uploads"
region_name = "us-east-1"
table_name = "user_uploads"


dynamo_db = boto3.resource('dynamodb', region_name=region_name)
table = dynamo_db.Table(table_name)

scan_response = table.scan(
   FilterExpression=Attr('upload_status').ne("Completed")
)

uploads = scan_response["Items"]

pprint(uploads)