import boto3
from botocore.exceptions import NoCredentialsError, ClientError

def download_bins_from_s3(bucket_name, object_name, file_name):
    """
    Download an object from an S3 bucket.
    
    Parameters:
    - bucket_name (str): The name of the S3 bucket.
    - object_name (str): The S3 object key (i.e., the file path in the bucket).
    - file_name (str): The local path where the file will be saved.
    """
    # Create an S3 client
    s3_client = boto3.client('s3')

    try:
        # Download the file
        s3_client.download_file(bucket_name, object_name, file_name)
        print(f"Successfully downloaded {object_name} from {bucket_name} to {file_name}")
    
    except FileNotFoundError:
        print(f"The file {file_name} was not found.")
    except NoCredentialsError:
        print("Credentials not available.")
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            print(f"The object {object_name} does not exist in bucket {bucket_name}.")
        else:
            print(f"An error occurred: {e}")
