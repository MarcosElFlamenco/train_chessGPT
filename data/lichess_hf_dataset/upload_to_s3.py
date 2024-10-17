import os
import boto3
import argparse
from botocore.exceptions import NoCredentialsError, ClientError
from tqdm import tqdm  # Optional: For displaying a progress bar

def upload_directory_to_s3(local_directory, bucket_name, s3_directory='', aws_region='us-east-1'):
    """
    Uploads a local directory to an S3 bucket.

    :param local_directory: Path to the local directory to upload.
    :param bucket_name: Name of the target S3 bucket.
    :param s3_directory: (Optional) S3 directory path within the bucket.
    :param aws_region: AWS region where the bucket is located.
    """
    # Initialize the S3 client
    s3_client = boto3.client('s3', region_name=aws_region)

    # Calculate total number of files for the progress bar
    total_files = sum(len(files) for _, _, files in os.walk(local_directory))
    progress_bar = tqdm(total=total_files, desc="Uploading", unit="file")

    try:
        for root, dirs, files in os.walk(local_directory):
            for file in files:
                local_path = os.path.join(root, file)

                # Compute the relative path to maintain directory structure in S3
                relative_path = os.path.relpath(local_path, local_directory)
                s3_path = os.path.join(s3_directory, relative_path).replace("\\", "/")  # For Windows compatibility

                try:
                    s3_client.upload_file(local_path, bucket_name, s3_path)
                except ClientError as e:
                    print(f"Failed to upload {local_path} to {s3_path}: {e}")
                except FileNotFoundError:
                    print(f"The file {local_path} was not found.")
                
                progress_bar.update(1)
    except NoCredentialsError:
        print("AWS credentials not found. Please configure your credentials.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        progress_bar.close()

    print("Upload completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Upload files from a local directory to a Hugging Face dataset repository with progress bars.')
    parser.add_argument('--local_dir', type=str, required=True, help='Path to the local source directory containing files to upload.')
    parser.add_argument('--bucket', type=str, required=True, help='Hugging Face repository ID (e.g., username/repo_name). If the repository does not exist, it will be created as a dataset repository.')
    parser.add_argument('--s3_dir', type=str, default=None, help='Hugging Face access token. If not provided, the script will look for the HF_TOKEN environment variable.')
    args = parser.parse_args()


    # Example usage
    local_dir = args.local_dir
    bucket = args.bucket
    s3_dir = args.s3_dir

    upload_directory_to_s3(local_dir, bucket, s3_dir)
