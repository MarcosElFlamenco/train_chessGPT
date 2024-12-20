import boto3
import os
import argparse
from tqdm import tqdm

def download_file_from_s3(bucket_name, s3_key, download_path):
    """
    Downloads a file from an S3 bucket to a specified local path with a progress bar.

    Args:
        bucket_name (str): Name of the S3 bucket.
        s3_key (str): S3 object key (path to file in S3).
        download_path (str): Local path where the file will be saved.
    """
    s3_client = boto3.client('s3')
#    os.makedirs(os.path.dirname(download_path), exist_ok=True)

    try:
        # Get the file size for progress bar
        file_info = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        file_size = file_info['ContentLength']

        with open(download_path, 'wb') as f, tqdm(
            total=file_size, unit='B', unit_scale=True, desc=s3_key
        ) as pbar:
            s3_client.download_fileobj(
                bucket_name, s3_key, f,
                Callback=lambda bytes_transferred: pbar.update(bytes_transferred)
            )
        print(f"\nDownloaded s3://{bucket_name}/{s3_key} to {download_path}")
    except Exception as e:
        print(f"Error downloading s3://{bucket_name}/{s3_key}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a file from S3 with progress tracking")
    parser.add_argument('--bucket_name', type=str, required=True, help='S3 bucket name')
    parser.add_argument('--s3_key', type=str, required=True, help='S3 object key for the file to download')
    parser.add_argument('--download_path', type=str, required=True, help='Local path to save the downloaded file')

    args = parser.parse_args()
    print('download path: ', args.download_path)
    download_file_from_s3(args.bucket_name, args.s3_key, args.download_path)
