#!/usr/bin/env python3
"""
upload_to_hf_dataset.py

A script to upload all files from a specified local directory to a Hugging Face dataset repository.
Displays a progress bar for each file being uploaded.

Usage:
    python upload_to_hf_dataset.py --source_dir /path/to/source --repo_id username/repo_name [--token your_hf_token] [--recursive]
"""

import os
import argparse
from tqdm import tqdm
from huggingface_hub import HfApi, Repository, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

class TqdmBufferedReader:
    """
    A file-like object that wraps another file object and updates a tqdm progress bar as data is read.
    """
    def __init__(self, file_path, tqdm_desc, chunk_size=1024*1024):
        self.file = open(file_path, 'rb')
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.file_size = os.path.getsize(file_path)
        self.tqdm_bar = tqdm(total=self.file_size, unit='B', unit_scale=True, desc=tqdm_desc, leave=True)
    
    def read(self, size=-1):
        data = self.file.read(size if size > 0 else self.chunk_size)
        self.tqdm_bar.update(len(data))
        return data
    
    def __getattr__(self, attr):
        return getattr(self.file, attr)
    
    def close(self):
        self.file.close()
        self.tqdm_bar.close()

def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Upload files from a local directory to a Hugging Face dataset repository with progress bars.')
    parser.add_argument('--source_dir', type=str, required=True, help='Path to the local source directory containing files to upload.')
    parser.add_argument('--repo_id', type=str, required=True, help='Hugging Face repository ID (e.g., username/repo_name). If the repository does not exist, it will be created as a dataset repository.')
    parser.add_argument('--token', type=str, default=None, help='Hugging Face access token. If not provided, the script will look for the HF_TOKEN environment variable.')
    parser.add_argument('--recursive', action='store_true', help='Recursively upload files in subdirectories.')
    return parser.parse_args()

def get_all_files(source_dir, recursive=False):
    """
    Retrieve all file paths from the source directory.
    
    Args:
        source_dir (str): Path to the source directory.
        recursive (bool): Whether to traverse subdirectories.
    
    Returns:
        list: List of file paths.
    """
    all_files = []
    if recursive:
        for root, dirs, files in os.walk(source_dir):
            for filename in files:
                filepath = os.path.join(root, filename)
                all_files.append(filepath)
    else:
        for filename in os.listdir(source_dir):
            filepath = os.path.join(source_dir, filename)
            if os.path.isfile(filepath):
                all_files.append(filepath)
    return all_files

def upload_file(api, repo_id, file_path, s3_key, token):
    """
    Upload a single file to a Hugging Face dataset repository with a progress bar.
    
    Args:
        api (HfApi): An instance of HfApi.
        repo_id (str): The repository ID on Hugging Face (e.g., "username/repo_name").
        file_path (str): Path to the local file to upload.
        s3_key (str): The destination path in the repository.
        token (str): Hugging Face access token.
    
    Returns:
        bool: True if upload succeeded, False otherwise.
    """
    filename = os.path.basename(file_path)
    tqdm_desc = f'Uploading {filename}'
    progress_file = TqdmBufferedReader(file_path, tqdm_desc)
    
    try:
        api.upload_file(
            path_or_fileobj=progress_file,
            path_in_repo=s3_key,
            repo_id=repo_id,
            repo_type='dataset',
            token=token,
            create_pr=False,
            commit_message=f'Add {filename}'
        )
        print(f"Successfully uploaded '{filename}' to repository '{repo_id}'.")
        return True
    except Exception as e:
        print(f"Error uploading '{filename}': {e}")
        return False
    finally:
        progress_file.close()

def main():
    print('main started')
    """
    Main function to handle uploading files to Hugging Face dataset repository.
    """
    args = parse_arguments()
    source_dir = args.source_dir
    repo_id = args.repo_id
    token = args.token or os.getenv('HF_TOKEN')
    recursive = args.recursive


    if not token:
        print("Error: Hugging Face token not provided. Use --token or set HF_TOKEN environment variable.")
        exit(1)
    
    if not os.path.isdir(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist or is not a directory.")
        exit(1)
    
    api = HfApi()
    try:
        user_info = api.whoami(token=token)
        print(f"Authenticated as {user_info['name']}.")
    except Exception as e:
        print(f"Authentication failed: {e}")
    
    # Check if repository exists; if not, create it as a dataset repository
    try:
        print(repo_id)
        print(token)
        repo_info = api.repo_info(repo_id, token=token)
        if repo_info.repo_type != 'dataset':
            print(f"Error: Repository '{repo_id}' exists as a '{repo_info.repo_type}' repository. Please use a dataset repository.")
            exit(1)
        else:
            print(f"Repository '{repo_id}' found as a 'dataset' repository.")
    except RepositoryNotFoundError:
        print(f"Repository '{repo_id}' not found. Creating a new dataset repository.")
        try:
            api.create_repo(repo_id, repo_type='dataset', token=token, private=False)
            print(f"Repository '{repo_id}' created as a 'dataset' repository.")
        except FileExistsError:
            print(f"Repository '{repo_id}' already exists.")
            exit(1)
        except Exception as e:
            print(f"Failed to create repository '{repo_id}': {e}")
            exit(1)
    except Exception as e:
        print(f"An error occurred while accessing repository '{repo_id}': {e}")
        exit(1)
    
    # Get list of files to upload
    files_to_upload = get_all_files(source_dir, recursive)
    if not files_to_upload:
        print(f"No files found in directory '{source_dir}' to upload.")
        exit(0)
    
    print(f"Found {len(files_to_upload)} files to upload from '{source_dir}' to repository '{repo_id}'.")
    
    # Iterate and upload each file
    for file_path in files_to_upload:
        # Determine relative path for repo key
        if recursive:
            relative_path = os.path.relpath(file_path, source_dir)
            s3_key = relative_path.replace("\\", "/")  # Ensure POSIX-style paths
        else:
            s3_key = os.path.basename(file_path)
        print(s3_key)
        
        success = upload_file(api, repo_id, file_path, s3_key, token)
        if not success:
            print(f"Failed to upload '{file_path}'.")
    
    print("All eligible files have been processed.")

if __name__ == "__main__":
    main()
