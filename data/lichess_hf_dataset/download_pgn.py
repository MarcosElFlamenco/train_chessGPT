from tqdm import tqdm
import tempfile
import requests
import os
import zstandard as zstd
import tarfile
import argparse

def download_decompress_extract_zst(urls, post_clean, extract_to=None, chunk_size=1024):
    """
    Downloads a .zst compressed file from the given URL with a progress bar,
    decompresses it, and extracts its contents if it's a tar archive.

    Args:
        url (str): The URL of the .zst file to download.
        extract_to (str, optional): Directory to extract files to. Defaults to a temporary directory.
        chunk_size (int, optional): Chunk size for downloading (in bytes). Defaults to 1024.
    """
    # Define the extraction directory
    if extract_to is None:
        extract_to = tempfile.mkdtemp(prefix="extracted_")
    else:
        os.makedirs(extract_to, exist_ok=True)

   # Create a temporary file to save the downloaded .zst file
    for url in urls:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zst") as tmp_zst:
            tmp_zst_path = tmp_zst.name
            try:
                print(f"Starting download from: {url}")
                # Initiate the GET request with streaming
                with requests.get(url, stream=True) as response:
                    response.raise_for_status()  # Raise an error for bad status codes

                    # Get the total file size from headers
                    total_size_in_bytes = int(response.headers.get('content-length', 0))
                    if total_size_in_bytes == 0:
                        print("Warning: 'Content-Length' header is missing. Progress bar may be inaccurate.")

                    # Initialize tqdm progress bar
                    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc="Downloading")

                    # Download the file in chunks with progress update
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:  # Filter out keep-alive chunks
                            tmp_zst.write(chunk)
                            progress_bar.update(len(chunk))

                    progress_bar.close()

                    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                        print("Warning: Downloaded size does not match the expected Content-Length.")
                
                print(f"Downloaded .zst file to: {tmp_zst_path}")

            except requests.exceptions.RequestException as e:
                print(f"Error downloading the file: {e}")
                return
            except Exception as e:
                print(f"An unexpected error occurred during download: {e}")
                return

        try:
            print(f"Starting decompression of: {tmp_zst_path}")
            # Initialize Zstandard decompressor
            dctx = zstd.ZstdDecompressor()

            # Create a temporary file for the decompressed tar archive
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tar") as tmp_tar:
                tmp_tar_path = tmp_tar.name
                with open(tmp_zst_path, 'rb') as compressed_file, tmp_tar:
                    # Decompress the .zst file in chunks with a progress bar
                    # Estimate the decompressed size if possible (Zstandard doesn't provide it)
                    # Hence, we show a progress bar based on the compressed size
                    compressed_size = os.path.getsize(tmp_zst_path)
                    decompression_progress = tqdm(total=compressed_size, unit='iB', unit_scale=True, desc="Decompressing")

                    # Define a callback to update the progress bar
                    def read_chunks(reader, writer):
                        while True:
                            chunk = reader.read(chunk_size)
                            if not chunk:
                                break
                            writer.write(chunk)
                            decompression_progress.update(len(chunk))

                    # Perform the decompression
                    with dctx.stream_reader(compressed_file) as reader:
                        with tmp_tar:
                            read_chunks(reader, tmp_tar)
                    
                    decompression_progress.close()

                print(f"Decompressed tar archive to: {tmp_tar_path}")

            # Check if the decompressed file is a tar archive
            if tarfile.is_tarfile(tmp_tar_path):
                print(f"Starting extraction of tar archive to: {extract_to}")
                with tarfile.open(tmp_tar_path, 'r') as tar:
                    members = tar.getmembers()
                    with tqdm(total=len(members), desc="Extracting files", unit="files") as extract_bar:
                        for member in members:
                            tar.extract(member, path=extract_to)
                            extract_bar.update(1)
                print(f"Extraction completed. Files are extracted to: {extract_to}")
            else:
                # Move or rename the decompressed file to the desired location
                file_name = url.split('/')[-1].replace('_rated_', '_').replace('.zst', '')
                final_output_path = os.path.join(extract_to,file_name)
                print(file_name)
                os.rename(tmp_tar_path, final_output_path)
                print(f"Decompressed file saved to: {final_output_path}")

        except zstd.ZstdError as e:
            print(f"Error decompressing the .zst file: {e}")
        except tarfile.TarError as e:
            print(f"Error extracting the tar archive: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during decompression or extraction: {e}")
        finally:
            # Clean up temporary files
            if post_clean == 1:
                try:
                    os.remove(tmp_zst_path)
                    print(f"Deleted temporary .zst file: {tmp_zst_path}")
                except OSError as e:
                    print(f"Error deleting temporary .zst file: {e}")

                # Optionally, delete the temporary tar file
                try:
                    if 'tmp_tar_path' in locals() and os.path.exists(tmp_tar_path):
                        os.remove(tmp_tar_path)
                        print(f"Deleted temporary tar file: {tmp_tar_path}")
                except OSError as e:
                    print(f"Error deleting temporary tar file: {e}")

# Usage Example
if __name__ == "__main__":
    # Replace with your actual .zst file URL
#    zst_url = 'https://database.lichess.org/standard/lichess_db_standard_rated_2013-02.pgn.zst'
    # Optionally, specify the extraction directory

    parser = argparse.ArgumentParser(description='Process a CSV file with transcripts.')
    parser.add_argument('--pgn_directory', type=str, help='The path to the input CSV file')
    parser.add_argument('--zst_url', type=str, nargs='+', required=True, help='The path to the input CSV file')
    parser.add_argument('--post_clean', type=int, help='The path to the input CSV file')

    args = parser.parse_args()

    extraction_directory = args.pgn_directory
    post_clean = args.post_clean
    zst_urls = args.zst_url

    download_decompress_extract_zst(zst_urls, post_clean, extract_to=extraction_directory)
