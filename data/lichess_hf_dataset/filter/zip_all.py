import zipfile
import os
import argparse
from tqdm import tqdm

def zip_file_with_progress(zip_dir, input_file, output_zip, chunk_size=1024*1024):
    """
    Zips a file with a progress bar using tqdm.

    Args:
        input_file (str): Path to the input file to zip.
        output_zip (str): Path to the output zip file.
        chunk_size (int, optional): Size of each chunk to read and write (in bytes). Defaults to 1MB.
    """
    try:
        file_size = os.path.getsize(input_file)
    except OSError as e:
        print(f"Error accessing file '{input_file}': {e}")
        return
    print('bindir', zip_dir)
    print("output dir ", output_zip)
    os.makedirs(os.path.dirname(output_zip), exist_ok=True)
        
        
 
# Initialize the zip file
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Open the zip entry for writing
        with zipf.open(os.path.basename(input_file), 'w') as dest_file:
            # Open the source file for reading in binary mode
            with open(input_file, 'rb') as src_file:
                # Initialize tqdm progress bar
                with tqdm(total=file_size, unit='B', unit_scale=True, desc=f'Zipping {os.path.basename(input_file)}') as pbar:
                    while True:
                        # Read a chunk of data from the source file
                        data = src_file.read(chunk_size)
                        if not data:
                            break  # EOF
                        # Write the chunk to the zip entry
                        dest_file.write(data)
                        # Update the progress bar
                        pbar.update(len(data))

def main():
    """
    Main function to parse arguments and zip the specified CSV files with progress bars.
    """
    # Initialize argument parser
    parser = argparse.ArgumentParser(description='Process a CSV file and push the dataset to Hugging Face Hub.')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--zip_directory', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--bin_directory', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--files', type=str, required=True, help='Path to the input CSV file')

    # Parse arguments
    args = parser.parse_args()

    # Extract the CSV file path
    csv_file = args.csv_file
    zip_directory = args.zip_directory
    bin_directory = args.bin_directory
    files = args.files

    # Validate the CSV file exists
    if not os.path.isfile(csv_file):
        print(f"Error: CSV file '{csv_file}' does not exist.")
        exit(1)

    # Generate the blocks filename by replacing '.csv' with '_blocks.csv'
    base, ext = os.path.splitext(csv_file)
    blocks_filename = f"{base}_blocks{ext}"
    train_bin_filename = os.path.join(bin_directory, 'train.bin')
    val_bin_filename = os.path.join(bin_directory, 'val.bin')

    # Define the output zip filenames
    csv_zip = os.path.join(zip_directory, f"{base}.zip")
    blocks_zip = os.path.join(zip_directory, f"{base}_blocks.zip")
    train_bin_zip =  os.path.join(zip_directory, f'{base}_train_bin.zip')
    val_bin_zip =  os.path.join(zip_directory, f'{base}_val_bin.zip')

    if files == 'blocks':
        # Zip the main CSV file with a progress bar
        if os.path.isfile(csv_file):
            print(f"Starting to zip '{csv_file}' into '{csv_zip}'...")
            zip_file_with_progress(zip_directory, csv_file, csv_zip)
            print(f"Finished zipping '{csv_file}' into '{csv_zip}'.\n")
        else:
            print(f"Warning: Blocks file '{csv_file}' does not exist. Skipping zipping of this file.")

        # Check if the blocks CSV file exists before attempting to zip
        if os.path.isfile(blocks_filename):
            print(f"Starting to zip '{blocks_filename}' into '{blocks_zip}'...")
            zip_file_with_progress(zip_directory, blocks_filename, blocks_zip)
            print(f"Finished zipping '{blocks_filename}' into '{blocks_zip}'.")
        else:
            print(f"Warning: Blocks file '{blocks_filename}' does not exist. Skipping zipping of this file.")
    if files == 'bins':
        if os.path.isfile(train_bin_filename):
            print(f"Starting to zip '{train_bin_filename}' into '{train_bin_zip}'...")
            zip_file_with_progress(zip_directory, train_bin_filename, train_bin_zip)
            print(f"Finished zipping '{train_bin_filename}' into '{train_bin_zip}'.")
        else:
            print(f"Warning: train_bin file '{train_bin_filename}' does not exist. Skipping zipping of this file.")

        if os.path.isfile(val_bin_filename):
            print(f"Starting to zip '{val_bin_filename}' into '{val_bin_zip}'...")
            zip_file_with_progress(zip_directory, val_bin_filename, val_bin_zip)
            print(f"Finished zipping '{val_bin_filename}' into '{val_bin_zip}'.")
        else:
            print(f"Warning: val_bin file '{val_bin_filename}' does not exist. Skipping zipping of this file.")


if __name__ == "__main__":
    main()
