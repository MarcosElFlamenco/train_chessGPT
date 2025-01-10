import csv
import random
import argparse
import numpy as np

def truncate_transcripts(input_file, output_file, mean_length=463.08, std_dev=50):
    # Open the input CSV file and output CSV file
    with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Read and write the header
        header = next(reader)
        writer.writerow(header)

        # Process each row one at a time to avoid memory issues
        for row in reader:
            original_transcript = row[0]  # Assuming the transcript is in the first column
            # Generate a random length based on a Gaussian distribution
            trunc_length = int(np.clip(np.random.normal(mean_length, std_dev), 1, len(original_transcript)))
            truncated_transcript = original_transcript[:trunc_length]
            writer.writerow([truncated_transcript])

# Usage
parser = argparse.ArgumentParser(description='Process a CSV file with transcripts.')
parser.add_argument('--output_file', type=str, help='The length to truncate the transcript to')
parser.add_argument('--std_dev', type=int, help='The length to truncate the transcript to')

args = parser.parse_args()

input_csv_file = "random_dataset/random.csv"  # Replace with your input CSV file name
output_csv_file = args.output_file

truncate_transcripts(input_csv_file, output_csv_file,std_dev=args.std_dev)
