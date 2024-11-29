import pandas as pd
from tqdm import tqdm
from collections import deque
import os
import csv
import argparse

# Function to transform the text
def transform_text(text, input_length):
    # Check if the input is not a string and return an empty string or handle it as you prefer
    if not isinstance(text, str):
        return ''  # Or return text as is if you'd prefer to keep it: return text
    
    # If the text doesn't contain '\n\n', return the first input_length characters
    if '\n\n' not in text:
        return text[:input_length]
    
    # Otherwise, split by '\n\n' and return the second part, truncated to input_length characters
    new_text = text.split('\n\n')[1].strip()
    return new_text[:input_length]

# Argument parsing to get input file and input length from command line
parser = argparse.ArgumentParser(description='Process a CSV file with transcripts.')
parser.add_argument('--input_file', type=str, help='The path to the input CSV file')
parser.add_argument('--input_length', type=int, help='The length to truncate the transcript to')
parser.add_argument('--give_stats', type=int, help='The length to truncate the transcript to')
parser.add_argument('--csv_type', type=str, help='The length to truncate the transcript to')

args = parser.parse_args()
input_file = args.input_file
input_length = args.input_length
output_filename = input_file.replace('.csv', '_blocks.csv')

if args.csv_type == "quotes":
    df = pd.read_csv(
        input_file, 
        delimiter='|',
        quotechar='"',
        quoting=csv.QUOTE_ALL,
        encoding='utf-8',
        usecols=["transcript"]
    )
else:
    df = pd.read_csv(
    input_file, 
    delimiter=',',
    quoting=csv.QUOTE_NONE,  # No quotes in the CSV
    encoding='utf-8',
    usecols=["transcript"]
    )

# Apply the transform_text function
df['transcript'] = df['transcript'].apply(lambda text: transform_text(text, input_length))

df['length'] = df['transcript'].apply(len)
df.sort_values(by='length', inplace=True)

# Prepare the new dataset for blocks
blocks = []
remaining_games = deque(df['transcript'].tolist())  # Use deque for efficient pops from the left
del df

original_length = len(remaining_games)  # Store the original length
block_size = 1024

# Initialize the progress bar
with tqdm(total=original_length, desc="Processing") as pbar:
    while remaining_games:
        block = ';'
        # Select the next game
        next_game = remaining_games.pop()
        block += next_game
        while len(block) < block_size and remaining_games:
            next_game = remaining_games.popleft()
            block += ';' + next_game
            if len(block) > block_size:
                if len(remaining_games) > 100:
                    remaining_games.insert(99, next_game)
                else:
                    break
                break

        if len(block) >= block_size:
            # Add the block to the blocks list
            blocks.append(block[:block_size])

        # Update the progress bar
        pbar.update(original_length - len(remaining_games) - pbar.n)

# Create a new DataFrame for the blocks
df = pd.DataFrame(blocks, columns=['transcript'])

# Save the blocks to a new CSV file
# df.to_csv(output_filename, index=False)

# df = pd.read_csv(output_filename)
df = df.sample(frac=1).reset_index(drop=True)

# Save the shuffled DataFrame to the same CSV file
df.to_csv(output_filename, index=False)

#df['length'] = df['transcript'].apply(len)
#print(df['length'].describe())