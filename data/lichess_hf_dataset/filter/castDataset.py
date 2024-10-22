import argparse
from datasets import load_dataset, Value
import os

# Argument parsing to allow CSV file name and dataset name as inputs
parser = argparse.ArgumentParser(description='Process a CSV file and push the dataset to Hugging Face Hub.')
parser.add_argument('--csv_file', type=str, help='Path to the input CSV file')
parser.add_argument('--dataset_name', type=str, help='Name of the dataset to push to Hugging Face Hub')

args = parser.parse_args()

csv_file = args.csv_file
dataset_name = args.dataset_name

int_columns = ['WhiteElo', 'BlackElo']

# Load the dataset
dataset = load_dataset(
    'csv',
    data_files=csv_file,
    delimiter='|',            # Specify the pipe delimiter
    quotechar='"',            # Specify the quote character
    quoting=1,                # Corresponds to csv.QUOTE_ALL
    header='infer',           # Indicates that the first row contains headers
    encoding='utf-8'          # Ensure the encoding matches your CSV file
)

# Function to safely cast columns to integers
def safe_cast_to_int(example, column, default=0):
    try:
        return {column: int(example[column])}
    except (ValueError, TypeError):
        return {column: default}

# Apply safe casting to integer columns
for column in int_columns:
    dataset = dataset.map(lambda x: safe_cast_to_int(x, column), batched=False)
    dataset = dataset.cast_column(column, Value('int64'))
    print(f"'{column}' casted to int64.")

# Verify the final data types
print("\nFinal Features:")
for column, dtype in dataset['train'].features.items():
    print(f"{column}: {dtype}")

# Display a few samples
print("Sample Data:")
for i in range(2):
    print(dataset['train'][i])

# Push the dataset to Hugging Face Hub
#dataset.push_to_hub(os.path.join(dataset_name))
 