#!/bin/bash

# Check if the input file is provided
if [ $# -lt 1 ]; then
  echo "Usage: $0 input_file [output_file]"
  exit 1
fi

input_file="$1"
output_file="${2:-filtered_output.txt}" # Default to 'filtered_output.txt' if no output file is provided

# Use awk to filter out lines shorter than 10 characters or starting with "1.-- --"
awk 'length($0) >= 10 && !($0 ~ /^1\.\-\- \-\-/)' "$input_file" > "$output_file"


echo "Filtered lines have been saved to $output_file"
