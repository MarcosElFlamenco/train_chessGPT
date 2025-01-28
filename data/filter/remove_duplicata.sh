#!/bin/bash

# Check if the input file is provided
if [ $# -lt 1 ]; then
  echo "Usage: $0 input_file [output_file]"
  exit 1
fi

input_file="$1"
output_file="${2:-filtered_output.txt}" # Default to 'filtered_output.txt' if no output file is provided

# Process the file
awk '
BEGIN { previous = "" }
{
    # Skip lines identical to the previous line
    if ($0 == previous) {
        next
    }
    
    # Skip lines that are substrings of the previous line
    if (index(previous, $0) > 0) {
        next
    }
    
    # Print the current line and update "previous"
    print
    previous = $0
}
' "$input_file" > "$output_file"

echo "Filtered lines have been saved to $output_file"
