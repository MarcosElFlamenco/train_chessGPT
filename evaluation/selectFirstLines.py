def rewrite_file_with_n_lines(input_file, output_file, n):
    with open(input_file, 'r') as infile:
        lines = infile.readlines()[:n]  # Read the first n lines
    
    with open(output_file, 'w') as outfile:
        outfile.writelines(lines)  # Write only the first n lines

# Example usage
input_file = "evaluation/lichess_db_standard_rated.pgn"  # Replace with your file path
output_file = "toy.pgn"  # Replace with your file path (can be the same as input_file if you want to overwrite)
n = 10  # Number of lines to keep
rewrite_file_with_n_lines(input_file, output_file, 20)
