def count_newlines_in_file(file_path):
    """
    Counts the number of newline characters in a file.

    :param file_path: Path to the file
    :return: Number of newlines (i.e., number of lines in the file)
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        count = 0
        for line in file:
            count += 1
    return count

# Example usage
file_path = 'poc_blocks.csv'  # Replace with your file path
newline_count = count_newlines_in_file(file_path)
print(f'The file {file_path} contains {newline_count} lines.')
