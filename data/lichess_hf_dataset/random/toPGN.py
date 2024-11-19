import pandas as pd

def csv_to_pgn(csv_file, pgn_file, move_column='transcript'):
    """
    Converts a CSV file containing chess game transcripts into a PGN file.

    Args:
        csv_file (str): Path to the input CSV file.
        pgn_file (str): Path to the output PGN file.
        move_column (str): Name of the column containing move lists.
    """
    # Read the CSV
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found.")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{csv_file}' is empty.")
        return
    except pd.errors.ParserError:
        print(f"Error: The file '{csv_file}' does not appear to be in CSV format.")
        return

    if move_column not in df.columns:
        print(f"Error: Column '{move_column}' not found in the CSV file.")
        print(f"Available columns: {list(df.columns)}")
        return

    # Open the PGN file for writing
    with open(pgn_file, 'w') as pgn:
        for idx, row in df.iterrows():
            move_list = row[move_column]

            # Skip if move_list is missing or not a string
            if pd.isnull(move_list) or not isinstance(move_list, str):
                print(f"Warning: Missing or invalid move list at row {idx+1}. Skipping.")
                continue

            # Create PGN headers with placeholder values
            pgn.write('[Event "Converted from CSV"]\n')
            pgn.write('[Result "*"]\n\n')  # '*' indicates an unfinished game
            print(f"this is the move lisr {move_list[:-6]}")
            # Write the move list
            # Ensure that the move list ends with a space for proper parsing
            formatted_moves = move_list[:-6].strip() + ' *\n\n'
            pgn.write(formatted_moves)

    print(f"Successfully converted '{csv_file}' to '{pgn_file}'.")

# Example usage:
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert CSV of chess transcripts to PGN format.")
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the input CSV file.')
    parser.add_argument('--pgn_file', type=str, required=True, help='Path to the output PGN file.')
    parser.add_argument('--move_column', type=str, default='transcript', help='Name of the column containing move lists.')

    args = parser.parse_args()

    csv_to_pgn(args.csv_file, args.pgn_file, args.move_column)
