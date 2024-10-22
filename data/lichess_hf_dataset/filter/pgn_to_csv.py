import os
import csv
import argparse
import re
from tqdm import tqdm

useful_keys = ['WhiteElo', 'BlackElo', 'Result', 'transcript']
# --------------------------- Helper Functions --------------------------- #
def remove_result(transcript, suffix):
    if transcript.endswith(suffix):
        # Remove the suffix using slicing
        return transcript[:-len(suffix) - 1]
    return transcript  # Return the original string if no suffix is found


def remove_braces(text):
    """
    Removes all substrings enclosed in curly braces {} along with the braces
    and the space following the closing brace.

    Parameters:
    text (str): The input string containing text with curly braces.

    Returns:
    str: The cleaned string with specified parts removed.
    """
    # Remove { ... } and the space after the closing brace
    cleaned_text = re.sub(r'\{[^}]*\}\s*', '', text)
    
    # Replace multiple spaces with a single space
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    # Remove any leading or trailing whitespace
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text
                
def parse_pgn_game(game_text):
    """
    Parses a single PGN game text and extracts headers and move text.
    
    Args:
        game_text (str): The raw text of a single PGN game.
    
    Returns:
        dict or None: A dictionary with extracted headers and move text, or None if parsing fails.
    """
    headers = {}
    move_text = ""
    in_header = True
    for line in game_text.strip().split('\n'):
        line = line.strip()
        if in_header:
            if line.startswith('['):
                # Extract header using regex
                match = re.match(r'\[(\w+)\s+"(.+)"\]', line)
                if match:
                    key, value = match.groups()
                    if key in useful_keys:
                        headers[key] = value
            else:
                in_header = False
                move_text += line + ' '
        else:
            move_text += line + ' '
    
    if not headers:
        return None  # Invalid game without headers
    
    return {
        'headers': headers,
        'move_text': move_text.strip()
    }

def is_valid_elo(white_elo, black_elo, min_elo, max_elo):
    """
    Checks if both WhiteElo and BlackElo are within the specified range.
    
    Args:
        white_elo (str): White player's ELO rating.
        black_elo (str): Black player's ELO rating.
        min_elo (int): Minimum ELO threshold.
        max_elo (int): Maximum ELO threshold.
    
    Returns:
        bool: True if both ELOs are within range, False otherwise.
    """
    try:
        white = int(white_elo)
        black = int(black_elo)
        return min_elo <= white <= max_elo and min_elo <= black <= max_elo
    except (ValueError, TypeError):
        return False

# --------------------------- Main Processing --------------------------- #

def parse_pgn_files(pgn_directory, output_csv, min_elo, max_elo, min_len, delimiter=',', verbose=False):
    """
    Parses PGN files in the specified directory and writes filtered games to a CSV file.
    Each game includes WhiteElo, BlackElo, Result, and raw transcript.
    
    Args:
        pgn_directory (str): Path to the directory containing PGN files.
        output_csv (str): Path to the output CSV file.
        min_elo (int): Minimum ELO rating to filter games.
        max_elo (int): Maximum ELO rating to filter games.
        min_len (int): Minimum length of the transcript string.
        delimiter (str, optional): Delimiter for the CSV file. Defaults to ','.
        verbose (bool, optional): Enable verbose output for debugging. Defaults to False.
    """
    # Define CSV fieldnames
    fieldnames = ['WhiteElo', 'BlackElo', 'Result', 'transcript']
    
    # Initialize a counter for written games
    written_games = 0
    skipped_games_for_len = 0
    skipped_games_for_elo = 0
    skipped_games_for_chars = 0

    # Open CSV file once for writing
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='|', quoting=csv.QUOTE_ALL )
        writer.writeheader()
        
        # Get list of PGN files
        pgn_files = [f for f in os.listdir(pgn_directory) if f.lower().endswith('.pgn')]
        
        if not pgn_files:
            print(f"No PGN files found in directory: {pgn_directory}")
            return
        
        # Initialize outer progress bar for files
        for filename in tqdm(pgn_files, desc="Processing PGN Files", unit="file"):
            file_path = os.path.join(pgn_directory, filename)
            # Read the entire file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as pgn_file:
                content = pgn_file.read()
            
            # Split the content into individual games using the improved regex
            # Pattern: Split at blank lines followed by '['
            games = re.split(r'\n\s*\n(?=\[)', content)
            # Initialize an inner progress bar for games
            with tqdm(games, desc=f"Processing {filename}", unit="game", leave=False) as game_bar:
                for game_text in game_bar:
                    parsed_game = parse_pgn_game(game_text)
                    if not parsed_game:
                        if verbose:
                            print("Skipping invalid game (no headers).")
                        continue  # Skip invalid games
                    
                    headers = parsed_game['headers']
                    move_text_with_result = parsed_game['move_text']

                    result = headers.get('Result')
                    white_elo = headers.get('WhiteElo')
                    black_elo = headers.get('BlackElo')
                    

                     # Validate ELO ratings
                    if not is_valid_elo(white_elo, black_elo, min_elo, max_elo):
                        skipped_games_for_elo += 1
                        if verbose:
                            print(f"Skipping game due to ELO: WhiteElo={white_elo}, BlackElo={black_elo}")
                        continue  # Skip games outside the ELO range or with missing ELO
                    
                    # Extract required headers
                    move_text_with_clock = remove_result(move_text_with_result, result)
                    move_text = remove_braces(move_text_with_clock)
                    if '?' in move_text:
                        skipped_games_for_chars += 1
                        continue
                   
                    # Validate transcript length
                    if len(move_text) < min_len:
                        skipped_games_for_len += 1
                        if verbose:
                            print(f"Skipping game due to transcript length: {len(move_text)}")
                        continue  # Skip games with transcripts shorter than min_len


                    # Write the game data directly to CSV
                    writer.writerow({
                        'WhiteElo': white_elo,
                        'BlackElo': black_elo,
                        'Result': result,
                        'transcript': move_text  # Append the raw move transcript as is
                    })
                    
                    # Increment the counter
                    written_games += 1
    
    print(f'Skipped {skipped_games_for_len} games due to length')
    print(f'Skipped {skipped_games_for_elo} games due to elo')
    print(f'Skipped {skipped_games_for_chars} games due to notes')
    total_games = written_games + skipped_games_for_len + skipped_games_for_chars + skipped_games_for_elo
    print(f"Successfully wrote {written_games} games to '{output_csv}', that is {int(written_games*100/(total_games))}% of initial games")

# --------------------------- Execute Script --------------------------- #

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Parse PGN files and filter by ELO range and transcript length")
    parser.add_argument('--pgn_directory', type=str, required=True, help='Directory containing PGN files')
    parser.add_argument('--output_csv', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--min_elo', type=int, required=True, help='Minimum ELO rating for filtering')
    parser.add_argument('--max_elo', type=int, required=True, help='Maximum ELO rating for filtering')
    parser.add_argument('--min_len', type=int, required=True, help='Minimum length of the transcript string (in characters)')
    parser.add_argument('--delimiter', type=str, default='|', help='Delimiter for the CSV file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output for debugging')
    
    args = parser.parse_args()
    
    # Validate ELO range
    if args.min_elo > args.max_elo:
        print("Error: min_elo cannot be greater than max_elo.")
        exit(1)
    
    # Check if PGN directory exists
    if not os.path.isdir(args.pgn_directory):
        print(f"Error: The directory '{args.pgn_directory}' does not exist.")
        exit(1)
    
    # Call the main function with the parsed arguments
    parse_pgn_files(
        pgn_directory=args.pgn_directory,
        output_csv=args.output_csv,
        min_elo=args.min_elo,
        max_elo=args.max_elo,
        min_len=args.min_len,
        delimiter=args.delimiter,
        verbose=args.verbose
    )
