#!/usr/bin/env python3

import os
import argparse
import csv
import logging
import re
import chess.pgn
import shutil

def parse_arguments():
    """
    Parses command-line arguments using argparse.
    """
    parser = argparse.ArgumentParser(
        description="Parse PGN files, filter games based on ELO and game length, and export game transcripts to CSV. "
                    "Tracks processed files to allow interruption and resumption without duplicating data."
    )
    
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to the directory containing PGN files."
    )
    
    parser.add_argument(
        "output_csv",
        type=str,
        help="Name (and path) of the output CSV file."
    )
    
    parser.add_argument(
        "--min_elo",
        type=int,
        default=0,
        help="Minimum ELO rating for both players. Default is 0."
    )
    
    parser.add_argument(
        "--max_elo",
        type=int,
        default=3000,
        help="Maximum ELO rating for both players. Default is 3000."
    )
    
    parser.add_argument(
        "--min_length",
        type=int,
        default=10,
        help="Minimum number of plies (half-moves) required for a game to be included. Default is 10."
    )
    
    parser.add_argument(
        "--debug",
        action='store_true',
        help="Enable debug mode for verbose output."
    )
    
    parser.add_argument(
        "--processed_files",
        type=str,
        default="processed_files.txt",
        help="Path to the file tracking processed PGN files. Default is 'processed_files.txt'."
    )
    
    return parser.parse_args()

def setup_logging(debug_mode):
    """
    Sets up logging configuration.
    """
    if debug_mode:
        logging_level = logging.DEBUG
    else:
        logging_level = logging.INFO
    
    logging.basicConfig(
        level=logging_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def process_move_string(moves):
    """
    Processes the move string by removing spaces after move numbers.
    Example: '6. c4' becomes '6.c4'
    Also removes the game result from the move string.
    Ensures the move string is a single line.
    """
    # Remove game result from the moves
    moves = re.sub(r'\s*(1-0|0-1|1/2-1/2|\*)$', '', moves)
    # Use regex to replace move numbers followed by space with move number and dot
    processed_moves = re.sub(r'(\d+)\.\s+', r'\1.', moves)
    # Replace any newlines with spaces to ensure single line
    processed_moves = processed_moves.replace('\n', ' ').replace('\r', ' ')
    return processed_moves

def get_game_length(moves):
    """
    Returns the number of plies (half-moves) in the game.
    """
    # Remove comments, variations, and result markers
    moves_clean = re.sub(r'\{[^}]*\}', '', moves)  # Remove comments
    moves_clean = re.sub(r'\([^)]*\)', '', moves_clean)  # Remove variations
    # Remove result markers already handled in process_move_string
    # Split the moves by whitespace
    tokens = moves_clean.strip().split()
    # Count plies by counting move tokens that are not move numbers
    plies = len([token for token in tokens if not re.match(r'^\d+\.$', token)])
    return plies

def extract_moves(game):
    """
    Extracts and processes the move string from a chess.pgn.Game object.
    """
    # Get the full move text
    exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
    moves = game.accept(exporter)
    # Process the move string
    processed_moves = process_move_string(moves)
    return processed_moves

def write_csv_header_if_needed(output_csv, csv_headers):
    """
    Writes the CSV header if the file does not already exist or is empty.
    """
    file_exists = os.path.isfile(output_csv)
    if not file_exists or os.path.getsize(output_csv) == 0:
        with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
            writer.writeheader()
            logging.debug(f"CSV header written: {csv_headers}")

def load_processed_files(processed_files_path):
    """
    Loads the list of already processed PGN files.
    """
    if not os.path.isfile(processed_files_path):
        return set()
    with open(processed_files_path, 'r', encoding='utf-8') as f:
        processed = set(line.strip() for line in f if line.strip())
    logging.debug(f"Loaded {len(processed)} processed files from {processed_files_path}")
    return processed

def mark_file_as_processed(processed_files_path, file_path):
    """
    Marks a PGN file as processed by appending its path to the processed_files file.
    """
    with open(processed_files_path, 'a', encoding='utf-8') as f:
        f.write(file_path + '\n')
    logging.debug(f"Marked file as processed: {file_path}")

def append_csv(main_csv, temp_csv):
    """
    Appends the contents of temp_csv to main_csv.
    """
    with open(temp_csv, 'r', encoding='utf-8') as src, open(main_csv, 'a', encoding='utf-8') as dst:
        shutil.copyfileobj(src, dst)
    logging.debug(f"Appended temporary CSV {temp_csv} to main CSV {main_csv}")

def filter_and_write_games(args):
    """
    Main function to filter games and write to CSV.
    """
    input_dir = args.input_dir
    output_csv = args.output_csv
    min_elo = args.min_elo
    max_elo = args.max_elo
    min_length = args.min_length
    debug = args.debug
    processed_files_path = args.processed_files
    
    # Counters
    skipped_elo = 0
    skipped_length = 0
    written_games = 0
    total_games = 0
    
    # Prepare CSV writing
    csv_headers = [
        "Moves"
    ]
    
    # Ensure the output CSV has headers
    write_csv_header_if_needed(output_csv, csv_headers)
    
    # Load processed files
    processed_files = load_processed_files(processed_files_path)
    
    # Iterate over all PGN files in the input directory
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith('.pgn'):
                file_path = os.path.abspath(os.path.join(root, filename))
                
                if file_path in processed_files:
                    logging.info(f"Skipping already processed file: {file_path}")
                    continue
                
                logging.info(f"Processing file: {file_path}")
                
                temp_csv = output_csv + ".part"
                # Ensure temp_csv is empty before starting
                with open(temp_csv, 'w', newline='', encoding='utf-8') as temp_file:
                    writer = csv.DictWriter(temp_file, fieldnames=csv_headers)
                    writer.writeheader()
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as pgn_file, \
                         open(temp_csv, 'a', newline='', encoding='utf-8') as temp_file:
                        writer = csv.DictWriter(temp_file, fieldnames=csv_headers)
                        
                        while True:
                            try:
                                game = chess.pgn.read_game(pgn_file)
                                if game is None:
                                    break  # End of file
                                
                                total_games += 1
                                
                                headers = game.headers
                                white_elo = headers.get("WhiteElo", "2500")
                                black_elo = headers.get("BlackElo", "2500")
                                
                                try:
                                    white_elo = int(white_elo)
                                    black_elo = int(black_elo)
                                except ValueError:
                                    logging.debug(f"Non-integer ELO found in game {total_games} of file {file_path}. Skipping.")
                                    skipped_elo += 1
                                    continue
                                
                                # Check ELO constraints
                                if not (min_elo <= white_elo <= max_elo and min_elo <= black_elo <= max_elo):
                                    logging.debug(f"Game {total_games} of file {file_path} skipped due to ELO constraints: WhiteElo={white_elo}, BlackElo={black_elo}")
                                    skipped_elo += 1
                                    continue
                                
                                # Extract and process moves
                                moves = extract_moves(game)
                                
                                # Measure game length
                                game_length = get_game_length(moves)
                                
                                logging.debug(f"Game {total_games} of file {file_path} length (plies): {game_length}")
                                
                                # Check game length
                                if game_length < min_length:
                                    logging.debug(f"Game {total_games} of file {file_path} skipped due to insufficient length: {game_length} plies")
                                    skipped_length += 1
                                    continue
                                
                                # Write the moves to temporary CSV
                                game_data = {
                                    "Moves": moves
                                }
                                
                                writer.writerow(game_data)
                                written_games += 1
                                
                                if debug:
                                    logging.debug(f"Game {total_games} of file {file_path} written to CSV.")
                                
                            except Exception as e:
                                logging.error(f"Error processing game {total_games} in file {file_path}: {e}")
                                continue  # Skip to the next game
                    
                    # After successfully processing the entire file, append temp_csv to main_csv
                    append_csv(output_csv, temp_csv)
                    # Remove the temporary CSV
                    os.remove(temp_csv)
                    logging.info(f"Finished processing and marked as processed: {file_path}")
                    # Mark the file as processed
                    mark_file_as_processed(processed_files_path, file_path)
                
                except Exception as e:
                    logging.error(f"Error processing file {file_path}: {e}")
                    # Optionally, remove the temporary CSV if an error occurred
                    if os.path.exists(temp_csv):
                        os.remove(temp_csv)
                    continue  # Skip to the next file
    
    # Tally Results
    print("\n=== Processing Complete ===")
    print(f"Total games processed: {total_games}")
    print(f"Total games written to CSV: {written_games}")
    print(f"Total games skipped due to ELO constraints: {skipped_elo}")
    print(f"Total games skipped due to length constraints: {skipped_length}")

def main():
    args = parse_arguments()
    setup_logging(args.debug)
    filter_and_write_games(args)

if __name__ == "__main__":
    main()
