#!/usr/bin/env python3

import os
import argparse
import logging

def parse_arguments():
    """
    Parses command-line arguments using argparse.
    """
    parser = argparse.ArgumentParser(
        description="Parse a large binary file to identify games separated by integer 15, "
                    "and calculate the total number of games along with their min, max, and average lengths. "
                    "Also, count how many times the integer 15 occurs at positions multiples of 1024."
    )
    
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input binary file."
    )
    
    parser.add_argument(
        "--debug",
        action='store_true',
        help="Enable debug mode for verbose output."
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

def process_binary_file(input_file):
    """
    Processes the binary file to identify games and calculate statistics.
    
    Parameters:
        input_file (str): Path to the binary file.
    
    Returns:
        tuple: (total_games, min_length, max_length, average_length, multiple_1024_15s)
    """
    total_games = 0
    min_length = None
    max_length = None
    total_length = 0
    multiple_1024_15s = 0
    
    current_game_length = 0
    file_position = 0  # Byte position in the file
    
    try:
        with open(input_file, 'rb') as f:
            while True:
                chunk = f.read(8192)  # Read in 8KB chunks
                if not chunk:
                    break  # End of file
                
                logging.debug(f"Read chunk of size: {len(chunk)} bytes")
                
                for byte in chunk:
                    # Check if current byte is 15
                    if byte == 15:
                        # Check if the position is a multiple of 1024
                        if file_position % 1024 == 0:
                            multiple_1024_15s += 1
                            logging.debug(f"Integer 15 found at position {file_position} (multiple of 1024)")
                        
                        if current_game_length > 0:
                            # End of the previous game
                            total_games += 1
                            logging.debug(f"Ending game #{total_games} with length {current_game_length} bytes")
                            
                            # Update min, max, total length
                            if (min_length is None) or (current_game_length < min_length):
                                min_length = current_game_length
                                logging.debug(f"New minimum game length: {min_length} bytes")
                            
                            if (max_length is None) or (current_game_length > max_length):
                                max_length = current_game_length
                                logging.debug(f"New maximum game length: {max_length} bytes")
                            
                            total_length += current_game_length
                            
                        # Start a new game with the current 15 byte
                        current_game_length = 1
                        logging.debug(f"Starting game #{total_games + 1}")
                    else:
                        if current_game_length > 0:
                            # Continue the current game
                            current_game_length += 1
                            
                    file_position += 1  # Increment the file position
                
        # After reading all chunks, check if there's an ongoing game
        if current_game_length > 0:
            total_games += 1
            logging.debug(f"Ending last game #{total_games} with length {current_game_length} bytes")
            
            # Update min, max, total length
            if (min_length is None) or (current_game_length < min_length):
                min_length = current_game_length
                logging.debug(f"New minimum game length: {min_length} bytes")
            
            if (max_length is None) or (current_game_length > max_length):
                max_length = current_game_length
                logging.debug(f"New maximum game length: {max_length} bytes")
            
            total_length += current_game_length
        
    except FileNotFoundError:
        logging.error(f"File not found: {input_file}")
        return (0, 0, 0, 0, 0)
    except Exception as e:
        logging.error(f"An error occurred while processing the file: {e}")
        return (total_games, min_length if min_length else 0, 
                max_length if max_length else 0, 
                (total_length / total_games) if total_games > 0 else 0,
                multiple_1024_15s)
    
    # Calculate average length
    average_length = (total_length / total_games) if total_games > 0 else 0
    
    return (total_games, min_length if min_length else 0, 
            max_length if max_length else 0, 
            average_length, multiple_1024_15s)

def main():
    args = parse_arguments()
    setup_logging(args.debug)
    
    logging.info(f"Starting processing of file: {args.input_file}")
    
    total_games, min_length, max_length, average_length, multiple_1024_15s = process_binary_file(args.input_file)
    
    logging.info("Processing complete.")
    print("\n=== Game Statistics ===")
    print(f"Total number of games: {total_games}")
    if total_games > 0:
        print(f"Minimum game length: {min_length} integers")
        print(f"Maximum game length: {max_length} integers")
        print(f"Average game length: {average_length:.2f} integers")
    else:
        print("No games found.")
    print(f"Number of times integer 15 occurred at positions multiple of 1024: {multiple_1024_15s}")

if __name__ == "__main__":
    main()
