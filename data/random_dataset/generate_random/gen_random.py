import chess
import chess.pgn
import random
import csv
import sys
import boto3
from botocore.exceptions import NoCredentialsError
import argparse

def upload_to_s3(file_name, bucket_name, object_name=None):
    """
    Upload a file to an S3 bucket

    Parameters:
    - file_name: The file to upload
    - bucket_name: The S3 bucket to upload to
    - object_name: S3 object name. If not specified, file_name is used.
    """
    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # Create an S3 client
    s3_client = boto3.client('s3')

    try:
        # Upload the file
        s3_client.upload_file(file_name, bucket_name, object_name)
        print(f"File '{file_name}' successfully uploaded to '{bucket_name}/{object_name}'.")
    except FileNotFoundError:
        print(f"File '{file_name}' not found.")
    except NoCredentialsError:
        print("Credentials not available.")


def generate_random_game(max_moves=174):
    """
    Generates a chess game with random moves.

    Parameters:
    - max_moves (int): Maximum number of half-moves (plies) to prevent excessively long games.

    Returns:
    - game (chess.pgn.Game): The generated chess game in PGN format.

    Note: 174 moves is the max ever necessary to fill up a 1024 string
    on my local computer this scipt generates at a speed of approx 5000 games/minute
    """
    board = chess.Board()
    game = chess.pgn.Game()
    node = game

    move_count = 0

    while not board.is_game_over() and move_count < max_moves:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break  # No legal moves available

        move = random.choice(legal_moves)
        board.push(move)
        node = node.add_variation(move)
        move_count += 1

    # Optionally, you can set the Result header to a dummy value or leave it as is
    # Since we are using dummy values, the Result in the game object is not used

    return game

def game_to_pgn_string(game):
    """
    Converts a chess.pgn.Game object to a single-line PGN string without line breaks.

    Parameters:
    - game (chess.pgn.Game): The chess game to convert.

    Returns:
    - pgn_str (str): The PGN string.
    """
    exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
    pgn_str = game.accept(exporter)
    # Remove line breaks and carriage returns
    pgn_str = pgn_str.replace('\n', ' ').replace('\r', ' ').strip()
    if pgn_str.endswith(' *'):
        pgn_str = pgn_str[:-2]
    return pgn_str

def process_pgn_to_transcript(pgn_str, total_length=1024):
    """
    Processes the PGN string to fit into the transcript format:
    - Starts with ';' (optional based on original code)
    - Followed by 1023 characters from PGN string
    - If PGN string is shorter, pad with spaces

    Parameters:
    - pgn_str (str): The PGN string to process.
    - total_length (int): Total length of the transcript including the ';' character.

    Returns:
    - transcript (str): The processed transcript string.
    """
    max_content_length = total_length - 1  # Account for the ';' character
    # Truncate or pad the PGN string
    if len(pgn_str) > max_content_length:
        content = pgn_str[:max_content_length]
    else:
        content = pgn_str
    return content

def generate_and_save_random_games_csv(number_of_games=1, output_file="random_chess_games.csv", log_freq=100, save_freq=1000, save_s3=False, bucket_name='go-bucket-craft', object_name='random.csv'):
    """
    Generates a specified number of random chess games and saves them to a CSV file.

    Parameters:
    - number_of_games (int): Number of random games to generate.
    - output_file (str): The filename for the output CSV file.
    """
    with open(output_file, mode='w', newline='', encoding='utf-8') as csv_file:
        # Initialize the CSV writer with '|' as delimiter and all fields quoted
        writer = csv.writer(csv_file, delimiter='|', quoting=csv.QUOTE_ALL)
        # Write the header
#        writer.writerow(['WhiteElo', 'BlackElo', 'Result', 'transcript'])
        writer.writerow(['transcript'])

        for i in range(1, number_of_games + 1):
            game = generate_random_game()
            pgn_str = game_to_pgn_string(game)
            transcript = process_pgn_to_transcript(pgn_str)
            # Write the row with dummy values and the transcript
#            writer.writerow([1000, 1000, '1-0', transcript])
            writer.writerow([transcript])
            if (i % log_freq == 0):
                print(f"Generated game {i}")
            if (save_s3==1 and i % save_freq == 0):
                upload_to_s3(output_file , bucket_name, object_name)

    print(f"\nAll {number_of_games} games have been saved to '{output_file}'.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process a CSV file with transcripts.')
    parser.add_argument('--num_games', type=int, help='The path to the input CSV file')
    parser.add_argument('--output_file', type=str, help='The length to truncate the transcript to')
    parser.add_argument('--log_freq', type=int, default = 1000, help='The length to truncate the transcript to')
    parser.add_argument('--save_freq', type=int, default = 1000, help='The length to truncate the transcript to')
    parser.add_argument('--save_s3', type=int, default= 0, help="Enable feature")
    parser.add_argument("--bucket_name", type=str, help="The name of the S3 bucket to upload to.")
    parser.add_argument("--object_name", type=str, help="The S3 object name. If not provided, the file name will be used.", default=None)
 
    args = parser.parse_args()
   # Default parameters
    num_games = args.num_games
    output = args.output_file
    log_freq = args.log_freq
    save_freq = args.save_freq
    bucket_name = args.bucket_name
    object_name = args.object_name



    generate_and_save_random_games_csv(number_of_games=num_games, output_file=output, log_freq=log_freq, save_freq=save_freq, save_s3=args.save_s3, bucket_name=bucket_name, object_name=object_name)

