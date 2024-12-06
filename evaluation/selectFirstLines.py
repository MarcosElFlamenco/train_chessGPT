def rewrite_file_with_n_lines(input_file, output_file, n):
    with open(input_file, 'r') as infile:
        lines = infile.readlines()[:n]  # Read the first n lines
    
    with open(output_file, 'w') as outfile:
        outfile.writelines(lines)  # Write only the first n lines


import chess.pgn

def filter_pgn(input_file, output_file, n, m):
    """
    Parses a PGN file and writes the first n games with more than m moves to a new file.
    
    :param input_file: Path to the input PGN file.
    :param output_file: Path to the output PGN file.
    :param n: The number of games to include in the output file.
    :param m: The minimum number of moves a game must have to be included.
    """
    # Open the input and output files
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        games_written = 0
        while games_written < n:
            # Parse the next game
            game = chess.pgn.read_game(infile)
            if game is None:  # End of file
                break
            
            # Count the moves in the game
            moves = sum(1 for _ in game.mainline_moves())
            
            # Check if the game meets the criteria
            if moves > m:
                outfile.write(str(game) + '\n\n')
                games_written += 1

        print(f"Done! Wrote {games_written} games to {output_file}.")

# Usage example
input_pgn = "evaluation/eval_datasets/lichess_db_standard_rated.pgn"
output_pgn = "evaluation/eval_datasets/lichess100games.pgn" # Replace with your desired output PGN file path
n_games = 100            # Number of games to write
min_moves = 85            # Minimum number of moves

filter_pgn(input_pgn, output_pgn, n_games, min_moves)
