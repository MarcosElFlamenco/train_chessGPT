import statistics
import json
import matplotlib.pyplot as plt

import chess.pgn
import chess
import chess.engine

import os
import time

import chess
import chess.pgn
import chess.engine
from io import StringIO


import chess
import chess.pgn
import chess.engine
from io import StringIO
import time

def evaluate_and_play_moves(pgn_string, stockfish_path, time_per_move=0.1, max_moves=10):
    """
    Evaluates a position from a PGN string sequence using Stockfish, prints the current evaluation,
    and plays the next up to 10 moves (or until checkmate/draw) with evaluation on every move.

    Parameters:
    - pgn_string (str): PGN string representing the sequence of moves.
    - stockfish_path (str): Path to the Stockfish binary.
    - time_per_move (float): Time limit per move for Stockfish evaluation (in seconds).
    - max_moves (int): Maximum number of moves to play.

    Returns:
    - None
    """
    # Initialize the Stockfish engine
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    # Create a chess board
    board = chess.Board()

    # Parse the PGN string and apply the moves to the board
    pgn = StringIO(pgn_string)
    game = chess.pgn.read_game(pgn)
    if game is None:
        print("Error: Invalid PGN string.")
        engine.quit()
        return

    node = game
    while node.variations:
        move = node.variation(0).move
        board.push(move)
        node = node.variation(0)

    # Print the initial position evaluation
    start_time = time.perf_counter()
    result = engine.analyse(board, chess.engine.Limit(time=time_per_move))
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    current_score = result["score"].white().score(mate_score=100000)  # From White's perspective
    current_evaluation = current_score / 100  # Convert to pawns

    print(f"Initial Evaluation: {current_evaluation:.2f} pawns (Time: {elapsed_time:.6f} seconds)\n")

    # Play up to max_moves or until the game is over
    move_count = 0
    while not board.is_game_over() and move_count < max_moves:
        # Play the best move with Stockfish
        start_time = time.perf_counter()
        result = engine.play(board, chess.engine.Limit(time=time_per_move))
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        move = result.move
        board.push(move)
        move_count += 1

        # Evaluate the new position
        start_time = time.perf_counter()
        result = engine.analyse(board, chess.engine.Limit(time=time_per_move))
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        new_score = result["score"].white().score(mate_score=100000)  # From White's perspective
        new_evaluation = new_score / 100  # Convert to pawns

        print(f"Move {move_count}: {move.uci()}")
        print(f"New Evaluation: {new_evaluation:.2f} pawns (Time: {elapsed_time:.6f} seconds)\n")

    # Print the final result of the game
    result = board.result()
    print(f"Final Result: {result}")

    # Close the engine
    engine.quit()

def evaluate_and_play_move(pgn_string, stockfish_path, time_per_move=0.1):
    """
    Evaluates a position from a PGN string sequence using Stockfish, prints the current evaluation,
    and returns the move Stockfish would play along with the new evaluation after the move.

    Parameters:
    - pgn_string (str): PGN string representing the sequence of moves.
    - stockfish_path (str): Path to the Stockfish binary.
    - time_per_move (float): Time limit per move for Stockfish evaluation (in seconds).

    Returns:
    - move (str): The move Stockfish would play in UCI format.
    - new_evaluation (float): The evaluation after the move in pawns.
    """
    # Initialize the Stockfish engine
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    # Create a chess board
    board = chess.Board()

    # Parse the PGN string and apply the moves to the board
    pgn = StringIO(pgn_string)
    game = chess.pgn.read_game(pgn)
    if game is None:
        print("Error: Invalid PGN string.")
        engine.quit()
        return None, None

    node = game
    while node.variations:
        move = node.variation(0).move
        board.push(move)
        node = node.variation(0)

    # Evaluate the current position
    start_time = time.perf_counter()
    result = engine.analyse(board, chess.engine.Limit(time=time_per_move))
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    current_score = result["score"].white().score(mate_score=100000)  # From White's perspective
    current_evaluation = current_score / 100  # Convert to pawns

    print(f"Current Evaluation: {current_evaluation:.2f} pawns (Time: {elapsed_time:.6f} seconds)")

    # Play the best move with Stockfish
    start_time = time.perf_counter()
    result = engine.play(board, chess.engine.Limit(time=time_per_move))
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    move = result.move
    board.push(move)

    # Evaluate the new position
    start_time = time.perf_counter()
    result = engine.analyse(board, chess.engine.Limit(time=time_per_move))
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    new_score = result["score"].white().score(mate_score=100000)  # From White's perspective
    new_evaluation = new_score / 100  # Convert to pawns

    print(f"Move: {move.uci()}")
    print(f"New Evaluation: {new_evaluation:.2f} pawns (Time: {elapsed_time:.6f} seconds)")

    # Close the engine
    engine.quit()

    return move.uci(), new_evaluation

# Example usage in a Jupyter notebook
# pgn_string = "1.e4 e5 2.Nf3 Nc6 3.Bb5"
# stockfish_path = "/path/to/stockfish"
# move, new_evaluation = evaluate_and_play_move(pgn_string, stockfish_path)

def print_nth_game(pgn_file, n):
    """
    Prints the nth game from a PGN file, including headers and moves.

    Parameters:
    - pgn_file (str): Path to the PGN file.
    - n (int): Index of the game to print (0-based index).
    """
    with open(pgn_file) as pgn:
        game_index = 0
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            if game_index == n:
                print(f"Game {n + 1}:")
                print("\nMoves:")
                print(game)
                print(f"give it a minute we're not quite there yet")
#                print(game.split("]")[-1])
                return
            game_index += 1

    print(f"Error: Game index {n} is out of range. Valid indices are 0 to {game_index - 1}.")

# Example usage in a Jupyter notebook
# print_nth_game('kasparov_games.pgn', n=0)

def analyze_best_scores(json_file,pgn_file):
    """
    Reads the JSON file containing game evaluations, computes the best score for the losing side
    in games where White or Black wins, and plots histograms for these scores.

    Parameters:
    - json_file (str): Path to the JSON file containing the game evaluations.
    """
    # Load the existing data from the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Initialize lists to store the best scores for the losing side
    best_black_scores_in_white_wins = []
    best_white_scores_in_black_wins = []

    # Iterate through each game in the data
    for game_data in data:
        win_color = game_data['win_color']
        evaluations = game_data['evaluations']
        
        if not evaluations:
            continue
        
        # Track the best scores for the losing side
        blunder = False
        if win_color == "white":
            # Find the most negative score for Black
            best_black_score = min(evaluations)
            if best_black_score <= -99000:
                blunder = True
            else:
                best_black_scores_in_white_wins.append(best_black_score)
        elif win_color == "black":
            # Find the most positive score for White
            best_white_score = max(evaluations)
            if best_white_score > 99000:
                blunder = True
            else:
                best_white_scores_in_black_wins.append(best_white_score)
        if blunder:
            move_number = evaluations.index(best_black_score) + 1
            game_number = data.index(game_data)
            print(f"Identified a blunder in game {game_number} at move {move_number}")
            print_nth_game(pgn_file,game_number)

    # Calculate statistics for Best Black Scores in White Wins
    if best_black_scores_in_white_wins:
        best_black_max_score = max(best_black_scores_in_white_wins)
        best_black_min_score = min(best_black_scores_in_white_wins)
        best_black_avg_score = statistics.mean(best_black_scores_in_white_wins)
    else:
        best_black_max_score = None
        best_black_min_score = None
        best_black_avg_score = None

    # Calculate statistics for Best White Scores in Black Wins
    if best_white_scores_in_black_wins:
        best_white_max_score = max(best_white_scores_in_black_wins)
        best_white_min_score = min(best_white_scores_in_black_wins)
        best_white_avg_score = statistics.mean(best_white_scores_in_black_wins)
    else:
        best_white_max_score = None
        best_white_min_score = None
        best_white_avg_score = None

    # Print the statistics
    print("\nBest Black Scores in White Wins:")
    print(f"Number of Games: {len(best_black_scores_in_white_wins)}")
    print(f"Max Score: {best_black_max_score / 100:.2f} pawns" if best_black_max_score is not None else "N/A")
    print(f"Min Score: {best_black_min_score / 100:.2f} pawns" if best_black_min_score is not None else "N/A")
    print(f"Avg Score: {best_black_avg_score / 100:.2f} pawns" if best_black_avg_score is not None else "N/A")

    print("\nBest White Scores in Black Wins:")
    print(f"Number of Games: {len(best_white_scores_in_black_wins)}")
    print(f"Max Score: {best_white_max_score / 100:.2f} pawns" if best_white_max_score is not None else "N/A")
    print(f"Min Score: {best_white_min_score / 100:.2f} pawns" if best_white_min_score is not None else "N/A")
    print(f"Avg Score: {best_white_avg_score / 100:.2f} pawns" if best_white_avg_score is not None else "N/A")

    # Plot histograms for Best Black Scores in White Wins and Best White Scores in Black Wins
    plt.figure(figsize=(14, 6))

    # Best Black Scores in White Wins Histogram
    plt.subplot(1, 2, 1)
    if best_black_scores_in_white_wins:
        plt.hist([score / 100 for score in best_black_scores_in_white_wins], bins=50, color='red', alpha=0.7, edgecolor='black')
        plt.title('Best Black Scores in White Wins')
        plt.xlabel('Best Score for Black (pawns)')
        plt.ylabel('Frequency')
        plt.axvline(x=best_black_avg_score / 100, color='blue', linestyle='dashed', linewidth=2, label=f'Avg: {best_black_avg_score / 100:.2f} pawns')
        plt.legend()
    else:
        plt.title('Best Black Scores in White Wins')
        plt.xlabel('Best Score for Black (pawns)')
        plt.ylabel('Frequency')
        plt.text(0.5, 0.5, 'No Games', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12)

    # Best White Scores in Black Wins Histogram
    plt.subplot(1, 2, 2)
    if best_white_scores_in_black_wins:
        plt.hist([score / 100 for score in best_white_scores_in_black_wins], bins=50, color='green', alpha=0.7, edgecolor='black')
        plt.title('Best White Scores in Black Wins')
        plt.xlabel('Best Score for White (pawns)')
        plt.ylabel('Frequency')
        plt.axvline(x=best_white_avg_score / 100, color='blue', linestyle='dashed', linewidth=2, label=f'Avg: {best_white_avg_score / 100:.2f} pawns')
        plt.legend()
    else:
        plt.title('Best White Scores in Black Wins')
        plt.xlabel('Best Score for White (pawns)')
        plt.ylabel('Frequency')
        plt.text(0.5, 0.5, 'No Games', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12)

    plt.tight_layout()
    plt.show()

# Example usage in a Jupyter notebook
# analyze_best_scores('evaluations.json')

def analyze_and_plot_game_statistics(json_file):
    """
    Reads the JSON file containing game evaluations, computes statistical information
    about wins, losses, and draws, and plots histograms for White and Black wins.
    Outliers (1000 and -1000) are counted but excluded from statistical calculations and plots.

    Parameters:
    - json_file (str): Path to the JSON file containing the game evaluations.
    """
    # Load the existing data from the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Initialize lists to store final scores for White wins and Black wins
    white_win_scores = []
    black_win_scores = []
    draw_scores = []

    # Initialize counters for outliers
    white_win_outliers = 0
    black_win_outliers = 0
    draw_outliers = 0

    # Iterate through each game in the data
    for game_data in data:
        win_color = game_data['win_color']
        evaluations = game_data['evaluations']
        
        if not evaluations:
            continue
        
        # Get the final evaluation score
        final_score = evaluations[-1]
#        print(final_score) 
        if win_color == "white":
            if final_score >= 99000:
                white_win_outliers += 1
            else:
                white_win_scores.append(final_score)
        elif win_color == "black":
            if final_score <= -99000:
                black_win_outliers += 1
            else:
                black_win_scores.append(-final_score)  # Negate the score for Black
        elif win_color == "draw":
            if abs(final_score) == 1000:
                draw_outliers += 1
            else:
                draw_scores.append(final_score)

    # Calculate statistics for White wins
    if white_win_scores:
        white_max_score = max(white_win_scores)
        white_min_score = min(white_win_scores)
        white_avg_score = statistics.mean(white_win_scores)
    else:
        white_max_score = None
        white_min_score = None
        white_avg_score = None

    # Calculate statistics for Black wins
    if black_win_scores:
        black_max_score = max(black_win_scores)
        black_min_score = min(black_win_scores)
        black_avg_score = statistics.mean(black_win_scores)
    else:
        black_max_score = None
        black_min_score = None
        black_avg_score = None

    # Calculate statistics for Draws
    if draw_scores:
        draw_max_score = max(draw_scores)
        draw_min_score = min(draw_scores)
        draw_avg_score = statistics.mean(draw_scores)
    else:
        draw_max_score = None
        draw_min_score = None
        draw_avg_score = None

    # Print the statistics
    print("\nWhite Wins:")
    print(f"Number of Wins: {len(white_win_scores) + white_win_outliers}")
    print(f"Outliers (Checkmate): {white_win_outliers}")
    print(f"Max Score at Game End: {white_max_score / 100:.2f} pawns" if white_max_score is not None else "N/A")
    print(f"Min Score at Game End: {white_min_score / 100:.2f} pawns" if white_min_score is not None else "N/A")
    print(f"Avg Score at Game End: {white_avg_score / 100:.2f} pawns" if white_avg_score is not None else "N/A")

    print("\nBlack Wins:")
    print(f"Number of Wins: {len(black_win_scores) + black_win_outliers}")
    print(f"Outliers (Checkmate): {black_win_outliers}")
    print(f"Max Score at Game End: {black_max_score / 100:.2f} pawns" if black_max_score is not None else "N/A")
    print(f"Min Score at Game End: {black_min_score / 100:.2f} pawns" if black_min_score is not None else "N/A")
    print(f"Avg Score at Game End: {black_avg_score / 100:.2f} pawns" if black_avg_score is not None else "N/A")

    print("\nDraws:")
    print(f"Number of Draws: {len(draw_scores) + draw_outliers}")
    print(f"Outliers (Checkmate): {draw_outliers}")
    print(f"Max Score at Game End: {draw_max_score / 100:.2f} pawns" if draw_max_score is not None else "N/A")
    print(f"Min Score at Game End: {draw_min_score / 100:.2f} pawns" if draw_min_score is not None else "N/A")
    print(f"Avg Score at Game End: {draw_avg_score / 100:.2f} pawns" if draw_avg_score is not None else "N/A")

    # Plot histograms for White and Black wins
    plt.figure(figsize=(14, 6))

    # White Wins Histogram
    plt.subplot(1, 2, 1)
    if white_win_scores:
        plt.hist([score / 100 for score in white_win_scores], bins=50, color='green', alpha=0.7, edgecolor='black')
        plt.title('White Wins')
        plt.xlabel('Final Evaluation Score (pawns)')
        plt.ylabel('Frequency')
        plt.axvline(x=white_avg_score / 100, color='red', linestyle='dashed', linewidth=2, label=f'Avg: {white_avg_score / 100:.2f} pawns')
        plt.legend()
    else:
        plt.title('White Wins')
        plt.xlabel('Final Evaluation Score (pawns)')
        plt.ylabel('Frequency')
        plt.text(0.5, 0.5, 'No White Wins', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12)

    # Black Wins Histogram
    plt.subplot(1, 2, 2)
    if black_win_scores:
        plt.hist([score / 100 for score in black_win_scores], bins=50, color='blue', alpha=0.7, edgecolor='black')
        plt.title('Black Wins')
        plt.xlabel('Final Evaluation Score (pawns)')
        plt.ylabel('Frequency')
        plt.axvline(x=black_avg_score / 100, color='red', linestyle='dashed', linewidth=2, label=f'Avg: {black_avg_score / 100:.2f} pawns')
        plt.legend()
    else:
        plt.title('Black Wins')
        plt.xlabel('Final Evaluation Score (pawns)')
        plt.ylabel('Frequency')
        plt.text(0.5, 0.5, 'No Black Wins', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12)

    plt.tight_layout()
    plt.show()

# Example usage in a Jupyter notebook
# analyze_and_plot_game_statistics('evaluations.json')

def plot_game_evaluation(json_file, game_index):
    """
    Reads the JSON file containing game evaluations, retrieves a specific game by index,
    and plots the evaluation scores as a function of the game moves. It also prints the winner of the game.

    Parameters:
    - json_file (str): Path to the JSON file containing the game evaluations.
    - game_index (int): Index of the game to plot (0-based index).
    """
    # Load the existing data from the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Check if the game index is within the range
    if game_index < 0 or game_index >= len(data):
        print(f"Error: Game index {game_index} is out of range. Valid indices are 0 to {len(data) - 1}.")
        return

    # Retrieve the game data
    game_data = data[game_index]

    # Extract the evaluation scores
    evaluations = game_data['evaluations']
    game_length = game_data['game_length']
    win_color = game_data['win_color']

    # Convert evaluations to pawns
    evaluations_pawns = [score / 100 for score in evaluations]

    # Plot the evaluation scores
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, game_length + 1), evaluations_pawns, marker='o', linestyle='-')
    plt.title(f"Game {game_index + 1} Evaluation Scores")
    plt.xlabel("Move Number")
    plt.ylabel("Evaluation (pawns)")
    plt.grid(True)
    plt.axhline(y=0, color='gray', linestyle='--', label='Equal Position')
    plt.legend()

    # Annotate the plot with the winner
    if win_color == "white":
        plt.text(game_length, evaluations_pawns[-1], f'Winner: White', verticalalignment='bottom', horizontalalignment='right', color='green', fontsize=12)
    elif win_color == "black":
        plt.text(game_length, evaluations_pawns[-1], f'Winner: Black', verticalalignment='bottom', horizontalalignment='right', color='red', fontsize=12)
    elif win_color == "draw":
        plt.text(game_length, evaluations_pawns[-1], f'Result: Draw', verticalalignment='bottom', horizontalalignment='right', color='blue', fontsize=12)

    plt.show()

    # Print the winner
    if win_color == "white":
        print(f"Game {game_index + 1} Winner: White")
    elif win_color == "black":
        print(f"Game {game_index + 1} Winner: Black")
    elif win_color == "draw":
        print(f"Game {game_index + 1} Result: Draw")
    else:
        print(f"Game {game_index + 1} Result: Unknown")

# Example usage in a Jupyter notebook
# plot_game_evaluation('evaluations.json', game_index=0)