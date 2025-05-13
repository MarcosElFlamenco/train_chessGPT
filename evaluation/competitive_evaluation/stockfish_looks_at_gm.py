import argparse
import chess
import chess.pgn
import chess.engine
import json
import os
import time

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate chess games using Stockfish.")
    parser.add_argument("--pgn_file", required=True, help="Path to the PGN file.")
    parser.add_argument("--output_file", required=True, help="Path to the output JSON file.")
    parser.add_argument("--max_games", type=int, default=None, help="Maximum number of games to process.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    parser.add_argument("--stockfish_path", required=True, help="Path to the Stockfish binary.")
    parser.add_argument("--time_per_move", type=float, default=0.1, help="Time limit per move for Stockfish evaluation (in seconds).")
    return parser.parse_args()

def load_existing_data(output_file):
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            data = json.load(f)
        return data
    return []

def save_data(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

def evaluate_game(game, engine, time_per_move, verbose):
    board = game.board()
    evaluations = []
    moves = list(game.mainline_moves())

    for i, move in enumerate(moves):
        board.push(move)
        start_time = time.perf_counter()
        result = engine.analyse(board, chess.engine.Limit(time=time_per_move))
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        score = result["score"].white().score(mate_score=100000)  # From White's perspective
        evaluations.append(score)

        if verbose:
            print(f"Move {i + 1}: {move}, Evaluation: {score / 100:.2f} pawns, Time: {elapsed_time:.6f} seconds")

    # Determine the winner
    outcome = game.headers.get("Result")
    win_color = None
    if outcome == "1-0":
        win_color = "white"
    elif outcome == "0-1":
        win_color = "black"
    elif outcome == "1/2-1/2":
        win_color = "draw"

    return {
        "game_length": len(moves),
        "win_color": win_color,
        "evaluations": evaluations
    }

def main():
    args = parse_arguments()

    if args.verbose:
        print(f"Loading existing data from {args.output_file}...")
    existing_data = load_existing_data(args.output_file)
    start_game_index = len(existing_data)

    if args.verbose:
        print(f"Starting from game index {start_game_index}...")

    with open(args.pgn_file) as pgn_file:
        engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)
        game_index = 0

        while True:
            try:
                game = chess.pgn.read_game(pgn_file)
            except Exception as e:
                print(f"got the following error {e}")

            if game is None:
                break

            game_index += 1
            if game_index <= start_game_index:
                continue

            if args.max_games is not None and game_index > args.max_games:
                break

            if args.verbose:
                print(f"Evaluating game {game_index}...")

            game_data = evaluate_game(game, engine, args.time_per_move, args.verbose)
            existing_data.append(game_data)

            if args.verbose:
                print(f"Game {game_index} evaluated. Total games: {len(existing_data)}")

            if len(existing_data) % 10 == 0:
                print(f"Saving data after game {game_index}...")
                save_data(existing_data, args.output_file)

        if len(existing_data) % 10 != 0:
            if args.verbose:
                print(f"Saving remaining data after game {game_index}...")
            save_data(existing_data, args.output_file)

        engine.quit()

if __name__ == "__main__":
    main()