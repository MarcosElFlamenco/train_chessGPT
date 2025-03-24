import pandas as pd

def compare_csvs():
    # Define file paths
    small_csv_path = "eval_datasets/random2128games.csv"  # 10K lines
    large_csv_path = "../data/random_dataset/random.csv"  # 16M lines

    # Define the column to compare (assuming first column, adjust if needed)
    column_name = "transcript"  # Replace with actual column name

    # Load small CSV into a set (efficient lookup)
    small_df = pd.read_csv(small_csv_path, usecols=[column_name])
    small_set = set(small_df[column_name].astype(str))  # Ensure consistent datatype

    del small_df  # Free memory

    # Count matches in large file
    match_count = 0
    chunk_size = 100000  # Process in chunks to avoid memory overload

    for chunk in pd.read_csv(large_csv_path, usecols=[column_name], chunksize=chunk_size):
        match_count += chunk[column_name].astype(str).isin(small_set).sum()

    print(f"Number of matching lines: {match_count}")

import chess.pgn
import io
import matplotlib.pyplot as plt
import statistics

# ---------------------------------------
# Helper function to assign IDs to pieces
# ---------------------------------------
# We'll track each piece on the board via a unique ID.
# For promotions, we keep the same ID.
# For castling, we also track the rook.

def assign_initial_piece_ids(board):
    # piece_id_for_square maps square -> unique_piece_id
    # move_count_for_id maps unique_piece_id -> count of moves
    piece_id_for_square = {}
    move_count_for_id = {}

    # Mapping from python-chess piece_type to letter
    piece_type_map = {chess.PAWN: "P", chess.KNIGHT: "N", chess.BISHOP: "B", chess.ROOK: "R", chess.QUEEN: "Q", chess.KING: "K"}

    for square, piece in board.piece_map().items():
        color_prefix = "w" if piece.color == chess.WHITE else "b"
        piece_letter = piece_type_map[piece.piece_type]
        sq_name = chess.square_name(square)  # e.g. "a1"
        unique_id = f"{color_prefix}{piece_letter}_{sq_name}"
        piece_id_for_square[square] = unique_id
        move_count_for_id[unique_id] = 0

    return piece_id_for_square, move_count_for_id


def update_piece_positions(board, move, piece_id_for_square, move_count_for_id):
    # The piece that just moved:
    from_sq = move.from_square
    to_sq = move.to_square

    # If we can't find the piece in piece_id_for_square, it might be a new promotion scenario,
    # or the dictionary is out of sync. We'll try to handle it gracefully.
    if from_sq not in piece_id_for_square:
        # We'll do a sync. But typically this should not happen.
        # For safety, attempt to re-sync entire board
        resync_piece_ids(board, piece_id_for_square, move_count_for_id)

    if from_sq in piece_id_for_square:
        mover_id = piece_id_for_square[from_sq]
        move_count_for_id[mover_id] += 1

        # If there's a capture, remove that piece from the dictionary
        if to_sq in piece_id_for_square:
            captured_id = piece_id_for_square[to_sq]
            del piece_id_for_square[to_sq]
            del move_count_for_id[captured_id]

        # If this is a promotion, the piece changes type, but we keep the same ID.
        if move.promotion:
            color_prefix = mover_id[0]  # 'w' or 'b'
            # Map promotion piece_type to letter
            promo_letter = {chess.QUEEN: "Q", chess.ROOK: "R", chess.BISHOP: "B", chess.KNIGHT: "N"}[move.promotion]
            # e.g. wQ
            # We'll keep the same square suffix to keep ID stable, or you can rename. We'll do a simpler approach.
            # For clarity, let's rename the ID to reflect new piece type.
            # e.g. wP_a7 could become wQ_a7. We'll keep the same suffix.
            old_suffix = "_".join(mover_id.split("_")[1:])
            new_id = f"{color_prefix}{promo_letter}_{old_suffix}"

            # Move the move count over to the new ID
            old_count = move_count_for_id[mover_id]
            move_count_for_id[new_id] = old_count
            del move_count_for_id[mover_id]
            mover_id = new_id

        # Move the piece ID to the new square
        piece_id_for_square[to_sq] = mover_id
        del piece_id_for_square[from_sq]

    # Handle castling: if it's castling, the rook moves automatically.
    # Let's check board state after the push to see if any rooks moved from -> to.

    # We do a re-sync to handle the rook movement automatically.
    resync_piece_ids(board, piece_id_for_square, move_count_for_id)


def resync_piece_ids(board, piece_id_for_square, move_count_for_id):
    # This will fully re-sync squares with existing IDs.
    # We'll match by color and piece type. If an existing ID can't be matched, we'll remove it.
    # If a new piece is found, we'll create a new ID.

    # Build a reverse mapping from ID -> current square(s?) in the dictionary.
    # In a perfect scenario, each ID is at exactly 1 square.
    id_to_squares = {}
    for sq, pid in piece_id_for_square.items():
        if pid not in id_to_squares:
            id_to_squares[pid] = []
        id_to_squares[pid].append(sq)

    # Now let's build a new dictionary from scratch.
    new_piece_id_for_square = {}

    piece_type_map = {chess.PAWN: "P", chess.KNIGHT: "N", chess.BISHOP: "B", chess.ROOK: "R", chess.QUEEN: "Q", chess.KING: "K"}

    # We'll try to match existing IDs with squares of the same color + piece type.

    # Step 1: create a list of squares/pieces from the board.
    squares_and_pieces = list(board.piece_map().items())

    # Attempt to match each square/piece with an existing ID.
    # We'll do a best effort: if there's exactly one existing ID that matches color + letter, reuse it. Otherwise, create a new ID.

    used_ids = set()

    for sq, piece in squares_and_pieces:
        color_prefix = "w" if piece.color == chess.WHITE else "b"
        piece_letter = piece_type_map[piece.piece_type]
        # find if there's an existing ID that starts with color_letter and that ends with the same letter if no promotion occurred.
        # Promotions might cause mismatch, but let's do a simpler approach.

        candidate_ids = []
        for pid, squares in id_to_squares.items():
            # check if pid is already used
            if pid in used_ids:
                continue
            # check if pid has the same color prefix
            if pid[0] == color_prefix:
                # check if the second char matches piece_letter
                # e.g. wP or wQ
                # but we might have a promoted piece. Let's just check if pid[1] == piece_letter or not.
                if len(pid) > 1 and pid[1] == piece_letter:
                    candidate_ids.append(pid)

        if len(candidate_ids) == 1:
            chosen_id = candidate_ids[0]
        else:
            # create a new ID
            sq_name = chess.square_name(sq)
            chosen_id = f"{color_prefix}{piece_letter}_{sq_name}"
            if chosen_id not in move_count_for_id:
                move_count_for_id[chosen_id] = 0

        new_piece_id_for_square[sq] = chosen_id
        used_ids.add(chosen_id)

    piece_id_for_square.clear()
    piece_id_for_square.update(new_piece_id_for_square)


# -------------------------------------------------------
# Main function to parse a single PGN and count moves.
# Returns a list of dicts for each game with individual piece counts.
# -------------------------------------------------------

def process_pgn_games(pgn_texts):
    all_game_counts = []

    for pgn_text in pgn_texts:
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if not game:
            continue
        board = game.board()
        piece_id_for_square, move_count_for_id = assign_initial_piece_ids(board)

        for move in game.mainline_moves():
            update_piece_positions(board, move, piece_id_for_square, move_count_for_id)
            board.push(move)

        # Now move_count_for_id holds the number of moves for each unique piece
        all_game_counts.append(move_count_for_id.copy())

    return all_game_counts


def process_csv_file(csv_path, pgn_column):
    # Reads the CSV, extracts PGN from pgn_column, processes each game.
    df = pd.read_csv(csv_path)
    pgn_texts = df[pgn_column].dropna().tolist()
    all_game_counts = process_pgn_games(pgn_texts)
    return all_game_counts


def average_piece_moves(all_game_counts):
    # all_game_counts is a list of dictionaries: { piece_id: moves }
    # We want to compute the average moves per piece_id across all games.
    # But each game might have different piece IDs. We'll unify them.

    from collections import defaultdict
    # sums[piece_id] = sum of moves across games
    # counts[piece_id] = number of games in which piece_id appeared
    sums = defaultdict(int)
    presence_counts = defaultdict(int)

    for game_dict in all_game_counts:
        for pid, moves in game_dict.items():
            sums[pid] += moves
            presence_counts[pid] += 1

    # Now compute average
    averages = {}
    for pid in sums:
        averages[pid] = sums[pid] / presence_counts[pid]

    return averages

# -------------------------------------------
# Compare two CSV files, produce bar chart
# -------------------------------------------

def main(file1, file2, pgn_column):
    all_game_counts_1 = process_csv_file(file1, pgn_column)
    all_game_counts_2 = process_csv_file(file2, pgn_column)

    avg_moves_1 = average_piece_moves(all_game_counts_1)
    avg_moves_2 = average_piece_moves(all_game_counts_2)

    # We'll plot only pieces that appear frequently, or we can unify all IDs.
    # Let's unify the keys.
    all_keys = set(avg_moves_1.keys()).union(set(avg_moves_2.keys()))
    # Sort them in a somewhat stable manner.
    all_keys = sorted(all_keys)

    # We'll build parallel arrays.
    values_1 = [avg_moves_1.get(k, 0) for k in all_keys]
    values_2 = [avg_moves_2.get(k, 0) for k in all_keys]

    import numpy as np
    x = np.arange(len(all_keys))

    plt.figure(figsize=(12,6))
    plt.bar(x - 0.2, values_1, width=0.4, label="File1")
    plt.bar(x + 0.2, values_2, width=0.4, label="File2")
    plt.xticks(x, all_keys, rotation=90)
    plt.xlabel("Unique Piece IDs")
    plt.ylabel("Average Moves")
    plt.title("Comparison of Average Individual Piece Moves")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Usage example (replace with real file paths and column name):
main("eval_datasets/kasparov2128games.csv", "eval_datasets/random2128games.csv", "transcript")
