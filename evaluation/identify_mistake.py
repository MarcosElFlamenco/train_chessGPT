import chess
import chess.pgn
import io

def get_path(from_square, to_square):
    """
    Returns a list of squares between from_square and to_square for sliding pieces.
    If the move is not in a straight line or diagonal, returns an empty list.
    """
    path = []
    from_rank = chess.square_rank(from_square)
    from_file = chess.square_file(from_square)
    to_rank = chess.square_rank(to_square)
    to_file = chess.square_file(to_square)

    delta_rank = to_rank - from_rank
    delta_file = to_file - from_file

    step_rank = 0
    step_file = 0

    if delta_rank > 0:
        step_rank = 1
    elif delta_rank < 0:
        step_rank = -1

    if delta_file > 0:
        step_file = 1
    elif delta_file < 0:
        step_file = -1

    # Check if the move is along a straight line or diagonal
    if delta_rank != 0 and delta_file != 0 and abs(delta_rank) != abs(delta_file):
        return path  # Not a valid sliding move

    current_rank = from_rank + step_rank
    current_file = from_file + step_file

    while (current_rank != to_rank) or (current_file != to_file):
        square = chess.square(current_file, current_rank)
        path.append(square)
        current_rank += step_rank
        current_file += step_file

    return path

def explain_illegal_move(board, san_move):
    """
    Analyzes an illegal SAN move and categorizes the reason for its illegality.

    Parameters:
    - board (chess.Board): The current state of the chess board.
    - san_move (str): The move in Standard Algebraic Notation (e.g., 'Nf3').

    Returns:
    - str: A message indicating the category of illegality or the UCI move if legal.
    """
    try:
        move = board.parse_san(san_move)
    except ValueError as e:
        return f"1: The move makes no sense in chess notation. ({e})"

    if move in board.legal_moves:
        return f"Legal move. UCI: {move.uci()}"

    # If the move is not legal, determine why
    piece = board.piece_at(move.from_square)

    if piece is None:
        return "2: There is no piece at the source square."

    # Check if the piece can make that move ignoring checks
    if not board.is_pseudo_legal(move):
        return "2: The piece cannot move that way."

    # For sliding pieces, check for obstruction
    if piece.piece_type in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
        path = get_path(move.from_square, move.to_square)
        for sq in path:
            blocking_piece = board.piece_at(sq)
            if blocking_piece:
                if blocking_piece.color != piece.color:
                    return "3: The move attempts to move through an enemy piece."
                else:
                    return "4: The move attempts to move through a friendly piece."

    # Check if moving this piece would leave the king in check
    board_copy = board.copy()
    board_copy.push(move)
    if board_copy.is_check():
        return "5: The move leaves the king in check or does not resolve check."

    # If none of the above, categorize as miscellaneous
    return "6: The move is illegal for an unspecified reason."

def san_to_uci(board, san_move):
    """
    Converts a SAN move to UCI format given the current board state.

    Parameters:
    - board (chess.Board): The current board state.
    - san_move (str): The move in Standard Algebraic Notation (e.g., 'Nf3').

    Returns:
    - str: The move in UCI format (e.g., 'g1f3'), or an error message.
    """
    explanation = explain_illegal_move(board, san_move)
    if explanation.startswith("Legal move."):
        return explanation.split("UCI: ")[1]
    else:
        return explanation

def display_board(board):
    """
    Prints the current board in a human-readable format.
    """
    print(board)

def pgn_to_uci_moves(pgn_string):
    """
    Converts all SAN moves in a PGN game to UCI format.

    Parameters:
    - pgn_string (str): The PGN content as a string.

    Returns:
    - List of tuples containing move numbers, SAN moves, and UCI moves or explanations.
    """
    uci_moves = []
    pgn_io = io.StringIO(pgn_string)
    game = chess.pgn.read_game(pgn_io)

    if game is None:
        print("No game found in the PGN input.")
        return uci_moves

    board = game.board()
    move_number = 1

    for move in game.mainline_moves():
        san = board.san(move)  # SAN move
        uci = move.uci()        # UCI move
        uci_moves.append((move_number, san, uci))
        board.push(move)
        # Increment move number after Black's move
        if board.turn == chess.WHITE:
            move_number += 1

    return uci_moves

def interactive_mode():
    """
    Provides an interactive mode where users can input SAN moves and receive UCI equivalents or explanations.
    """
    board = chess.Board()
    print("Initial Board:")
    display_board(board)

    while not board.is_game_over():
        try:
            san_move = input(f"{'White' if board.turn == chess.WHITE else 'Black'} to move: ")
            if san_move.lower() in ['exit', 'quit']:
                print("Exiting interactive mode.")
                break

            explanation = explain_illegal_move(board, san_move)
            if explanation.startswith("Legal move."):
                uci_move = explanation.split("UCI: ")[1]
                board.push_uci(uci_move)
                print(f"Move executed: {san_move} -> {uci_move}")
            else:
                print(f"Illegal move: {explanation}")

            display_board(board)

        except KeyboardInterrupt:
            print("\nExiting interactive mode.")
            break

def main():
    """
    Main function to demonstrate converting SAN moves to UCI and explaining illegal moves.
    """
    # Example PGN
    pgn_string = """
    [Event "F/S Return Match"]
    [Site "Belgrade, Serbia JUG"]
    [Date "1992.11.04"]
    [Round "29"]
    [White "Fischer, Robert J."]
    [Black "Spassky, Boris V."]
    [Result "1/2-1/2"]

    1. e4 e5 2. Nf3 Nc6 3. Bb5 a6
    """

    print("Converting PGN moves to UCI:")
    moves = pgn_to_uci_moves(pgn_string)

    for move_num, san, uci in moves:
        print(f"Move {move_num}: {san} -> {uci}")

    print("\nInteractive Mode:")
    print("You can input SAN moves, and the script will convert them to UCI or explain why they're illegal.")
    print("Type 'exit' or 'quit' to end the interactive session.\n")
    interactive_mode()

if __name__ == "__main__":
    main()
