import chess
import chess.pgn
import io

def get_legal_moves_from_pgn(pgn_string):
    # Parse the PGN game
    game = chess.pgn.read_game(io.StringIO(pgn_string))
    
    # Check if the PGN is valid
    if not game:
        print("Invalid PGN string.")
        return
    
    # Get the final board position after all moves in PGN
    board = game.end().board()
    
    # Print all legal moves in the final position
    print("Legal moves from this position:")
    legal_moves = [ board.san(move) for move in board.legal_moves]
    print(legal_moves)

def main():
    while(True):
        print("Enter the PGN string for the game:")
        pgn_input = input("> ")
        get_legal_moves_from_pgn(pgn_input)

if __name__ == "__main__":
    main()