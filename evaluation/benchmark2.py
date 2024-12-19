import torch
import torch.nn.functional as F
import numpy as np
import pickle
import os
import re
import chess
import sys
from model import GPT, GPTConfig  # Ensure model.py is in the same directory or adjust the import path accordingly
from tqdm import tqdm  # Added for progress bars

def remove_prefix_from_state_dict(state_dict, prefix='_orig_mod.'):
    """
    Removes a specified prefix from the state_dict keys.

    Args:
        state_dict (dict): Original state_dict with prefixed keys.
        prefix (str): Prefix to remove from each key.

    Returns:
        dict: Updated state_dict with prefixes removed.
    """
    flag = False
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]  # Remove the prefix
            flag = True
        else:
            new_key = key  # Keep the key as is
        new_state_dict[new_key] = value
    if flag:
        print('Unwanted prefixes were found in the checkpoint and have been removed.')
    return new_state_dict

def load_meta(data_dir):
    """
    Loads meta information from meta.pkl.

    Args:
        data_dir (str): Directory where meta.pkl is located.

    Returns:
        tuple: (vocab_size, stoi, itos)
    """
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        vocab_size = meta['vocab_size']
        stoi = meta['stoi']  # String to index mapping
        itos = meta['itos']  # Index to string mapping
        return vocab_size, stoi, itos
    else:
        raise FileNotFoundError(f"Meta file not found at {meta_path}")

def tokenize(string, stoi):
    """
    Converts string to list of token IDs.

    Args:
        string (str): Input string.
        stoi (dict): String to index mapping.

    Returns:
        np.ndarray: Array of token IDs.
    """
    # Handle unknown characters by assigning a default index (e.g., <unk>)
    unk_index = stoi.get('<unk>', 0)
    return np.array([stoi.get(c, unk_index) for c in string], dtype=np.int64)

def detokenize(tokens, itos):
    """
    Converts list of token IDs back to string.

    Args:
        tokens (list): List of token IDs.
        itos (dict): Index to string mapping.

    Returns:
        str: Reconstructed string.
    """
    return ''.join([itos.get(t, '<unk>') for t in tokens])

def load_model(checkpoint_path, device):
    """
    Loads the GPT model from a checkpoint.

    Args:
        checkpoint_path (str): Path to the model checkpoint.
        device (str): Device to load the model on ('cuda' or 'cpu').

    Returns:
        GPT: Loaded GPT model.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = checkpoint['model_args']
    
    # Initialize model
    config = GPTConfig(**model_args)
    model = GPT(config)
    
    # Load state dict
    model.load_state_dict(remove_prefix_from_state_dict(checkpoint['model']))
    model.to(device)
    model.eval()  # Set model to evaluation mode
    return model

def generate_next_move(model, prompt_pgn, move_number, stoi, itos, device, verbose=False, max_length=1023, temperature=1.0, troubleshoot_verbose = False):
    """
    Generates the next move in PGN format using the GPT model.

    Args:
        model (GPT): The GPT model.
        prompt_pgn (str): Current PGN string up to the last move.
        move_number (int): The current move number to generate (e.g., 3 for "3.").
        stoi (dict): String to index mapping.
        itos (dict): Index to string mapping.
        device (str): Device to run the model on ('cuda' or 'cpu').
        verbose (bool): Enable verbose output for debugging.
        max_length (int): Maximum sequence length the model can handle.
        temperature (float): Sampling temperature for diversity.

    Returns:
        str: The move generated by the model.
    """
    # Determine if it's White's or Black's turn
    is_white_turn = (move_number % 2) == 1
    generated_move = ""
    attempts = 0
    max_attempts = 10  # Prevent infinite loops

    while attempts < max_attempts:
        if troubleshoot_verbose and False:
            print(f"Currently prompting with the following pgn {prompt_pgn}")

        # Tokenize input
        input_tokens = tokenize(prompt_pgn, stoi)
        input_tensor = torch.from_numpy(input_tokens).unsqueeze(0).to(device)  # Shape: (1, seq_len)

        if input_tensor.size(1) > max_length:
            return None
            raise ValueError(f"Input sequence length {input_tensor.size(1)} exceeds maximum block size {max_length}.")

        with torch.no_grad():
            # Forward pass
            try:
                logits, _ = model(input_tensor)  # logits shape: (1, seq_len, vocab_size)
            except Exception as e:
                if verbose:
                    print(f"Got the following error {e}")
                    print(f"we got to prompting with this prompt {prompt_pgn} of length {len(prompt_pgn)}")
                break

            # Focus on the last token's logits to predict the next token
            last_token_logits = logits[:, -1, :] / temperature  # Shape: (1, vocab_size)

            # Compute probabilities
            probabilities = F.softmax(last_token_logits, dim=-1)  # Shape: (1, vocab_size)

            # Sample a token
            sampled_token = torch.multinomial(probabilities, num_samples=1)  # Shape: (1, 1)

            # Convert token ID to character
            sampled_char = detokenize(sampled_token.cpu().tolist()[0], itos)

        # Check if space is generated, indicating the move is complete
        if sampled_char == ' ':
            break

        # Append the generated character to the move
        generated_move += sampled_char

        # Update the prompt_pgn to include the generated characters so far
        prompt_pgn += sampled_char
        attempts += 1

    return generated_move

def parse_pgn(pgn_string):
    """
    Parses the PGN string into individual moves.

    Args:
        pgn_string (str): The input PGN string.

    Returns:
        list: List of individual moves in order.
    """
    # Remove header lines (lines within square brackets)
    pgn_no_headers = re.sub(r'\[.*?\]', '', pgn_string)
    # Remove comments
    pgn_clean = re.sub(r'\{[^}]*\}', '', pgn_no_headers)
    # Remove move numbers
    pgn_clean = re.sub(r'\d+\.', '', pgn_clean)
    # Remove result indicators
    pgn_clean = re.sub(r'1-0|0-1|1/2-1/2|\*', '', pgn_clean)
    # Split by spaces
    moves = pgn_clean.strip().split()
    return moves

def validate_move_precomputed(precomputed_moves, position_index, move_san):
    """
    Validates if the move in SAN notation is in the precomputed legal moves.
    
    Args:
        precomputed_moves (list): List of sets of legal moves for each position.
        position_index (int): The index of the current position.
        move_san (str): Move in Standard Algebraic Notation.
    
    Returns:
        bool: True if move is legal, False otherwise.
    """
    if position_index < len(precomputed_moves):
        return move_san in precomputed_moves[position_index]
    else:
        return False

def validate_move(board, move_san):
    """
    Validates if the move in SAN notation is legal.
    Args:
        board (chess.Board): Current game board.
        move_san (str): Move in Standard Algebraic Notation.
    Returns:
        bool: True if move is legal, False otherwise.
    """
    try:
        move = board.parse_san(move_san)
        return move in board.legal_moves
    except ValueError:
        return False

def precompute_legal_moves(pgn_files, output_file, verbose=False, troubleshoot_verbose = False, max_moves = 0 ):
    """
    Precomputes legal moves for each position in the given PGN files and saves them.

    Args:
        pgn_files (list): List of PGN file paths.
        output_file (str): Output file path to save the precomputed moves.
        verbose (bool): Enable verbose output for debugging.
    """

    precomputed_games = []
    for pgn_file in pgn_files:
        with open(pgn_file, 'r') as f:
            content = f.read()
            # Split the content into segments separated by two newlines
            segments = content.strip().split('\n\n')
            # Filter out segments that contain only headers
            games = [segment[:1023] for segment in segments if not all(line.startswith('[') for line in segment.split('\n'))]
            for game in games:
                print(len(game))

            # Wrap games in tqdm for progress bar
            for game_index, game in enumerate(tqdm(games, desc=f"Processing {pgn_file}", unit="game")):
                precomputed_moves = []
                moves_original = parse_pgn(game)
                if max_moves == 0 or max_moves > 180:
                    moves_length_filtered = moves_original[:max_moves]
                else:
                    moves_length_filtered = moves_original[:-2]

                board = chess.Board()
                for move_index, move in enumerate(moves_length_filtered):
                    legal_moves_san = [board.san(m) for m in board.legal_moves]
                    precomputed_moves.append(set(legal_moves_san))
                    try:
                        board.push_san(move)
                    except ValueError:
                        if verbose:
                            print(f"Invalid move '{move}' from original pgn game in game {game_index+1}, move {move_index+1}")
                        break  # Skip to next game if move is invalid
                game_info = {
                    "game_moves" : moves_length_filtered,
                    "precomputed_moves" : precomputed_moves
                }
                precomputed_games.append(game_info)

    # Save precomputed moves and games

    with open(output_file, 'wb') as f:
        pickle.dump(precomputed_games, f)
    print(f"Precomputed legal moves and games saved to {output_file}")

def run_validation_single_model(args):
    """
    Runs the move generation and validation process.

    Args:
        args: Parsed command-line arguments.
    """
    storage_file = 'evaluation/test_results.json'
    import json
    with open(storage_file, "r") as f:
        test_results = json.load(f)

    model2 = (args.checkpoint).split('/')[-1].split('.')[0]
    model_split = model2.split('_')
    model_iters = model_split[-1]
    model_name = model2[:-(len(model_iters) + 1)]

    dataset = (args.dataset).split('/')[-1].split('.')[0]


    if (model_name in test_results) and (model_iters in test_results[model_name]) and (dataset in test_results[model_name][model_iters]):
        print(f"The model {model_name} after {model_iters} iters has already been evaluated on {dataset} dataset")
        return

    # Load meta information
    vocab_size, stoi, itos = load_meta(args.data_dir)
    print(f"Vocab Size: {vocab_size}")

    # Load the model
    model = load_model(args.checkpoint, args.device)
    print(f"Model loaded successfully from {args.checkpoint}.\n")

    # Load precomputed legal moves
    with open(args.dataset, 'rb') as f:
        precomputed_games = pickle.load(f)

    print(f"Precomputed legal moves loaded from {args.dataset}")

    # Parse the input PGN into individual games
    total_generated_moves = 0
    illegal_moves_count = 0
    illegal_move_index = 0
    game_length_sums = 0
    illegal_moves_examples = []

    # Stats per game
    game_stats = []
    game_dics = []
    # Wrap games in tqdm for progress bar
    for game_index, game_info in enumerate(tqdm(precomputed_games, desc="Evaluating games", unit="game")):

        moves = game_info["game_moves"]
        precomputed_moves = game_info["precomputed_moves"]
        game_length_sums += len(moves)
        if args.troubleshoot_verbose:
            print(f"this is game number {game_index}")
            print(f"moves length is {len(moves)}")
            print(f"precomputed length is {len(precomputed_moves)}")
            print(f"The first few are {precomputed_moves[0]}, {precomputed_moves[1]}, {precomputed_moves[2]}")
        current_pgn = ';'

        game_total_moves = 0
        game_illegal_moves = 0
        illegal_move_indices = []

        for idx, move in enumerate(tqdm(moves, desc=f'Handling game {game_index}', unit = 'move') ):
            move_number = idx + 1
            is_white_turn = (move_number % 2) == 1

            # Prepare the prompt for the model
            move_prefix = ""
            if is_white_turn:
                move_prefix = f"{(move_number + 1) // 2}."
                if args.spaced:
                    move_prefix += ' '

            prompt_pgn = current_pgn + move_prefix
            if args.verbose:
                print(f"Prompting with the following pgn: {prompt_pgn}")
            if args.troubleshoot_verbose:
                print(f"Prompting with the following pgn: {prompt_pgn}")
                print(f"valid moves include {precomputed_moves[idx]}")
            # Generate the next move
            generated_move = generate_next_move(
                model, prompt_pgn, move_number, stoi, itos, args.device,
                verbose=args.verbose, temperature=args.temperature, troubleshoot_verbose=args.troubleshoot_verbose
            )
            if generated_move == None:
                print(f"When trying to generate move number {idx}, we exceded the size limtis of the model") 
                break
           # print(f'For move number {move_number}, the model generated {generated_move}, it was validated against the following moves {precomputed_moves[position_index]}')

            total_generated_moves += 1
            game_total_moves += 1

            # Validate the generated move using precomputed legal moves
            is_legal = validate_move_precomputed(precomputed_moves, idx, generated_move)

            if is_legal:
                if args.verbose:
                    print(f"Move {move_number}: Generated move '{generated_move}' is LEGAL.\n")
            else:
                illegal_move_indices.append(move_number)
                illegal_moves_count += 1
                game_illegal_moves += 1
                illegal_move_index += move_number
                illegal_moves_examples.append((move_number, generated_move, "Illegal move generated"))
                if args.verbose:
                    print(f"Move {move_number}: Generated move '{generated_move}' is ILLEGAL.\n")
                if args.troubleshoot_verbose:

                    print(f"Move {move_number}: When prompted with \n {prompt_pgn} \n Generated move '{generated_move}' is ILLEGAL.\n")
                    print(f'The valid moves were: {precomputed_moves[idx]}')

            current_pgn = prompt_pgn + move + ' '
        game_error_frequency = 1.0
        if game_total_moves != 0:
            game_error_frequency = game_illegal_moves/game_total_moves
        game_dic = {
               "error_indices" : illegal_move_indices,
               "num_moves" : game_total_moves,
               "num_errors" : game_illegal_moves,
               "error_frequency" : game_error_frequency
                }
        game_dics.append(game_dic)
        

    # Reporting
#ALL GAME STATS
    all_games_total_generated_moves = 0
    all_games_illegal_moves_count = 0
    max_freq = 0.0
    min_freq = 100.0
    if args.stats_per_game:
        print("--- Frequency of Invalid Move Generation Per Game ---")
    # Per-Game Statistics
    for stats in game_stats:
        if args.stats_per_game:
            print(f"Game {stats['game_index']}:")
            print(f"  Total Moves: {stats['total_moves']}")
            print(f"  Illegal Moves: {stats['illegal_moves']}")
            print(f"  Frequency: {stats['frequency']:.2f}%\n")
        all_games_total_generated_moves += stats["total_moves"]
        all_games_illegal_moves_count += stats["illegal_moves"]
        freq = stats["frequency"]
        max_freq = max(max_freq, freq)
        min_freq = min(min_freq, freq)


    print("\n--- Validation Report ---")
    print(f"Model: {args.checkpoint.split('/')[-1].split('.')[0]} Dataset: {args.dataset.split('/')[-1].split('.')[0]}")
    print(f"Total Generated Moves: {total_generated_moves}")
    print(f"Illegal Moves Generated: {illegal_moves_count}")
    if total_generated_moves > 0:
        frequency = (illegal_moves_count / total_generated_moves) * 100
        print(f"Overall Frequency of Illegal Moves: {frequency:.2f}%\n")
        
    else:
        print("No moves were generated.\n")
        frequency = 0
    if illegal_moves_count > 0:
        avg_index = illegal_move_index/illegal_moves_count
        print(f"The average index of illegal moves is {avg_index:.2f}")
    else:
        print("No illegal moves generated")
        avg_index = 0
    num_games = len(precomputed_games)
    if num_games > 0:
        avg_game_length = game_length_sums/num_games
        print(f"Average game length is {avg_game_length:.2f}")
    else:
        print("no games")
        avg_game_length = 0
    print(f"  Max frequency: {max_freq:.2f}%\n")
    print(f"  Min frequency: {min_freq:.2f}%\n")

    ##update file
    add_result(test_results, args.checkpoint, args.dataset, game_dics)

    with open(storage_file, "w") as f:
        json.dump(test_results,f)
 

def add_result(test_results, model_name, dataset_name, game_dics):
    model2 = model_name.split('/')[-1].split('.')[0]
    model_split = model2.split('_')
    model_iters = model_split[-1]
    model_name = model2[:-(len(model_iters) + 1)]
    dataset = dataset_name.split('/')[-1].split('.')[0]

    if model_name not in test_results:
        test_results[model_name] = {}
    if model_iters not in test_results[model_name]:
        test_results[model_name][model_iters] = {}
    test_results[model_name][model_iters][dataset] = game_dics



def precompute_legal_moves_wrapper(args):
    """
    Wrapper for precompute_legal_moves to handle verbosity.

    Args:
        args: Parsed command-line arguments.
    """
    precompute_legal_moves(args.pgn_files, args.output_file, verbose=args.verbose, troubleshoot_verbose=args.troubleshoot_verbose)

def main():
    import argparse

    parser = argparse.ArgumentParser(description="GPT Model Inference and Move Validation for Chess PGN")
    subparsers = parser.add_subparsers(dest='mode', help='Mode of operation')

    # Subparser for precompute mode
    precompute_parser = subparsers.add_parser('precompute', help='Precompute legal moves for PGN files')
    precompute_parser.add_argument('--pgn_files', nargs='+', required=True, help='List of PGN files to process')
    precompute_parser.add_argument('--output_file', type=str, required=True, help='Output file to save precomputed moves')
    precompute_parser.add_argument('--verbose', action='store_true', help='Enable verbose output for debugging')
    precompute_parser.add_argument('--troubleshoot_verbose', action='store_true', help='Enable verbose output for debugging')

    # Subparser for evaluation mode
    eval_parser = subparsers.add_parser('eval', help='Evaluate model using precomputed legal moves')
    eval_parser.add_argument('--checkpoints', type=str, required=True, nargs= '+', help='Path to the model checkpoint (e.g., checkpoint.pth)')
    eval_parser.add_argument('--datasets', type=str, required=True, nargs = '+',help='File containing precomputed legal moves')
    eval_parser.add_argument('--data_dir', type=str, default='data/openwebtext', help='Directory where meta.pkl is located')
    eval_parser.add_argument('--results_file', type=str, default='data/openwebtext', help='Directory where meta.pkl is located')
    eval_parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run the model on')
    eval_parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature for move generation (default: 1.0)')
    eval_parser.add_argument('--verbose', action='store_true', help='Enable verbose output for debugging')
    eval_parser.add_argument('--troubleshoot_verbose', action='store_true', help='Enable verbose output for debugging')
    eval_parser.add_argument('--graph', action='store_true', help='Enable graph output for debugging')
    eval_parser.add_argument('--illegal_info', action='store_true', help='Enable detailed illegal move information')
    eval_parser.add_argument('--spaced', action='store_true', help='Enable detailed illegal move information')
    eval_parser.add_argument('--stats_per_game', action='store_true', help='Enable detailed illegal move information')

    args = parser.parse_args()


    if args.mode == 'precompute':
        precompute_legal_moves_wrapper(args)
    elif args.mode == 'eval':
        for checkpoint in args.checkpoints:
            for dataset in args.datasets:
                args.checkpoint = checkpoint
                args.dataset = dataset
                run_validation_single_model(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
