import argparse
import os
import pickle
import chess
import chess.engine
import torch
import torch.nn.functional as F
import numpy as np
from model import GPT, GPTConfig  # Ensure model.py is available
import json
import math

MODEL_DIR = "../models"
ELO_RESULTS_FILE = 'evaluation/elo_results.json'
torch.manual_seed(42)

# --- Utility functions (load_json_file, save_json_file, load_meta, load_model, tokenize, detokenize) ---
def load_json_file(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return {}

def save_json_file(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def load_meta(data_dir):
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        return meta['vocab_size'], meta['stoi'], meta['itos']
    else:
        raise FileNotFoundError(f"Meta file not found at {meta_path}")

def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = checkpoint['model_args']
    config = GPTConfig(**model_args)
    model = GPT(config)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def tokenize(string, stoi):
    unk_index = stoi.get('<unk>', 0)
    return np.array([stoi.get(c, unk_index) for c in string], dtype=np.int64)

def detokenize(tokens, itos):
    return ''.join([itos.get(t, '<unk>') for t in tokens])

# --- Beam search based generation ---
def generate_next_move_beam_search(model, prompt_pgn, stoi, itos, device, 
                                   beam_width=3, max_length=1023, temperature=1.0, verbose=False):
    """
    Uses beam search to generate a chess move (in PGN) from a given prompt.
    
    Args:
        model (GPT): The GPT model.
        prompt_pgn (str): The current PGN string (prompt).
        stoi (dict): Mapping from characters to token IDs.
        itos (dict): Mapping from token IDs to characters.
        device (str): Device to run the model on.
        beam_width (int): Number of beams to keep.
        max_length (int): Maximum number of tokens to generate.
        temperature (float): Temperature for scaling logits.
        verbose (bool): If True, print debugging information.
    
    Returns:
        str: The move generated (with trailing spaces stripped), or None if nothing was generated.
    """
    # Each beam is a tuple: (current_prompt, generated_move, cumulative_log_prob)
    beams = [(prompt_pgn, "", 0.0)]
    completed_beams = []

    for step in range(max_length):
        new_beams = []
        for current_prompt, generated, cum_log_prob in beams:
            # Check termination: if the last generated character is a space, we consider it finished.
            if generated and generated[-1] == " ":
                completed_beams.append((current_prompt, generated, cum_log_prob))
                continue

            # Tokenize the current prompt and run the model
            input_tokens = tokenize(current_prompt, stoi)
            input_tensor = torch.from_numpy(input_tokens).unsqueeze(0).to(device)  # shape: (1, seq_len)
            if input_tensor.size(1) > max_length:
                if verbose:
                    print(f"Input length {input_tensor.size(1)} exceeds max_length {max_length}.")
                continue

            with torch.no_grad():
                try:
                    logits, _ = model(input_tensor)
                except Exception as e:
                    if verbose:
                        print(f"Error during model inference: {e}")
                    continue

            # Focus on the logits for the last token and adjust by temperature
            last_logits = logits[:, -1, :] / temperature  # shape: (1, vocab_size)
            probabilities = F.softmax(last_logits, dim=-1).squeeze(0)  # shape: (vocab_size)
            
            # Get top beam_width tokens
            topk = torch.topk(probabilities, beam_width)
            top_probs = topk.values.cpu().numpy()
            top_indices = topk.indices.cpu().numpy()
            
            for prob, idx in zip(top_probs, top_indices):
                char = detokenize([idx], itos)
                new_generated = generated + char
                # Add log probability for numerical stability (log product becomes sum)
                new_cum_log_prob = cum_log_prob + math.log(prob + 1e-10)
                new_prompt = current_prompt + char
                new_beams.append((new_prompt, new_generated, new_cum_log_prob))
        
        if not new_beams:
            break

        # Keep only the beam_width most promising beams
        new_beams.sort(key=lambda x: x[2], reverse=True)
        beams = new_beams[:beam_width]

        # If all beams have finished (i.e. ended with a space), we can stop early.
        if all(beam[1].endswith(" ") for beam in beams):
            completed_beams.extend(beams)
            break

    # If we have any completed beams, choose the one with the highest log probability.
    if completed_beams:
        completed_beams.sort(key=lambda x: x[2], reverse=True)
        best_move = completed_beams[0][1].strip()
    else:
        beams.sort(key=lambda x: x[2], reverse=True)
        best_move = beams[0][1].strip() if beams else ""
    
    if verbose:
        print(f"Beam search generated move: '{best_move}' with log-probability: {completed_beams[0][2] if completed_beams else beams[0][2]}")
    
    return best_move if best_move else None

# --- Updated chess move generation using beam search ---
def chess_gpt_generated_move_beam_search(model, board, prompt_pgn, stoi, itos, device, 
                                         max_retries, beam_width, idx, verbose, troubleshooting_verbose):
    """
    Generates a move using beam search and validates it against the chess board.
    If the generated move is invalid (e.g. not legal in the current board position),
    it retries up to max_retries times.
    
    Args:
        model (GPT): The GPT model.
        board (chess.Board): The current chess board.
        prompt_pgn (str): The current PGN string.
        stoi (dict): Mapping from characters to token IDs.
        itos (dict): Mapping from token IDs to characters.
        device (str): Device to run the model on.
        max_retries (int): Number of times to try generating a valid move.
        beam_width (int): Number of beams to use in beam search.
        idx (int): Current move number (for logging purposes).
        verbose (bool): If True, print additional output.
        troubleshooting_verbose (bool): If True, print troubleshooting info.
    
    Returns:
        str: The generated move if valid, or a flag string if it fails (e.g. "oversize").
    """
    retries = 0
    invalid_generations = []
    while retries < max_retries:
        generated_move = generate_next_move_beam_search(
            model, prompt_pgn, stoi, itos, device, beam_width=beam_width, verbose=verbose
        )
        if generated_move is None:
            if verbose:
                print(f"Move generation failed (possibly due to context size) on move {idx}.")
            return "oversize"
        else:
            try:
                # Try to apply the move; if it fails, it will throw an exception.
                board.push_san(generated_move)
                return generated_move
            except ValueError:
                invalid_generations.append(generated_move)
                retries += 1
                if troubleshooting_verbose:
                    print(f"Invalid move generated on move {idx}: '{generated_move}'. Retrying {retries}/{max_retries}...")
    if verbose:
        print(f"After {max_retries} retries, invalid moves were: {invalid_generations}")
    return None

# --- (The rest of your evaluation code remains largely unchanged) ---
def play_game_against_stockfish(model, engine, stoi, itos, device, stockfish_path, time_per_move, 
                                max_retries, color, verbose, troubleshooting_verbose, beam_width):
    if verbose:
        print("------------------------- \n Starting game against Stockfish \n -------------------------")

    board = chess.Board()
    prompt_pgn = ';'
    move_number = 0
    color_slider = 0 if (color == "white") else 1
    if verbose:
        print(f"Our model is playing {color}")

    while not board.is_game_over():
        move_number += 1
        # Update PGN prompt – add move numbers appropriately.
        if move_number % 2 == 1:
            prompt_pgn += f"{move_number//2 + 1}."

        if (move_number + color_slider) % 2 == 1:
            # Model’s turn (using beam search generation)
            gpt_move = chess_gpt_generated_move_beam_search(
                model, board, prompt_pgn, stoi, itos, device, max_retries, beam_width, move_number, verbose, troubleshooting_verbose
            )
            if gpt_move is None:
                if verbose:
                    print(f"Game lost due to inability to generate a valid move (max retries: {max_retries}).")
                return "loss_invalid_gen", move_number
            elif gpt_move == "oversize":
                if verbose:
                    print("Game lost because input exceeded model context size.")
                return "loss_context_size", move_number
            else:
                pgn_move = gpt_move
        else:
            # Stockfish’s turn
            result = engine.play(board, chess.engine.Limit(time=time_per_move))
            pgn_stockfish_move = board.san(result.move)
            board.push(result.move)
            pgn_move = pgn_stockfish_move

        prompt_pgn += pgn_move + ' '

    # Determine win/loss based on who made the last move.
    if (move_number % 2 == 0 and color == "black") or (move_number % 2 == 1 and color == "white"):
        return 'win', move_number
    else:
        return 'loss', move_number

# --- (The rest of your evaluation functions remain largely unchanged) ---
# For example: update_elo, print_evaluation_report, quick_save_checkpoint, run_evaluation, etc.
# When calling play_game_against_stockfish, be sure to pass the beam_width parameter.
def evaluate_models(model_names, args):
    for model_name in model_names:
        args.checkpoint = model_name
        args.save_file = os.path.join(args.save_dir, f"{model_name}_results.pkl")
        entry_key = run_evaluation(args)
        print_evaluation_report(args,entry_key)

def print_evaluation_report(args,entry_key):
    elo_results = args.elo_results[entry_key]
    model_name = elo_results["model_name"]
    stockfish_name = elo_results["stockfish_name"]
    counters = elo_results["counters"]
    ng = counters["games_evaluated"]
    print("--------------------EVALUATION REPORT--------------------")
    print(f"Model {model_name} vs {stockfish_name}")
    print(f"Total games: {ng}")
    print(f"Wins: {counters["win"]} ({100 * (counters["win"]/ng):.2f}%)")
    print(f"Loss: {counters["loss"]} ({100 * (counters["loss"]/ng):.2f}%)")
    print(f"Context size: {counters["context_size"]} ({100 * (counters["context_size"]/ng):.2f}%)")
    print(f"Invalid generation: {counters["invalid_gen"]} ({100 * (counters["invalid_gen"]/ng):.2f}%)")


# --- Main script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM against Stockfish to compute Elo.")
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint.')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory with meta.pkl.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on.')
    parser.add_argument('--num_games', type=int, default=100, help='Number of games to play.')
    parser.add_argument('--time_per_move', type=float, default=0.1, help='Time per move for Stockfish (seconds).')
    parser.add_argument('--max_retries', type=int, default=3, help='Max retries for invalid LLM moves.')
    parser.add_argument('--evaluation_games', type=int, default=3, help='Number of evaluation games.')
    parser.add_argument('--desired_elo', type=int, default=3, help='Desired Elo rating for Stockfish.')
    parser.add_argument('--save_interval', type=int, default=10, help='Save progress every N games.')
    parser.add_argument('--save_file', type=str, default='results.pkl', help='File to save progress.')
    parser.add_argument('--models_dir', type=str, help='Directory with model checkpoints.')
    parser.add_argument('--save_dir', type=str, help='Directory to save results.')
    parser.add_argument('--stockfish_path', type=str, required=True, help='Path to Stockfish executable.')
    parser.add_argument('--verbose', action="store_true", help='Enable verbose output.')
    parser.add_argument('--troubleshooting_verbose', action="store_true", help='Enable troubleshooting verbose output.')
    # New argument for beam width:
    parser.add_argument('--beam_width', type=int, default=3, help='Number of beams to use in beam search.')
    args = parser.parse_args()

    # Load or create main results dictionary
    elo_results = load_json_file(ELO_RESULTS_FILE)
    args.elo_results = elo_results

    # (Rest of your evaluation logic goes here. When calling play_game_against_stockfish,
    #  pass args.beam_width as the beam_width parameter.)
    
    if args.models_dir:
        model_names = [f[:-4] for f in os.listdir(args.models_dir) if f.endswith('.pth')]
        evaluate_models(model_names, args)
    else:
        entry_key = run_evaluation(args)
        print_evaluation_report(args,entry_key)
