import argparse
import os
import pickle
import chess
import chess.engine
import torch
import torch.nn.functional as F
import numpy as np
from model import GPT, GPTConfig  # Ensure model.py is available


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
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    return model


def predict_next_characters(model, input_string, stoi, itos, device, max_length=1024):
    input_tokens = np.array([stoi[c] for c in input_string], dtype=np.int64)
    input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(device)

    if input_tensor.size(1) > max_length:
        raise ValueError(f"Input sequence length {input_tensor.size(1)} exceeds max block size {max_length}.")

    with torch.no_grad():
        logits, _ = model(input_tensor)
        probs = F.softmax(logits, dim=-1)
        predicted_tokens = torch.argmax(probs, dim=-1).squeeze(0).tolist()
        predicted_chars = ''.join([itos[t] for t in predicted_tokens])

    return predicted_chars


def play_game_against_stockfish(model, stoi, itos, device, stockfish_path, time_per_move, max_retries):
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    board = chess.Board()

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            retries = 0
            move = None
            while retries < max_retries:
                input_string = board.fen() + '\n'
                generated_pgn = predict_next_characters(model, input_string, stoi, itos, device)
                try:
                    move = chess.Move.from_uci(generated_pgn[:4])
                    if move in board.legal_moves:
                        break
                except:
                    pass
                retries += 1

            if retries == max_retries:
                engine.quit()
                return "0-1" if board.turn == chess.WHITE else "1-0"  # Loss by illegal moves

            board.push(move)
        else:
            result = engine.play(board, chess.engine.Limit(time=time_per_move))
            board.push(result.move)

    engine.quit()
    return board.result()


def update_elo(elo_a, elo_b, result, k=32):
    prob_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
    score_a = 1 if result == "1-0" else 0 if result == "0-1" else 0.5
    return elo_a + k * (score_a - prob_a)


def run_evaluation(args):
    vocab_size, stoi, itos = load_meta(args.data_dir)
    model = load_model(args.checkpoint, args.device)

    stockfish_path = args.stockfish_path
    elo = 1200  # Starting Elo
    stockfish_elo = 1200
    results = []

    for game_idx in range(args.num_games):
        result = play_game_against_stockfish(
            model, stoi, itos, args.device, stockfish_path, args.time_per_move, args.max_retries
        )
        elo = update_elo(elo, stockfish_elo, result)
        results.append((game_idx + 1, result, elo))

        if (game_idx + 1) % args.save_interval == 0:
            with open(args.save_file, 'wb') as f:
                pickle.dump(results, f)

    # Final save
    with open(args.save_file, 'wb') as f:
        pickle.dump(results, f)

    return elo


def evaluate_models(model_names, args):
    for model_name in model_names:
        args.checkpoint = os.path.join(args.models_dir, f"{model_name}.pth")
        args.save_file = os.path.join(args.save_dir, f"{model_name}_results.pkl")
        final_elo = run_evaluation(args)
        print(f"Model: {model_name}, Final Elo: {final_elo}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM against Stockfish to compute Elo.")
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint.')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with meta.pkl.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on.')
    parser.add_argument('--num_games', type=int, default=100, help='Number of games to play.')
    parser.add_argument('--time_per_move', type=float, default=0.1, help='Time per move for Stockfish (seconds).')
    parser.add_argument('--max_retries', type=int, default=3, help='Max retries for invalid LLM moves.')
    parser.add_argument('--save_interval', type=int, default=10, help='Save progress every N games.')
    parser.add_argument('--save_file', type=str, default='results.pkl', help='File to save progress.')
    parser.add_argument('--models_dir', type=str, help='Directory with model checkpoints.')
    parser.add_argument('--save_dir', type=str, help='Directory to save results.')
    parser.add_argument('--stockfish_path', type=str, required=True, help='Path to Stockfish executable.')
    args = parser.parse_args()

    if args.models_dir:
        model_names = [f[:-4] for f in os.listdir(args.models_dir) if f.endswith('.pth')]
        evaluate_models(model_names, args)
    else:
        final_elo = run_evaluation(args)
        print(f"Final Elo: {final_elo}")
