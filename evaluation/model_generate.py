import torch
import torch.nn.functional as F
import numpy as np
import pickle
import os
import re
import chess
from model import GPT, GPTConfig  # Ensure model.py is in the same directory or adjust the import path accordingly

# Optional: If your checkpoint is stored on S3 or a remote location,
# implement the logic to download it. For simplicity, this script assumes it's local.
# from remote.save_checkpoints import load_checkpoint

def count_and_get_identical_indexes(str1, str2):
    """
    Counts the number of identical characters at the same indexes in two strings and returns the matching indices.

    Args:
        str1 (str): The first string.
        str2 (str): The second string.

    Returns:
        tuple: (int, list) where:
            - int: The number of indexes where the characters are identical.
            - list: A list of indices where the characters are identical.
    """
    # Ensure the strings are of the same length to avoid out-of-range errors
    length = min(len(str1), len(str2))
    # Count matching characters at each index and collect matching indices
    matching_indices = [i for i in range(length) if str1[i] == str2[i]]
    identical_count = len(matching_indices)

    return identical_count, matching_indices


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
        print('unwanted prefixes were found in the checkpoint')
    return new_state_dict


def load_meta(data_dir):
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
    # Convert string to list of token IDs
    ids = np.array([stoi[c] for c in string], dtype=type)
    return ids

def detokenize(tokens, itos):
    # Convert list of token IDs back to string
    cs = ''.join([itos[t] for t in tokens])

    return cs

def load_model(checkpoint_path, device):
    # Load checkpoint
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

def predict_next_characters(model, input_string, stoi, itos, device, max_length=1024):
    """
    Predicts the next character for each position in the input string.

    Args:
        model: The GPT model.
        input_string: The input string.
        stoi: String to index mapping.
        itos: Index to string mapping.
        device: 'cuda' or 'cpu'.
        max_length: Maximum sequence length.

    Returns:
        List of predicted characters.
    """
    # Tokenize input
    input_tokens = tokenize(input_string, stoi)
    input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(device)  # Shape: (1, seq_len)

    if input_tensor.size(1) > max_length:
        raise ValueError(f"Input sequence length {input_tensor.size(1)} exceeds maximum block size {max_length}.")

    with torch.no_grad():
        # Forward pass
        print(f"input tensor is {input_tensor} of length {len(input_tensor)}")
        logits, _ = model(input_tensor)  # logits shape: (1, seq_len, vocab_size)

        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)  # Shape: (1, seq_len, vocab_size)

        # Get the predicted token for each position
        predicted_tokens = torch.argmax(probs, dim=-1)  # Shape: (1, seq_len)

    predicted_tokens = predicted_tokens.squeeze(0).tolist()  # Shape: (seq_len,)

    # Detokenize to get characters
    predicted_chars = detokenize(predicted_tokens, itos)

    return predicted_chars




def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description="GPT Model Inference for Next Character Prediction")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint (e.g., checkpoint.pth)')
    parser.add_argument('--input', type=str, required=True, help='Input string for which to predict next characters')
    parser.add_argument('--data_dir', type=str, default='data/openwebtext', help='Directory where meta.pkl is located')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run the model on')
    args = parser.parse_args()

    # Load meta information for tokenizer
    vocab_size, stoi, itos = load_meta(args.data_dir)
    print(f"Vocab Size: {vocab_size}")

    # Load the model
    model = load_model(args.checkpoint, args.device)
    print("Model loaded successfully.")

    # Predict next characters
    input_string = args.input

    predicted_chars = predict_next_characters(model, input_string, stoi, itos, args.device)

    print("Input String:")
    print(input_string) 
    print("Predicted Next Characters:")
    print(predicted_chars)
    identical_count, matching_indices = count_and_get_identical_indexes(input_string, (';' + predicted_chars))
    num_tokens = len(predicted_chars)
    print(f'Num identical tokens: {identical_count}, num total tokens: {num_tokens}, identical rate: {identical_count/num_tokens}')
    print(f'Identical token rate: {identical_count/num_tokens} ({identical_count}/{num_tokens})')



if __name__ == "__main__":
    main()
