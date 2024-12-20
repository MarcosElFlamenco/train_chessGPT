def analyze_games(file_path):
    """
    Analyzes a PGN-like file to count games and compute statistics on game lengths.

    Args:
        file_path (str): Path to the file.

    Returns:
        dict: A dictionary with game count, min, max, and average game lengths.
    """
    game_count = 0
    total_length = 0
    min_length = float('inf')
    max_length = 0

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        for iter, line in enumerate(file, start=1):
            if iter % 10000 == 0:
                print(f"Processed {iter} lines...")

            games = line.split(';')
            line_game_lengths = map(len, games)

            # Update statistics
            game_count += len(games)
            total_length += sum(line_game_lengths)

            # Update min/max in a single pass
            for length in line_game_lengths:
                if length < min_length:
                    min_length = length
                if length > max_length:
                    max_length = length

    # Calculate average length
    average_length = total_length / game_count if game_count > 0 else 0

    return {
        "game_count": game_count,
        "min_length": min_length,
        "max_length": max_length,
        "average_length": average_length,
    }


# Example Usage
file_path = "lichess_hf_dataset/lichess_9gb_blocks.csv"
stats = analyze_games(file_path)

print(f"Game Count: {stats['game_count']}")
print(f"Min Game Length: {stats['min_length']} characters")
print(f"Max Game Length: {stats['max_length']} characters")
print(f"Average Game Length: {stats['average_length']:.2f} characters")
