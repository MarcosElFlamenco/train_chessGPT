def analyze_games(file_path):
    """
    Analyzes a binary file to count games and compute statistics on game lengths.

    Args:
        file_path (str): Path to the binary file.

    Returns:
        dict: A dictionary with game count, min, max, and average game lengths.
    """
    game_count = 0
    total_length = 0
    min_length = float('inf')
    max_length = 0

    # Define the delimiter for splitting (15 as a 32-bit integer, little-endian format)
    delimiter = b'\x0f\x06\x04\x1d\t\x00\x1e'

    with open(file_path, 'rb') as file:
        # Read the entire file as binary
        first_line = file.readline()
        print(first_line)
        data = file.read()

        # Split the data into chunks based on the delimiter
        games = data.split(delimiter)

        # Calculate the length of each game
        game_lengths = list(map(len, games))

        # Update statistics
        game_count = len(games)
        total_length = sum(game_lengths)
        min_length = min(game_lengths) if game_lengths else float('inf')
        max_length = max(game_lengths) if game_lengths else 0

    # Calculate average length
    average_length = total_length / game_count if game_count > 0 else 0

    return {
        "game_count": game_count,
        "min_length": min_length,
        "max_length": max_length,
        "average_length": average_length,
    }

# Example Usage
file_path = "data/lichess_hf_dataset/train9gb.bin"
stats = analyze_games(file_path)

print(f"Game Count: {stats['game_count']}")
print(f"Min Game Length: {stats['min_length']} bytes")
print(f"Max Game Length: {stats['max_length']} bytes")
print(f"Average Game Length: {stats['average_length']:.2f} bytes")
