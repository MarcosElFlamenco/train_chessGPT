import pandas as pd
import re
import matplotlib.pyplot as plt
import bisect
import matplotlib.cm as cm

def visualize_error_frequency(csv_file, datasets_to_include, model_prefixes):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Extract iterations from the 'model' column
    df['Iterations'] = df['model'].apply(lambda x: int(re.search(r'_(\d+)K', x).group(1)) * 1000 if re.search(r'_(\d+)K', x) else None)

    # Filter data by datasets and model prefixes
    df = df[df['dataset'].isin(datasets_to_include)]
    filtered_dfs = []
    for prefix in model_prefixes:
        filtered_dfs.append(df[df['model'].str.startswith(prefix)])
    df_filtered = pd.concat(filtered_dfs)

    # Group data by model_prefix, dataset, and iterations
    df_filtered['Model_Prefix'] = df_filtered['model'].apply(lambda x: next((prefix for prefix in model_prefixes if x.startswith(prefix)), None))
    grouped = df_filtered.groupby(['Model_Prefix', 'dataset', 'Iterations'], as_index=False)['error_frequency'].mean()

    # Plotting
    plt.figure(figsize=(10, 6))
    for prefix in model_prefixes:
        for dataset in datasets_to_include:
            subset = grouped[(grouped['Model_Prefix'] == prefix) & (grouped['dataset'] == dataset)]
            if not subset.empty:
                plt.plot(subset['Iterations'], subset['error_frequency'], marker='o', label=f'{prefix} on {dataset}')

    # Customize the plot
    plt.xlabel('Iterations')
    plt.ylabel('Error Frequency')
    plt.title('Error Frequency vs. Iterations')
    plt.legend(title='Model and Dataset')
    plt.grid(True)
    plt.show()
    plt.savefig("evaluation/graph.jpg", dpi=300, bbox_inches='tight')

# Input parameters
csv_file = 'evaluation/benchmark_results.csv'  # Replace with your CSV file path
datasets_to_include = ['random100games', 'lichess13_100g_180m']  # Replace with your datasets
model_prefixes = ['lichess9gb', 'random16M']  # Replace with your model prefixes



def plot_error_frequencies(data, model_types, benchmark_datasets, max_moves_list):
    """
    Enhanced version of the error frequency plot:
    - Same model-dataset combinations share a base color.
    - Intensity of the color varies with 'max_moves' using a colormap gradient.
    """
    plt.figure(figsize=(12, 7))
    base_cmap = cm.get_cmap('viridis')  # Colormap for gradients

    # Assign unique colors for each (model, dataset) combination
    color_mapping = {}
    color_idx = 0
    unique_pairs = [(model, dataset) for model in model_types for dataset in benchmark_datasets]

    for model, dataset in unique_pairs:
        if model not in data:
            print(f"Warning: Model '{model}' not found in data.")
            continue

        # Assign a base color for each model-dataset pair
        base_color = base_cmap(color_idx / len(unique_pairs))
        color_mapping[(model, dataset)] = base_color
        color_idx += 1

        for idx, max_moves in enumerate(sorted(max_moves_list)):
            iteration_vals = []
            error_freqs = []

            # Iterate over checkpoints (iterations)
            for iteration, iteration_dict in data[model].items():
                if dataset not in iteration_dict:
                    continue

                errors_info = iteration_dict[dataset]
                errors_lists = [a["error_indices"] for a in errors_info]
                moves_lists = [a['num_moves'] for a in errors_info]
                total_moves_tested_for = 0
                num_mistakes = 0

                # Count errors under 'max_moves'
                for i in range(len(moves_lists)):
                    total_moves_in_game = moves_lists[i]
                    total_moves_tested_for_in_game = min(max_moves, total_moves_in_game)
                    total_moves_tested_for += total_moves_tested_for_in_game

                    errors_list = errors_lists[i]
                    num_mistakes_game = bisect.bisect_right(errors_list, max_moves)
                    num_mistakes += num_mistakes_game

                error_freq = num_mistakes / total_moves_tested_for
                iteration_vals.append(int(iteration[0:-1]))
                error_freqs.append(error_freq)

            # Plot with gradient intensity for 'max_moves'
            if iteration_vals:
                iteration_vals, error_freqs = zip(*sorted(zip(iteration_vals, error_freqs)))
                gradient_alpha = 0.3 + 0.7 * (idx / len(max_moves_list))  # Vary transparency with max_moves
                marker = 'o' if dataset == 'lichess13_100g_180m' else 'x' 
                plt.plot(iteration_vals, error_freqs, marker=marker,
                         label=f"{model} - {dataset} - {max_moves}",
                         color=base_color, alpha=gradient_alpha, linewidth=2)

    # Graph customization
    plt.title("Error Frequency vs. Iterations (Gradient Intensity for Max Moves)")
    plt.xlabel("Iterations (K)")
    plt.ylabel("Error Frequency (mistakes / total moves)")
    plt.grid(True)

    # Simplify the legend to show only model-dataset pairs
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = {}
    for h, l in zip(handles, labels):
        key = " - ".join(l.split(" - ")[:-1])  # Combine model and dataset as key
        if key not in unique_labels:
            unique_labels[key] = h

    plt.legend(unique_labels.values(), unique_labels.keys(), title="Model - Dataset", loc='upper right')
    plt.tight_layout()
    plt.show()


def plot_error_cloud(data, model_names, datasets, max_moves_considered, max_games_considered):
    """
    Plot a cloud of error indices for specified models, iterations, and datasets.

    Args:
    - data (dict): The dictionary containing error information.
    - model_names (list): List of model names in the form `model_baseName_iteration`.
    - datasets (list): List of datasets to include in the graph.

    This function creates a scatter plot for each model-dataset pair.
    """
    plt.figure(figsize=(12, 7))

    for model_full_name in model_names:
        # Parse model base name and iteration from the full name
        if '_' not in model_full_name:
            print(f"Invalid model name: {model_full_name}")
            continue

        model_parts = model_full_name.split('_')
        base_name = '_'.join(model_parts[:-1])  # Everything before the last part
        iteration_str = model_parts[-1]  # The last part (e.g., '30K')
        iteration = iteration_str # Convert '30K' to integer (e.g., 30000)

        if base_name not in data:
            print(f"Warning: Model '{base_name}' not found in data.")
            continue

        if iteration not in data[base_name]:
            print(f"Warning: Iteration {iteration} not found for model '{base_name}'.")
            continue

        for dataset in datasets:
            if dataset not in data[base_name][iteration]:
                print(f"Warning: Dataset '{dataset}' not found for model '{base_name}' at iteration {iteration}.")
                continue

            # Retrieve error indices for the model-dataset combination
            max_games_considered = min(max_games_considered,100)
            errors_info = data[base_name][iteration][dataset]
            games_considered = [game for game in errors_info][:max_games_considered]
            error_indices_as_sets = [game['error_indices'] for game in games_considered]
            error_indices_under_max_moves = [e[:bisect.bisect_right(e,max_moves_considered)] for e in error_indices_as_sets]
            error_indices = [indices for indices_set in error_indices_under_max_moves for indices in indices_set]
#            error_indices = [error for e in errors_info for error in e["error_indices"]]

            # Plot a line of points for the error indices
            plt.scatter(error_indices, [f"{base_name}--{iteration}--{dataset}"] * len(error_indices), alpha=0.6, label=f"{base_name}_{dataset}", marker='|')

    plt.title("Error Indices for Models and Datasets")
    plt.xlabel("Error Index")
    plt.ylabel("Model-Dataset")
    plt.grid(True)
    plt.legend(loc='best', fontsize=8, title="Model-Dataset")
    plt.tight_layout()
    plt.show()
