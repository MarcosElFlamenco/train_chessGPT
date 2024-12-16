import pandas as pd
import re
import matplotlib.pyplot as plt

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


def plot_error_frequencies(data, model_types, benchmark_datasets, max_moves):
    """
    data: 
        A nested dictionary with the structure:
        {
            "model_name": {
                <iteration_number>: {
                    "dataset_name": {
                        ... possibly other info ...,
                        "errors": [list_of_mistake_indices]
                    },
                    ...
                },
                ...
            },
            ...
        }

    model_types:
        A list of model names (keys in 'data') to plot.
    
    benchmark_datasets:
        A list of dataset names to evaluate.
    
    max_moves:
        An integer threshold. Only mistakes with index < max_moves are counted.

    This function produces a Matplotlib line plot of iterations vs. error frequency 
    for each (model, dataset) combination.
    """

    plt.figure(figsize=(10, 6))

    for model in model_types:
        # Check if model exists in data
        if model not in data:
            print(f"Warning: Model '{model}' not found in data.")
            continue

        for dataset in benchmark_datasets:
            iteration_vals = []
            error_freqs = []

            # Iterate over each checkpoint (iteration)
            # Example: data[model] = {10000: {...}, 20000: {...}, ...}
            for iteration, iteration_dict in data[model].items():
                # Check if dataset is present at this iteration
                if dataset not in iteration_dict:
                    continue

                # We assume there's a list of errors under the key "errors"
                # or a direct list if your structure is simpler.
                errors_info = iteration_dict[dataset]

                # If the nested structure is: iteration_dict[dataset] = [list_of_errors],
                # then `errors_list = iteration_dict[dataset]`.
                # But if it's iteration_dict[dataset] = {"errors": [list_of_errors]},
                # then:
                errors_lists = [a["error_indices"] for a in errors_info]
                moves_lists = [a['num_moves'] for a in errors_info]
                total_moves = 0
                num_mistakes = 0

                # Filter mistakes that occur before 'max_moves'
                for i in range(len(moves_lists)):
                    total_moves += moves_lists[i]
                    errors_list = errors_lists[i]
                    valid_mistakes = [e for e in errors_list if e < max_moves]
                    num_mistakes += len(valid_mistakes) ##can be made faster

                error_freq = num_mistakes/total_moves
                iteration_vals.append(iteration)
                error_freqs.append(error_freq)

            # Sort (iteration, frequency) pairs by iteration before plotting
            if iteration_vals:
                iteration_vals, error_freqs = zip(*sorted(zip(iteration_vals, error_freqs)))
                plt.plot(iteration_vals, error_freqs, marker='o',
                         label=f"{model} - {dataset}")

    plt.title("Error Frequency vs. Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Error Frequency (mistakes / max_moves)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

