import pandas as pd
import re
import matplotlib.pyplot as plt
import bisect
import matplotlib.cm as cm

def plot_error_frequencies(data, model_types, benchmark_datasets, max_moves_list):
    """
    Enhanced version of the error frequency plot:
    - Bright and distinct colors for each model.
    - Different line styles for different datasets.
    - Simplified legend showing color-to-model and line style-to-dataset correspondence.
    """
    plt.figure(figsize=(12, 7))
    colors = ['#FF5733', '#33C3FF', '#FF33A6', '#75FF33']  # Bright colors for models
    line_styles = {'random100games': '-', 'lichess13_100g_180m': '--'}  # Line styles for datasets

    # Assign unique colors for each model
    color_mapping = {model: colors[i % len(colors)] for i, model in enumerate(model_types)}

    # Plot the lines
    for model in model_types:
        if model not in data:
            print(f"Warning: Model '{model}' not found in data.")
            continue

        for dataset in benchmark_datasets:
            if dataset not in line_styles:
                print(f"Warning: Dataset '{dataset}' not configured with a line style.")
                continue

            for max_moves in sorted(max_moves_list):
                iteration_vals = []
                error_freqs = []

                for iteration, iteration_dict in data[model].items():
                    if dataset not in iteration_dict:
                        continue

                    errors_info = iteration_dict[dataset]
                    errors_lists = [a["error_indices"] for a in errors_info]
                    moves_lists = [a['num_moves'] for a in errors_info]
                    total_moves_tested_for = 0
                    num_mistakes = 0

                    for i in range(len(moves_lists)):
                        total_moves_in_game = moves_lists[i]
                        total_moves_tested_for_in_game = min(max_moves, total_moves_in_game)
                        total_moves_tested_for += total_moves_tested_for_in_game

                        errors_list = errors_lists[i]
                        num_mistakes_game = bisect.bisect_right(errors_list, max_moves)
                        num_mistakes += num_mistakes_game

                    error_freq = num_mistakes / total_moves_tested_for
                    iteration_val =  30
                    try:
                        iteration_val = int(iteration[:-1])
                    except Exception as e:
                        print(f'we got an exception {e}')
                    iteration_vals.append(iteration_val)
                    error_freqs.append(error_freq)

                if iteration_vals:
                    iteration_vals, error_freqs = zip(*sorted(zip(iteration_vals, error_freqs)))
                    color = color_mapping[model]
                    plt.plot(
                        iteration_vals,
                        error_freqs,
                        line_styles[dataset],
                        color=color,
                        linewidth=2
                    )
                    plt.scatter(
                        iteration_vals,
                        error_freqs,
                        color=color,
                        edgecolor='black',  # Optional, adds an outline to the points
                        zorder=5  # Ensures the points are above the line
                    )

    # Add a simplified legend
    legend_elements = []
#    legend_elements = ["lichess_trained","random_trained","random_tested","lichess_tested"]
    
    # Add color-to-model legend
    for model, color in color_mapping.items():
        legend_elements.append(plt.Line2D([0], [0], color=color, lw=2, label=model))
    
    # Add line style-to-dataset legend
    for dataset, style in line_styles.items():
        legend_elements.append(plt.Line2D([0], [0], color='black', linestyle=style, lw=2, label=dataset))

    plt.legend(handles=legend_elements, title="Legend", loc='upper right')

    # Customize the graph
    plt.title("Error Frequency vs. Iterations (Simplified Legend)")
    plt.xlabel("Iterations (K)")
    plt.ylabel("Error Frequency (mistakes / total moves)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
