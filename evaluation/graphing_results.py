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
    - Separate legends for models and datasets.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_yscale("log")

    # Define color palettes
    colors = ["#6ebf06", "#e89915", "#e31609", "#2292a4", "#714955"]
    line_styles = {
        'random100games': '-',
        'lichess13_100g_180m': '--',
        'random2128games': '-',
        'kasparov2128games': '--'
    }

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
                    iteration_val = 30
                    try:
                        iteration_val = int(iteration[:-1])
                    except Exception as e:
                        print(f'we got an exception {e}')
                    iteration_vals.append(iteration_val)
                    error_freqs.append(error_freq)

                if iteration_vals:
                    iteration_vals, error_freqs = zip(*sorted(zip(iteration_vals, error_freqs)))
                    color = color_mapping[model]
                    ax.plot(
                        iteration_vals,
                        error_freqs,
                        line_styles[dataset],
                        color=color,
                        linewidth=2,
                        label=f"{model} - {dataset}"  # Temporary legend label (not final)
                    )
                    ax.scatter(
                        iteration_vals,
                        error_freqs,
                        color=color,
                        edgecolor='black',
                        zorder=5
                    )

    # Create legends
    legend_names = {
        "lichess_karvhyp": "lichess",
        "random_karvhypNSNR": "small_random",
        "big_random16M_vocab32": "big_random"
    }

    # Model legend (color-based)
    model_legend_elements = [
        plt.Line2D([0], [0], color=color_mapping[model], lw=2, label=legend_names.get(model, model))
        for model in model_types
    ]

    # Dataset legend (line-style-based)
    dataset_legend_elements = [
        plt.Line2D([0], [0], color='black', linestyle=line_styles[dataset], lw=2, label=dataset)
        for dataset in benchmark_datasets
    ]

    model_legend = ax.legend(handles=model_legend_elements, title="Models", loc='upper right')
    ax.add_artist(model_legend)  # Retain the first legend

    dataset_legend = ax.legend(handles=dataset_legend_elements, title="Datasets", loc='upper left')

    # Customize the graph
    ax.set_title("Error Frequency vs. Iterations")
    ax.set_xlabel("Iterations (K)")
    ax.set_ylabel("Error Frequency (log-scale)")
    ax.grid(True)
    plt.tight_layout()

    plt.savefig("generation_results.png")
    plt.show()

