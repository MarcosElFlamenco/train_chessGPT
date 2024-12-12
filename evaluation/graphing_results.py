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

# Run the visualization
visualize_error_frequency(csv_file, datasets_to_include, model_prefixes)
