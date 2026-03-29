import sys

import numpy as np
import matplotlib.pyplot as plt
import json
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(project_root)
print(project_root)
from controller.dynamic_partition.hnsw.analysis.analysis_hnsw_recall import calculate_x

# Row-level security imports
from controller.baseline.pg_row_security.row_level_security import (
    disable_row_level_security, drop_database_users, create_database_users, enable_row_level_security
)
from basic_benchmark.common_function import ground_truth_func

topk = None
sel = None

def piecewise_recall_model(x, k, beta):
    """
    Piecewise model combining a linear function and a shifted sigmoid function:
    - Linear for x <= k * topk
    - Sigmoid for x > k * topk
    """
    global sel, topk

    # Calculate x_c as proportional to topk
    x_c = k * topk / sel

    # Sigmoid growth rate
    b = beta * 4 * sel / topk

    # Shift for smooth transition
    shift = x_c * sel / topk - 0.5

    # Piecewise function
    return np.piecewise(
        x,
        [x <= x_c, x > x_c],
        [
            lambda x: x * sel / topk,  # Linear part
            lambda x: (1 / (1 + np.exp(-b * (x - x_c)))) + shift  # Sigmoid part
        ]
    )


def validate_fitted_model(query_dataset, ef_search_values, fitted_params):
    """
    Validate the fitted piecewise model by plotting the model predictions against actual data points.

    Parameters:
    - query_dataset: List of query data with user_id, query_vector, etc.
    - ef_search_values: List of ef_search values for validation
    - fitted_params: Parameters of the fitted model (a, b, c, x_c)
    """
    global topk, sel
    topk = query_dataset[0]["topk"]
    sel = np.mean([query["query_block_selectivity"] for query in query_dataset])  # Average selectivity

    actual_recalls = []
    model_recalls = []

    for ef_search in ef_search_values:
        recalls = []
        for query in query_dataset:
            user_id = query["user_id"]
            query_vector = query["query_vector"]
            recall_actual = calculate_actual_recall(user_id, query_vector, topk, ground_truth_func, ef_search)
            recalls.append(recall_actual)
        actual_recalls.append(np.mean(recalls))

        # Calculate model-predicted recall using fitted parameters
        x = calculate_x(ef_search, sel, topk)
    x_fit = np.linspace(min(ef_search_values), max(ef_search_values), 100)
    model_recalls = piecewise_recall_model(x_fit, *fitted_params)

    # Plot comparison
    plt.figure(figsize=(12, 6))
    plt.scatter(ef_search_values, actual_recalls, color='blue', label="Actual Recalls")
    plt.plot(x_fit, model_recalls, color='red', linestyle='--', label="Model Predicted Recalls")
    plt.xlabel("Ef_Search")
    plt.ylabel("Recall")
    plt.title(f"Model Validation: Actual vs Predicted\nsel = {sel:.4f}, topk = {topk}")
    plt.legend()
    plt.grid(True)

    # Save the validation plot
    validation_plot_filename = "recall_model_validation.pdf"
    plt.savefig(validation_plot_filename, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Validation plot saved to: {validation_plot_filename}")


def main():
    """
    Main function to perform model fitting and validation on recall data.
    """
    disable_row_level_security()
    drop_database_users()
    create_database_users()
    enable_row_level_security()

    # Load query dataset
    benchmark_folder = os.path.join(project_root, "basic_benchmark")
    query_dataset_path = os.path.join(benchmark_folder, "query_dataset.json")
    with open(query_dataset_path, "r") as f:
        query_dataset = json.load(f)

    # Define Ef_Search values to test
    ef_search_values = [5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 250, 300, 400, 800]

    # Perform model fitting
    print("Fitting the piecewise recall model...")
    fitted_params = [0.46751805, 0.44240961]

    # Perform model validation
    print("Validating the fitted model...")
    validate_fitted_model(query_dataset, ef_search_values, fitted_params)

    print("Model fitting and validation completed successfully.")


# Run the main function when the script is executed
if __name__ == "__main__":
    main()
