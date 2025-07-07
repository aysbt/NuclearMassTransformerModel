import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
from dataloader import get_eval_dataloader
from model import TransformerMassExcessPredictor
from evaluate import plot_error_diagnostics
from evaluate import evaluate_model
from evaluate import compute_metrics
import numpy as np

# ---------------------------
# Configuration
# ---------------------------
d_ff = 512
d_model = 128
num_heads = 8
num_layers = 4
output_dim = 1
batch_size = 16
dropout = 0.1
test_csv_path = "data/test.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Feature Set and Model Paths
# ---------------------------
feature_sets = {
    "CoreModel": ["N", "Z", "A"],
   # "ShellModel": ["N", "Z", "A", "Zshell_category", "Nshell_category"],
   # "ZEOModel": ["N", "Z", "A", "ZEO", "NEO"],
   # "MagicModel": ["N", "Z", "A", "deltaN", "deltaZ"],
   # "LiquidDropModel": ["N", "Z", "A", "A^2/3", "Z(Z-1)/A^1/3", "(N-Z)^2/A"],
   # "FullFeatureModel": ["N", "Z", "A", "Zshell_category", "Nshell_category", "ZEO", "NEO", "deltaN", "deltaZ", "A^2/3", "Z(Z-1)/A^1/3", "(N-Z)^2/A"]
}
model_dir = "model_checkpoint"

# ---------------------------
# Evaluation and Plotting
# ---------------------------
results = {}

for model_name, selected_features in feature_sets.items():
    print(f"\n🔍 Evaluating {model_name}...")

    # Prepare DataLoader for test set
    #test_loader = get_eval_dataloader(test_csv_path, selected_features, batch_size=batch_size, shuffle=False, split="test")
    
    # Load test DataLoader using existing utility
    test_loader = get_eval_dataloader(
        csv_path="data/test.csv",
        selected_features=selected_features,
        batch_size=batch_size,
        shuffle=False
    )
    categorical_sizes_per_model = {
    "CoreModel": [],
    #"ShellModel": [20, 20],          # Zshell_category, Nshell_category
    #"ZEOModel": [2, 2],              # ZEO, NEO
    #"MagicModel": [52, 23],            # deltaN, deltaZ
    #"LiquidDropModel": [],
    #"FullFeatureModel": [20, 20, 2, 2, 52, 23]  # Zshell, Nshell, ZEO, NEO, deltaN, deltaZ
    }
    #categorical_features = ["Zshell_category", "Nshell_category", "ZEO", "NEO", "deltaN","deltaZ"]
    #test_loader = get_test_dataloader(selected_features, batch_size, shuffle=False, split="test")
    # Feature info
    num_cont_features = len(test_loader.dataset.continuous_features)
    #categorical_feature_sizes = [len(test_loader.dataset.data[col].unique()) for col in test_loader.dataset.categorical_features]
    #categorical_feature_sizes = [20, 20, 2, 2, 7, 8]
    categorical_feature_sizes = categorical_sizes_per_model[model_name]


    print(f"CategoricalFeture size: {categorical_feature_sizes}")
    # Load and setup model

    model = TransformerMassExcessPredictor(
        num_cont_features,
        categorical_feature_sizes,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    model_path = os.path.join(model_dir, f"best_{model_name}.pth")
    print(f"Model Path: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()

    # Evaluate
    N_vals, Z_vals, preds, targets, _ = evaluate_model(model, test_loader, device)
    
    # ✅ Apply inverse log transform to get back to keV scale
    restored_preds = [np.sign(p) * np.expm1(abs(p)) for p in preds]
    restored_targets = [np.sign(t) * np.expm1(abs(t)) for t in targets]
    errors = [p - t for p, t in zip(restored_preds, restored_targets)]

    results[model_name] = {
        "N": N_vals,
        "Z": Z_vals,  # Optional but useful for computing A
        "predicted": restored_preds,
        "true": restored_targets,
        "error": errors
    }

    

# ---------------------------
# Plot Results
# ---------------------------

import numpy as np

fig, ax1 = plt.subplots(figsize=(12, 10))  # Only one axis, no need for sharex

# Define unique colors and markers (optional)
colors = {
    "CoreModel": "tab:blue",
    "ShellModel": "tab:orange",
    "ZEOModel": "tab:green",
    "MagicModel": "tab:red",
    "LiquidDropModel": "tab:purple",
    "FullFeatureModel": "tab:brown"
}

markers = {
    "CoreModel": "o",
    "ShellModel": "s",
    "ZEOModel": "D",
    "MagicModel": "^",
    "LiquidDropModel": "v",
    "FullFeatureModel": "P"
}

# Plot for each model
for name, data in results.items():
    metrics = compute_metrics(data["predicted"], data["true"])
    #model_results[model_name + "_val"] = metrics["rmse"]

    # Print metrics
    print(f"\n📊 Evaluation Metrics for {name}:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


    N_vals = np.array(data["N"])
    preds = np.array(data["predicted"])
    trues = np.array(data["true"])
    errors = np.array(data["error"])    

   
    mask = N_vals > 150  # Only keep N > 150

    

    if np.any(mask):
        filtered_N = N_vals[mask]
        filtered_pred = np.array(data["predicted"])[mask]
        filtered_error = np.array(data["error"])[mask]
        filtered_true = np.array(data["true"])[mask]

        filtered_N = N_vals[mask]
        filtered_pred = preds[mask]
        filtered_true = trues[mask]
        filtered_error = errors[mask]

        # Recalculate RMSE using filtered data
        rmse = np.sqrt(np.mean((filtered_pred - filtered_true) ** 2))
        label_with_rmse = f"{name} (RMSE: {rmse:.2f} keV)"

        # Plotting
        ax1.scatter(filtered_N, filtered_pred, s=14, alpha=0.7, label=label_with_rmse,
                color=colors.get(name, None), marker=markers.get(name, "o"))
        #ax1.scatter(filtered_N, filtered_true, s=14, alpha=0.3, label=label_with_rmse,
                #color=colors.get(name, None), marker=markers.get(name, "o"))

    
    plot_error_diagnostics(data["N"], data["Z"], data["predicted"], data["true"], model_name+"test")
#plot_error_distribution(data["predicted"], data["true"],model_name+"_test")##

# Format top plot
ax1.set_ylabel("Predicted Mass Excess (keV)", fontsize=16)
ax1.set_title("Mass Excess Predictions of Different Models (Test Set)", fontsize=15)
ax1.legend(fontsize=10, loc="upper left")
ax1.grid(True)




# Add fixed RMSE & MAE annotation manually (for publication)

plt.tight_layout()
plt.savefig("plots/model_comparison_prediction_test.png", dpi=300)

#plot_error_diagnostics(data["N"], data["Z"], data["predicted"], data["true"], model_name+"_test")
#plot_error_distribution(data["predicted"], data["true"],model_name+"_test")

