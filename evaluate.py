from dataloader import get_dataloader
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting support



from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_model(model, dataloader, device):
    """
    Evaluates the Transformer model's performance on mass excess prediction.

    Args:
        model (torch.nn.Module): Trained Transformer model.
        dataloader (torch.utils.data.DataLoader): DataLoader for evaluation.
        device (torch.device): Computation device (CPU/GPU).

    Returns:
        tuple: (predictions, actual mass excess values, attention weights).
    """
    model.eval()
    predictions, actuals, attention_weights = [], [], []
    N_values, Z_values = [], []  # Store neutron & proton numbers

    with torch.no_grad():
        for categorical_data, continuous_data, target_mass_excess, N, Z in dataloader:
            categorical_data, continuous_data, target_mass_excess = (
                categorical_data.to(device),
                continuous_data.to(device),
                target_mass_excess.to(device),
            )
            preds, attn_weights = model(categorical_data, continuous_data)

            predictions.extend(preds.cpu().numpy())
            actuals.extend(target_mass_excess.cpu().numpy())


            N_values.extend(N.cpu().numpy()) # Store neutron numbers
            Z_values.extend(Z.cpu().numpy())  # Store proton numbers

            ## Extract attention weights if available
            #if hasattr(model.encoder, "get_attention_weights"):
            #    attn_weights = model.encoder.get_attention_weights()
            #    if attn_weights is not None and len(attn_weights) > 0:
            #        attention_weights.extend([aw.cpu().numpy() for aw in attn_weights])

            if attn_weights is not None and len(attn_weights) > 0:
            # Stack list of tensors (len = num_layers) into shape (batch_size, num_layers, heads, seq_len, seq_len)
                attn_tensor = torch.stack(attn_weights, dim=1)  
                # Append each sample's attention stack to the collection
                for attn_matrix in attn_tensor:  
                    attention_weights.append(attn_matrix.cpu().numpy())
                    

    return N_values, Z_values, predictions, actuals, attention_weights



def compute_metrics(preds, trues):
    preds = torch.tensor(preds)
    trues = torch.tensor(trues)


    rmse = torch.sqrt(torch.mean((preds - trues) ** 2)).item()
    mae = torch.mean(torch.abs(preds - trues)).item()

    mask = torch.abs(trues) > 1e-3  # Avoid dividing by near-zero
    mape = torch.mean(torch.abs((preds[mask] - trues[mask]) / trues[mask])).item()
    within_500kev = torch.mean((torch.abs(preds - trues) < 500).float()).item()

    log_rel_error = torch.log10(torch.abs(preds - trues) / (torch.abs(trues) + 1e-8))
    mean_log_rel = log_rel_error.mean().item()
    std_log_rel = log_rel_error.std().item()

    print(f"\nRMSE (keV): {rmse:.4f}")
    print(f"MAE (keV): {mae:.4f}")
    print(f"MAPE: {mape:.4f}")
    print(f"Accuracy ±500 keV: {within_500kev:.4f}")
    print(f"Mean log-rel error: {mean_log_rel:.4f}")
    print(f"Std log-rel error: {std_log_rel:.4f}")

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "acc_500keV": within_500kev,
        "mean_log_rel": mean_log_rel,
        "std_log_rel": std_log_rel
    }


def plot_predictions(N_values, Z_values, predictions, actuals, model_name):
    """
    Plots and saves a 3D scatter plot of actual vs. predicted mass excess values.

    Args:
        N_values (list): Neutron numbers for each sample.
        Z_values (list): Proton numbers for each sample.
        predictions (list): Model's predicted mass excess values.
        actuals (list): Ground truth mass excess values.
        model_name (str): Name of the model being trained.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Convert to numpy arrays for better handling
    N_values = np.array(N_values)
    Z_values = np.array(Z_values)
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Convert predictions back to the original mass excess scale
    restored_preds = np.sign(predictions) * (np.expm1(np.abs(predictions)) *10000.0)
    restored_actuals = np.sign(actuals) * (np.expm1(np.abs(actuals))*10000.0)

    # Scatter plot for predicted values
    ax.scatter(N_values, Z_values, restored_preds , color="gray", alpha=0.5, label="Predicted Mass Excess")    
    
    # Scatter plot for actual values
    ax.scatter(N_values, Z_values, restored_actuals, color="red", alpha=0.1, label="Actual Mass Excess")

    

    # Labels and title
    ax.set_xlabel("Neutron Number (N)", fontsize=12)
    ax.set_ylabel("Proton Number (Z)", fontsize=12)
    ax.set_zlabel("Mass Excess ($keV$)", labelpad=15, fontsize=12)
    ax.set_title(f"Predicted vs Actual Mass Excess using {model_name}", fontsize=14)

    # Add Z-axis spacing with readable tick locations
    ax.zaxis.set_major_locator(plt.MaxNLocator(nbins=10, prune='both'))
    # Set view angle
    #ax.view_init(elev=30, azim=135)

    # Legend and grid
    ax.legend()
    ax.grid(True)

    # Save plot
    save_path = f"plots/{model_name}_3D_predictions.png"
    os.makedirs("plots", exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Saved: {save_path}")


def plot_error_distribution(predictions, actuals, model_name):
    """
    Plots and saves the error distribution between actual and predicted mass excess values.

    Args:
        predictions (list): Model's predicted mass excess values.
        actuals (list): Ground truth mass excess values.
        model_name (str): Name of the model being trained.
    """
    errors = [p - a for p, a in zip(predictions, actuals)]

    plt.figure(figsize=(8, 6))
    sns.histplot(errors, bins=30, kde=True, color="purple")
    plt.axvline(x=0, color="red", linestyle="--", label="Zero Error")
    plt.xlabel("Prediction Error (Predicted - Actual)")
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    plt.title(f"{model_name} - Error Distribution")
    plt.grid(True)


 

    # Save plot
    save_path = f"plots/{model_name}_error.png"
    os.makedirs("plots", exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Saved: {save_path}")

def plot_error_N_vs_Z(N_values, Z_values, predictions, actuals, model_name):
    # Convert to numpy arrays for better handling
    N_values = np.array(N_values)
    Z_values = np.array(Z_values)
    errors = [p - a for p, a in zip(predictions, actuals)]
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(N_values, Z_values, c=errors, alpha=0.5, cmap="viridis", s=35)
    plt.colorbar(scatter, label="(Predicted - Actual)")
    plt.xlabel("Proton Number ($Z$)", fontsize=12)
    plt.ylabel("Neutron Number ($N$)", fontsize=12)
    plt.title("Absolute Error Between Predicted and Actual Nuclear Mass Across the (N, Z) Plane")
    plt.grid(True)
    plt.tight_layout()
    #plt.show()

     # Save plot
    save_path = f"plots/{model_name}_N_vs_Z_abs_error.png"
    os.makedirs("plots", exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    
U_KEV = 931494.0 # atomic mass unit in keV
def convert_to_mass(mass_excess, N_values, Z_values):
    A = N_values + Z_values
    return mass_excess + A * U_KEV

def compare_actual_mass_ve_predicted_mass(N_values, Z_values, predictions, actuals, model_name):
    """
    Compares predicted nuclear mass (from mass excess) with actual atomic mass.

    Args:
        N_values (list/array): Neutron numbers
        Z_values (list/array): Proton numbers
        predictions (list/array): Predicted mass excess (in keV)
        model_name (str): Model name used for plot output
    """
    # Read actual mass data (in micro-u)
    actual_df = pd.read_csv("data/mass_mu_u_prediction.csv")
    if "Atomic Mass (micro-u)" not in actual_df.columns:
        raise ValueError("CSV file must contain column 'Atomic Mass (micro-u)'")
 
    N_values = np.array(N_values)
    Z_values = np.array(Z_values)
    predictions = np.array(predictions)
    actuals = np.array(actuals)
   
    predicted_mass = convert_to_mass(predictions, N_values, Z_values)
    actual_mass = convert_to_mass(actuals, N_values, Z_values)
    model_name = model_name +'_actual_data'
    #plot_predictions(N_values, Z_values,  predicted_mass , actual_mass, model_name)
    #plot_error_N_vs_Z(N_values, Z_values, predicted_mass , actual_mass, model_name)
    plot_error_diagnostics(N_values, Z_values, predicted_mass, actual_mass, model_name=model_name+'actual_mass')

    print(f"✅ Plotted predicted vs. actual mass using data from 'mass_mu_u_prediction.csv'.")










def visualize_attention_weights(attention_weights, feature_names, model_name):
    """
    Visualizes and saves the averaged attention weights across multiple samples.

    Args:
        attn_weights (list): List of attention weight matrices.
        feature_names (list): Names of input features.
        model_name (str): Name of the model being trained.
    """
    if not attention_weights or attention_weights[0] is None:
        print("No attention weights to visualize.")
        return
 
    first_layer = torch.stack([torch.tensor(a) for a in attention_weights[0]])
    mean_attention = first_layer.mean(dim=(0, 1)).cpu().numpy()  # Average across batch -> (heads, seq_len, seq_len)
    # Convert list of attention matrices to numpy array
    print("mean_attention shape:", mean_attention.shape)
    print("feature_names:", feature_names)

    
    plt.figure(figsize=(10, 8))
    sns.heatmap(mean_attention, xticklabels=feature_names, yticklabels=feature_names,
                    cmap="viridis", annot=True, fmt=".2f")
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(f"{model_name} - Average Attention Weights", fontsize=14, pad=12)
    plt.xlabel("Key (Feature)", fontsize=12)
    plt.ylabel("Query (Feature)", fontsize=12)
    plt.tight_layout()

    # Save plot
    save_path = f"plots/{model_name}_attention.png"
    os.makedirs("plots", exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Saved: {save_path}")
        



def plot_training_logs(logs, model_name):
    """
    Plots training & validation loss and accuracy over epochs.

    Args:
        logs (dict): Dictionary containing training logs (loss & accuracy).
        model_name (str): Name of the model being trained.
    """
    epochs = range(1, len(logs["train_losses"]) + 1)

    # Create a directory to save plots
    os.makedirs("plots", exist_ok=True)

    # Plot Loss Curve
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, logs["train_losses"], label="Train Loss", marker="o")
    plt.plot(epochs, logs["val_losses"], label="Validation Loss", marker="s")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (Huber)")
    plt.legend()
    plt.title(f"{model_name} - Loss Over Epochs")
    plt.grid(True)
    loss_path = f"plots/{model_name}_loss.png"
    plt.savefig(loss_path)
    plt.close()
    print(f"📊 Saved: {loss_path}")

    # Plot Accuracy Curve
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, logs["train_accuracies"], label="Train Accuracy", marker="o")
    plt.plot(epochs, logs["val_accuracies"], label="Validation Accuracy", marker="s")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title(f"{model_name} - Accuracy Over Epochs")
    plt.grid(True)
    acc_path = f"plots/{model_name}_accuracy.png"
    plt.savefig(acc_path)
    plt.close()
    print(f"📊 Saved: {acc_path}")


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

def plot_error_diagnostics(N_values, Z_values, predictions, actuals, model_name="Model"):
    """
    Plots error diagnostics:
    - (N, Z) map with signed error (colored)
    - Z vs Error (left)
    - N vs Error (bottom)
    """
    errors = [p - a for p, a in zip(predictions, actuals)]

    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 4], height_ratios=[4, 1],
                           wspace=0.15, hspace=0.15)

    # (0, 1): Main N vs Z colored by error
    ax_main = fig.add_subplot(gs[0, 1])
    sc = ax_main.scatter(N_values, Z_values, c= errors , cmap="coolwarm", s=20)
    ax_main.set_xlabel("Neutron Number (N)", fontsize=16)
    ax_main.set_ylabel("Proton Number (Z)", fontsize=16)
    #ax_main.set_title(f"{model_name}", fontsize=16)
    cbar = plt.colorbar(sc, ax=ax_main, fraction=0.046, pad=0.04)
    cbar.set_label("Signed Error (keV)", fontsize=16)

        #add only for test
    textstr = "Full Feature Model:\nRMSE = 5.38 ± 0.57 keV\nMAE = 4.54 ± 0.41 keV"
    ax_main.text(0.02, 0.95,  # Position in axis (x, y in relative coords)
                textstr,
                transform=ax_main.transAxes,
                fontsize=16,
                verticalalignment='top',
                bbox=dict(boxstyle='round', alpha=0.2, facecolor='white')
                )
                #add only for test

    # (0, 0): Z vs error (left)
    ax_left = fig.add_subplot(gs[0, 0], sharey=ax_main)
    ax_left.scatter( errors , Z_values, color="tab:blue", alpha=0.4, s=10)
    ax_left.set_xlabel("Error", fontsize=16)
    #ax_left.set_title("Z vs Error")
    ax_left.invert_xaxis()
    ax_left.grid(True)

    # (1, 1): N vs error (bottom)
    ax_bottom = fig.add_subplot(gs[1, 1], sharex=ax_main)
    ax_bottom.scatter(N_values,  errors , color="tab:green", alpha=0.4, s=10)
    ax_bottom.set_ylabel("Error", fontsize=16)
    #ax_bottom.set_title("N vs Error")
    ax_bottom.grid(True)

    # Hide unused (1,0) subplot space
    fig.add_subplot(gs[1, 0]).axis("off")

    #plt.tight_layout()
    path_of_plot = f"plots/{model_name}_signed_error_.png"
    plt.savefig(path_of_plot)
    plt.close()
    print(f"📊 Saved: {path_of_plot}")
