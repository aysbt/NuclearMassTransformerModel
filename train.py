import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
from dataloader import get_dataloader
from dataloader import get_eval_dataloader

from model import TransformerMassExcessPredictor
from evaluate import evaluate_model
from evaluate import plot_predictions
from evaluate import plot_error_distribution
from evaluate import visualize_attention_weights
from evaluate import plot_training_logs
from evaluate import plot_error_N_vs_Z
from evaluate import compare_actual_mass_ve_predicted_mass
from trainer import train_model
from evaluate import compute_metrics
from evaluate import plot_error_diagnostics
# Define feature sets for different models
feature_sets = {
  #"CoreModel": ["N", "Z", "A"],
   #"ShellModel": ["N", "Z", "A", "Zshell_category", "Nshell_category"],
   #"ZEOModel": ["N", "Z", "A", "ZEO", "NEO"],
    #"MagicModel": ["N", "Z", "A", "deltaN", "deltaZ"],
    #"LiquidDropModel": ["N", "Z", "A", "A^2/3", "Z(Z-1)/A^1/3", "(N-Z)^2/A"],
    "FullFeatureModel": ["N", "Z", "A", "Zshell_category", "Nshell_category", "ZEO", "NEO", "deltaN", "deltaZ", "A^2/3", "Z(Z-1)/A^1/3", "(N-Z)^2/A"]
}

def main(csv_path, batch_size, num_epochs, lr):

    """
    Main function to train and evaluate the Transformer model for mass excess prediction.

    Args:
        csv_path (str): Path to dataset CSV file.
        batch_size (int): Batch size for training and evaluation.
        num_epochs (int): Number of training epochs.
        lr (float): Learning rate.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Store results for comparison
    model_results = {}

    # Create a directory for saving plots if it doesn't exist
    os.makedirs("plots", exist_ok=True)

    for model_name, selected_features in feature_sets.items():
        print(f"\n🚀 Training Model: {model_name} with features: {selected_features}")

        # Load dataset for current feature set
        train_loader = get_dataloader(csv_path, selected_features, batch_size=batch_size, shuffle=True, split="train")
        val_loader = get_dataloader(csv_path, selected_features, batch_size=batch_size, shuffle=False, split="val")
        # Evaluation DataLoader (includes N and Z values for plotting)
        eval_loader = get_eval_dataloader(csv_path, selected_features, batch_size=batch_size, shuffle=False)

        for categorical_data, continuous_data, target_mass_excess in train_loader:
            #print(f"continuous_data: {continuous_data}")
            #print(f"categorical_data: {categorical_data}")
            seq_length = (categorical_data.shape[1] if categorical_data.dim() > 1 else 0) + 1
            print(f"🔍 Sequence Length: {seq_length}")
            break  # Only print for the first batch

        # Determine feature dimensions
        num_cont_features = len(train_loader.dataset.continuous_features)
        categorical_feature_sizes = [len(train_loader.dataset.data[col].unique()) for col in train_loader.dataset.categorical_features]
      
        print(f"Number of continues Features: {num_cont_features}")
        print(f"Number of categorical Features: {categorical_feature_sizes}")
        # Initialize model
        model = TransformerMassExcessPredictor(
            num_cont_features, 
            categorical_feature_sizes, 
            d_model=128, 
            num_heads=8, 
            d_ff=512, 
            num_layers=4, 
            dropout=0.1).to(device)

        # Define loss function and optimizer

        #criterion = nn.MSELoss()  # Mean Squared Error for regression
        criterion = nn.L1Loss()

        #Huber loss
        #criterion = nn.SmoothL1Loss(beta=100)  # beta controls sensitivity to large errors
        

        #optimizer = optim.Adam(model.parameters(), lr=lr)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        
        #adaptive learning rate scheduler for  reduce the learning rate when validation loss stops improving  
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        # OneCycleLR scheduler
        #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=num_epochs, pct_start=0.1)


        # Train the model
        checkpoint_path = f"model_checkpoint/best_{model_name}.pth"
        os.makedirs("model_checkpoint", exist_ok=True)

        # Train the model and collect logs
        model, logs = train_model(      
            train_loader,
            val_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            num_epochs=num_epochs,
            checkpoint_path=checkpoint_path
        )
        # Load best model for evaluation
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        # Evaluate model
        N_values, Z_values, predictions, actuals, attention_weights = evaluate_model(model, eval_loader, device)
       

        


        # Calculate final validation loss
        #final_loss = ((torch.tensor(predictions) - torch.tensor(actuals)) ** 2).mean().item()
        final_loss = criterion(torch.tensor(predictions), torch.tensor(actuals)).item()
        #model_results[model_name + "_val"] = final_loss

    

        # Compute metrics
        metrics = compute_metrics(predictions, actuals)
        model_results[model_name + "_val"] = metrics["rmse"]

        # Print metrics
        print(f"\n📊 Evaluation Metrics for {model_name}:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
    
        
        # Save training curves (log loss & accuracy)
        if logs:
            plot_training_logs(logs, model_name)

        else:
            print(f"⚠️ No logs returned for {model_name}")

        # Save and plot results
        plot_predictions(
                N_values=N_values,  # Extract N from dataset
                Z_values=Z_values,  # Extract Z from dataset
                predictions=predictions,
                actuals=actuals,
                model_name=model_name
        )
        plot_error_diagnostics(N_values, Z_values, predictions, actuals, model_name=model_name)


        plot_error_distribution(predictions, actuals, model_name)

        #plot_error_N_vs_Z( N_values=N_values,  Z_values=Z_values, predictions=predictions, actuals=actuals,model_name=model_name+'_mass_Excess') 
        
        #call the funtion to get resl mass comparasion
        compare_actual_mass_ve_predicted_mass(
                           N_values=N_values,  
                           Z_values=Z_values, 
                           predictions=predictions,
                           actuals=actuals,
                           model_name=model_name)

        # Extract feature names
        feature_names = train_loader.dataset.continuous_features + train_loader.dataset.categorical_features
        categorical_features = ["Zshell_category", "Nshell_category", "ZEO", "NEO", "deltaN","deltaZ"]
        # Save attention weights plot
        visualize_attention_weights(attention_weights, categorical_features, model_name)

        print(f"✅ {model_name} Training Completed! Final Validation Loss: {final_loss:.4f}")
    
    # Print model comparison results
    print("\n📊 Model Performance Comparison:")
    for model_name, loss in sorted(model_results.items(), key=lambda x: x[1]):
        print(f"{model_name}: Final Validation Loss = {loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Transformer model for mass excess prediction.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the dataset CSV file")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    

    args = parser.parse_args()
    main(args.csv_path, args.batch_size, args.num_epochs, args.lr)
