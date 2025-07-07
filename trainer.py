import torch
import matplotlib.pyplot as plt
import numpy as np
from utils import EarlyStopping

def train_model(train_loader, val_loader, model, criterion, optimizer, scheduler, checkpoint_path, num_epochs):
    """
    Trains a Transformer model for mass excess prediction and logs training progress.

    Args:
        train_loader (DataLoader): Dataloader for training data.
        val_loader (DataLoader): Dataloader for validation data.
        model (torch.nn.Module): Transformer model to train.
        criterion (torch.nn.Module): Loss function (e.g., MSELoss).
        optimizer (torch.optim.Optimizer): Optimizer (e.g., Adam).
        checkpoint_path (str): Path to save the best model.
        num_epochs (int): Number of training epochs.


    Returns:
        torch.nn.Module: Trained model with the best weights.
        dict: Dictionary containing training logs (loss & accuracy).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_loss = float('inf')
 
    # EarlyStopping should be initialized outside the training loop
    early_stopping = EarlyStopping(patience=15, min_delta=0.001)


    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss, correct_train, total_train = 0, 0, 0

        for categorical_data, continuous_data, target_mass_excess in train_loader:
            categorical_data, continuous_data, target_mass_excess = (
                categorical_data.to(device),
                continuous_data.to(device),
                target_mass_excess.to(device),
            )

            optimizer.zero_grad()
            predictions, _ = model(categorical_data, continuous_data)
            loss = criterion(predictions, target_mass_excess)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            total_train_loss += loss.item()


            # Compute accuracy (within 5% error tolerance)
            correct_train += ((torch.abs(predictions - target_mass_excess) / target_mass_excess) < 0.05).sum().item()
            total_train += target_mass_excess.size(0)

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accuracies.append(correct_train / total_train)


        # Validation phase
        model.eval()
        total_val_loss, correct_val, total_val = 0, 0, 0

        with torch.no_grad():
            for categorical_data, continuous_data, target_mass_excess in val_loader:
                categorical_data, continuous_data, target_mass_excess = (
                    categorical_data.to(device),
                    continuous_data.to(device),
                    target_mass_excess.to(device),
                )
                predictions, _ = model(categorical_data, continuous_data)
                loss = criterion(predictions, target_mass_excess)
                total_val_loss += loss.item()

                # Compute accuracy (within 5% error tolerance)
                correct_val += ((torch.abs(predictions - target_mass_excess) / target_mass_excess) < 0.05).sum().item()
                total_val += target_mass_excess.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accuracies.append(correct_val / total_val)

        # In your training loop, use EarlyStopping
        # Step the scheduler **after** validation loss computation
        scheduler.step(avg_val_loss)

        early_stopping(avg_val_loss)

        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break
 

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Train Acc: {train_accuracies[-1]:.4f} | Val Acc: {val_accuracies[-1]:.4f}")

        # Save the best model checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✅ Model checkpoint saved at epoch {epoch+1} (Val Loss: {best_val_loss:.4f})")

    # Store training logs
    logs = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
    }

    return model, logs
