import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class NuclearMassDataset(Dataset):
    """
    Custom dataset for loading nuclear mass data with categorical encoding and feature normalization.

    Args:
        csv_path (str): Path to the dataset CSV file.
        selected_features (list): List of selected features for training.
        target_column (str): Name of the target variable.
        split (str): Either 'train' or 'val', to specify whether to load training or validation data.
        val_ratio (float): Percentage of data to use for validation.
        include_nz (bool): Whether to include N (neutrons) and Z (protons) in dataset output.
    """    
    
    def __init__(self, csv_path, selected_features, target_column="Mass Excess (keV)", split="train", seq_length=10, include_nz=False):
        self.data = pd.read_csv(csv_path)
        self.seq_length = seq_length  # Define sequence length
        # ✅ Store original neutron and proton numbers before normalization
        self.data["original_N"] = self.data["N"]
        self.data["original_Z"] = self.data["Z"]
        self.data["actual_mass"] =self.data["Mass Excess (keV)"]

        # Step 4: Create bin feature for stratification (coarse binning on N and Z)
        def bin_feature(n, z):
            return f"{int(n/5)}_{int(z/5)}"     

        self.data["bin"] = self.data.apply(lambda row: bin_feature(row["N"], row["Z"]), axis=1)

        # Step 5: Remove bins with < 2 entries
        bin_counts = self.data["bin"].value_counts()
        valid_bins = bin_counts[bin_counts >= 2].index
        self.data = self.data[self.data["bin"].isin(valid_bins)]
        # Step 5: Stratified train/validation split
        if split == "train" or split == "val":
            train_df, val_df = train_test_split(
                self.data,
                test_size=0.30,
                stratify=self.data["bin"],
                random_state=17
            )
            # Clean up bin column
            train_df = train_df.drop(columns=["bin"])
            val_df = val_df.drop(columns=["bin"])
            
            if split == "train":
                self.data = train_df.reset_index(drop=True)
            else:
                self.data = val_df.reset_index(drop=True)

        else:
            # test gibi durumlarda stratify kullanılmaz
            self.data = self.data.reset_index(drop=True)


        # Define all possible categorical features
        all_categorical_features = ["Zshell_category", "Nshell_category", "ZEO", "NEO", "deltaN","deltaZ"]
        # ✅ Store whether we need N and Z values
        self.include_nz = include_nz  


        # Separate selected features into categorical and continuous
        self.categorical_features = [f for f in selected_features if f in all_categorical_features]
        self.continuous_features = [f for f in selected_features if f not in all_categorical_features]
        
        # Ensure "N" and "Z" are always included
        if "N" not in self.continuous_features:
            self.continuous_features.append("N")
        if "Z" not in self.continuous_features:
            self.continuous_features.append("Z")

        # Convert categorical features to numerical codes
        for col in self.categorical_features:
            self.data[col] = self.data[col].astype('category').cat.codes  # Convert each column
        
        # Convert continuous features to numeric and handle NaNs
        for col in self.continuous_features:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')  # Convert to float

        # Fill missing values with the column mean
        self.data[self.continuous_features] = self.data[self.continuous_features].fillna(self.data[self.continuous_features].mean())

        # Normalize continuous features
        self.data[self.continuous_features] = (self.data[self.continuous_features] - self.data[self.continuous_features].mean()) / self.data[self.continuous_features].std()

        # ✅ Apply Signed Log Transform to Mass Excess (target) for better spread
        self.data[target_column] = np.sign(self.data[target_column]) * np.log1p(np.abs(self.data[target_column])/10000.0)


        # Convert processed data into PyTorch tensors
        self.categorical = torch.tensor(self.data[self.categorical_features].values, dtype=torch.long) if self.categorical_features else None
        self.continuous = torch.tensor(self.data[self.continuous_features].values, dtype=torch.float32)

        # Extract target variable
        self.target = torch.tensor(self.data[target_column].values, dtype=torch.float32)
        
        # ✅ Only initialize N_values and Z_values if needed
        if self.include_nz:
            self.N_values = torch.tensor(self.data["original_N"].values, dtype=torch.float32)
            self.Z_values = torch.tensor(self.data["original_Z"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #start = max(0, idx - self.seq_length + 1)  # Ensure sequence has enough past data
        categorical_data = self.categorical[idx] if self.categorical_features else torch.tensor([])
        #categorical_seq = self.categorical[start:idx+1] if self.categorical_features else torch.tensor([])
        continuous_seq = self.continuous[idx]
        target_value = self.target[idx]  # Single value target


        # ✅ Return N and Z only when `include_nz=True`
        if self.include_nz:
            return categorical_data , self.continuous[idx], self.target[idx], self.N_values[idx], self.Z_values[idx]
        else:
            return categorical_data , self.continuous[idx], self.target[idx]

def get_dataloader(csv_path, selected_features, batch_size, shuffle=True, num_workers=0, split="train",include_nz=False):
    """
    Creates a DataLoader for training and evaluation.

    Args:
        csv_path (str): Path to the dataset CSV file.
        selected_features (list): List of features to be used.
        batch_size (int): Batch size for DataLoader.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int): Number of worker threads.

    Returns:
        DataLoader: PyTorch DataLoader for training or evaluation.
    """
    dataset = NuclearMassDataset(csv_path, selected_features, split=split, include_nz=include_nz)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def get_eval_dataloader(csv_path, selected_features, batch_size, shuffle=False):
    """Returns a DataLoader for evaluation (includes N and Z for plotting)."""
    dataset = NuclearMassDataset(csv_path, selected_features, include_nz=True)  # ✅ Includes N, Z
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)



