# NuclearMassTransformerModel

A Transformer-based deep learning model for predicting nuclear mass excess values using both categorical and continuous nuclear features. This project leverages modern attention-based architectures to model nuclear structure behavior across the chart of nuclides.

## 📁 Repository Structure

```plaintext
NuclearMassTransformerModel/

├── data/
│   ├── trainval.csv              # Processed training and validation data
│   └── test.data                 # Held-out test data (AME2020 updates)
├── dataloader.py          # Data loading and batching functions
├── evaluate.py            # Evaluation script for the trained model
├── model.py               # Transformer model architecture definition
├── test.py                # Script for inference or testing on held-out data
├── train.py               # Entry point for training the model
├── trainer.py             # Training loop logic including optimizer and scheduler setup
├── utils.py               # Utility functions (e.g., metrics, plotting, logging)
└── README.md              # Project documentation (this file)
```

## 🚀 How to Run

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/aysbt/NuclearMassTransformerModel.git
cd NuclearMassTransformerModel
pip install -r requirements.txt
```

### 🧪 Train the Model
```bash
python train.py --csv_path data/trainval.csv
```

The training script uses default model architecture settings, which are currently defined directly in train.py. If you want to modify core hyperparameters such as:

* d_model: embedding dimension (default: 128)

* d_ff: feedforward layer size (default: 512)

* num_layers: number of Transformer encoder blocks (default: 4)

* dropout: dropout rate for regularization (default: 0.1)

  ### 🧪 Test the Model
  ```bash
    python test.py 
    ```
 **Note:** The test dataset path is already defined inside the `test.py` script. You do not need to pass it explicitly unless you modify the script.

 **Note 2:** Before running `test.py`, open the `dataloader.py` file and comment out or disable the line:
```python
 stratify = self.data["bin"]
 ```
 This line is used for stratified splitting during training and is not applicable when working with the test set.
 




