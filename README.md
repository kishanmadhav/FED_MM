# Federated Learning for Pneumonia Detection

This project implements a  Multi-model federated learning system employing novel scheduling and aggregation methods for pneumonia detection using chest X-ray images.This a proof of concept to show that this system can be used with multiple models for hospitals who want to find the best fit for their data without necessarily wanting to share it. In our examples the three models used are the ResNET-50,Densenet 121 and VGG net for a broad range of model parameters.
## Main Features

- **Federated Learning Server/Client:** Simulate a federated learning setup with multiple clients and a central server.
- **Model Checkpointing:** Save model weights after each round and extra round.
- **Ensemble Distillation:** Combine multiple models into a single student model using knowledge distillation.
- **Evaluation:** Compare the performance of different models on a test set.

## Main Scripts

- `manual_scaffold_server.py`: Starts the federated learning server.
- `manual_scaffold_client.py`: Starts a federated client (run separately for each client).
- `ensemble_distillation.py`: Performs ensemble knowledge distillation using saved models.
- `test.py`: Evaluates and compares the ensemble and individual models.
- `dicom_to_jpg.py`: Converts DICOM images to JPG format for preprocessing.

## Data

- Place your training labels CSV (e.g., `stage_2_train_labels.csv`) and image folders (e.g., `train_preprocess/`) in the project directory as required by the scripts.

## How to Run

1. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

2. **Start the server:**
   ```
   python manual_scaffold_server.py
   ```

3. **Start each client in a separate terminal:**
   ```
   python manual_scaffold_client.py 1
   python manual_scaffold_client.py 2
   python manual_scaffold_client.py 3
   ```

4. **After training, run ensemble distillation:**
   ```
   python ensemble_distillation.py
   ```

5. **Evaluate models:**
   ```
   python test.py
   ```

## Notes

- Make sure your data paths in the scripts match your local setup.
- The project uses PyTorch and Flower for federated learning.
- CUDA is recommended for faster training, but CPU is also supported.


