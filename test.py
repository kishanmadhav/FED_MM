import torch
from torch.utils.data import DataLoader
from model_and_data import PneumoniaCNN, get_test_dataset
import os
from tqdm import tqdm
import sys
import pandas as pd
import torch.nn.functional as F
from datetime import datetime
import numpy as np

CSV_PATH = r"C:\CDAC\stage_2_train_labels.csv"
IMG_DIR = r"C:\CDAC\train_preprocess"
BATCH_SIZE = 16
PREDICTION_THRESHOLD = 0.4  

ENSEMBLE_PATH = "saved_models_rounds/ensemble_student_densenet121.pth"
ROUND5_PATHS = {
    "densenet121": "saved_models_rounds\densenet121_extended_rr.pth",
    "resnet18": "saved_models_rounds/resnet18_extended_rr.pth",
    "mobilenet_v2": "saved_models_rounds/mobilenet_v2_extended_rr.pth"
}

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_predictions(predictions, ground_truth, confidences, dataset, output_dir="predictions"):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, "test_results_{}.csv".format(timestamp))
    
    # Get original patient IDs from the dataset
    patient_ids = []
    for idx in range(len(dataset)):
        # Access the underlying dataset through the Subset
        if hasattr(dataset, 'dataset'):
            original_idx = dataset.indices[idx]
            patient_id = dataset.dataset.pneumonia_frame.iloc[original_idx]['patientId']
        else:
            patient_id = dataset.pneumonia_frame.iloc[idx]['patientId']
        patient_ids.append(patient_id)
    
    # Create DataFrame with results
    results_df = pd.DataFrame({
        'patient_id': patient_ids,
        'predicted_label': predictions,
        'ground_truth': ground_truth,
        'confidence': confidences,
        'correct_prediction': (predictions == ground_truth)
    })
    
    # Save to CSV
    results_df.to_csv(output_file, index=False)
    print("\nPredictions saved to: {}".format(output_file))
    
    # Save summary statistics
    summary_file = os.path.join(output_dir, "summary_{}.txt".format(timestamp))
    with open(summary_file, 'w') as f:
        f.write("Prediction Summary\n")
        f.write("=================\n\n")
        f.write("Total samples: {}\n".format(len(predictions)))
        f.write("Correct predictions: {}\n".format(sum(predictions == ground_truth)))
        f.write("Accuracy: {:.2f}%\n".format(100 * sum(predictions == ground_truth) / len(predictions)))
        f.write("\nConfidence Statistics:\n")
        f.write("Mean confidence: {:.4f}\n".format(confidences.mean()))
        f.write("Median confidence: {:.4f}\n".format(np.median(confidences)))
        f.write("Min confidence: {:.4f}\n".format(confidences.min()))
        f.write("Max confidence: {:.4f}\n".format(confidences.max()))
    
    print("Summary statistics saved to: {}".format(summary_file))

def evaluate_model(model, test_loader, device, loss_fn):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    confusion_matrix = torch.zeros(2, 2)
    all_predictions = []
    all_ground_truth = []
    all_confidences = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.float().to(device)
            labels = labels.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = loss_fn(outputs, labels)
            total_loss += loss.item() * images.size(0)
            probabilities = F.softmax(outputs.float(), dim=1)
            predicted = (probabilities[:, 1] > PREDICTION_THRESHOLD).long()
            confidences = probabilities[:, 1]
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            all_predictions.extend(predicted.cpu().numpy())
            all_ground_truth.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
    accuracy = 100.0 * correct / total
    avg_loss = total_loss / total
    tn, fp, fn, tp = confusion_matrix.flatten().tolist()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': (tn, fp, fn, tp),
        'predictions': all_predictions,
        'ground_truth': all_ground_truth,
        'confidences': all_confidences
    }

def main():
    # Check if dataset files exist
    if not os.path.exists(CSV_PATH):
        print("Error: CSV file '{}' not found.".format(CSV_PATH))
        sys.exit(1)
    if not os.path.exists(IMG_DIR):
        print("Error: Image directory '{}' not found.".format(IMG_DIR))
        sys.exit(1)

    print("\nInitializing test environment...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load dataset
    try:
        print("\nLoading test dataset...")
        test_dataset = get_test_dataset(CSV_PATH, IMG_DIR)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        print("Test dataset size: {} samples".format(len(test_dataset)))
        class_weights = test_dataset.dataset.get_class_weights()
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        print("Using class weights for evaluation:", class_weights.cpu().numpy())
    except Exception as e:
        print("Error loading dataset:", str(e))
        sys.exit(1)

    results = []
    # Evaluate ensemble student
    if os.path.exists(ENSEMBLE_PATH):
        try:
            model = PneumoniaCNN(adaptive_config={'model_type': 'densenet121'})
            model.load_state_dict(torch.load(ENSEMBLE_PATH, map_location=device))
            model.to(device)
            res = evaluate_model(model, test_loader, device, loss_fn)
            res['model'] = "ensemble_student"
            results.append(res)
        except Exception as e:
            print("Error loading ensemble student model: {}".format(e))
    else:
        print("Ensemble student model not found at {}".format(ENSEMBLE_PATH))

    # Evaluate each round 5 model
    for model_type, path in ROUND5_PATHS.items():
        if os.path.exists(path):
            try:
                model = PneumoniaCNN(adaptive_config={'model_type': model_type})
                model.load_state_dict(torch.load(path, map_location=device))
                model.to(device)
                res = evaluate_model(model, test_loader, device, loss_fn)
                res['model'] = model_type
                results.append(res)
            except Exception as e:
                print("Error loading {} model: {}".format(model_type, e))
        else:
            print("Model checkpoint not found for {} at {}".format(model_type, path))

    # Print summary table
    print("\n==================== Model Evaluation Summary ====================")
    print("{:<18} {:>8} {:>10} {:>10} {:>10}".format(
        "Model", "Acc(%)", "Loss", "Prec(%)", "F1(%)"))
    print("-------------------------------------------------------------")
if __name__ == "__main__":
    main() 