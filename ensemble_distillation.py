import torch
import torch.nn as nn
from model_and_data import PneumoniaCNN, RSNAPneumoniaDataset
from torch.utils.data import DataLoader
import os
import socket
import threading

# Set the student (global) architecture here
STUDENT_ARCH = 'densenet121'  # Change to 'resnet18' or 'mobilenet_v2' if desired
# Use extra round checkpoints for ensemble
EXTRA_ROUND_NUM = 0  # Set which extra round to use

def load_models(model_dir, extra_round_num, device):
    models = {}
    for model_type in ['densenet121', 'resnet18', 'mobilenet_v2']:
        model = PneumoniaCNN(adaptive_config={'model_type': model_type})
        model_path = os.path.join(model_dir, "{}_extra_{}.pth".format(model_type, extra_round_num))
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        models[model_type] = model
        print("[LOAD] Loaded {} from {}".format(model_type, model_path))
    print("[INFO] All three models loaded for ensemble distillation.")
    return models

def distill_ensemble(models, student, dataloader, device, epochs=5, temperature=3.0, lr=1e-3):
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    criterion = nn.KLDivLoss(reduction='batchmean')
    softmax = nn.Softmax(dim=1)
    log_softmax = nn.LogSoftmax(dim=1)
    student.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, _ in dataloader:
            images = images.to(device)
            with torch.no_grad():
                logits_list = [model(images).detach() / temperature for model in models.values()]
                avg_logits = torch.mean(torch.stack(logits_list), dim=0)
                teacher_probs = softmax(avg_logits)
            student_logits = student(images) / temperature
            loss = criterion(log_softmax(student_logits), teacher_probs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print("Epoch {}, Loss: {:.4f}".format(epoch+1, total_loss/len(dataloader)))
    return student

def main():
    initialize_global_models()
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((HOST, PORT))
    server_sock.listen(MAX_CLIENTS)
    logger.info("Server listening on {}:{}".format(HOST, PORT))
    client_id_counter = 1
    try:
        while True:
            conn, addr = server_sock.accept()
            threading.Thread(target=handle_client, args=(conn, addr, client_id_counter), daemon=True).start()
            client_id_counter += 1
            # Minimal fix: if all clients have connected and disconnected, break
            if client_id_counter > MAX_CLIENTS and len(active_clients) == 0:
                logger.info("All clients have finished. Shutting down server.")
                break
    except Exception as e:
        logger.info("Server shutting down due to exception: {}".format(e))
    finally:
        server_sock.close()
        save_final_models(global_models)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use the same dataset as training for demonstration (replace with public/held-out if available)
    public_dataset = RSNAPneumoniaDataset("stage_2_train_labels.csv", "train_preprocess")
    dataloader = DataLoader(public_dataset, batch_size=16, shuffle=True)
    # Load models from extra round 0
    models = load_models("saved_models_rounds", EXTRA_ROUND_NUM, device)
    # Initialize student (can be any architecture)
    student = PneumoniaCNN(adaptive_config={'model_type': STUDENT_ARCH}).to(device)
    print("[INFO] Student (global) model architecture: {}".format(STUDENT_ARCH))
    # Distill
    student = distill_ensemble(models, student, dataloader, device)
    # Save student
    student_path = "saved_models_rounds/ensemble_student_{}.pth".format(STUDENT_ARCH)
    torch.save(student.state_dict(), student_path)
    print("[SAVE] Ensemble student model saved to {}".format(student_path))
    main() 