import socket
import pickle
import torch
import random
import struct
import time
from torch.utils.data import DataLoader, Subset
from model_and_data import PneumoniaCNN, RSNAPneumoniaDataset
import sys
import logging
from collections import defaultdict
from typing import List
from sklearn.metrics import f1_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

CSV_PATH = r"C:\CDAC\stage_2_train_labels.csv"
IMG_DIR = r"C:\CDAC\train_preprocess"

# Network configuration
HOST = '127.0.0.1'
PORT = 8081
MAX_RETRIES = 10
RETRY_DELAY = 2  # seconds
CHUNK_SIZE = 8192  # 8KB chunks for data transfer

def get_client_dataset(client_id):
    dataset = RSNAPneumoniaDataset(CSV_PATH, IMG_DIR)
    total = len(dataset)
    
    # Each client gets a portion of the data
    if client_id == "1":
        indices = range(0, int(0.4 * total))  # 40% of data
    elif client_id == "2":
        indices = range(int(0.4 * total), int(0.75 * total))  # 35% of data
    else:
        indices = range(int(0.75 * total), total)  # 25% of data
    
    subset = Subset(dataset, indices)
    logger.info("Client {} initialized with {:.1f}% of total data ({} samples)".format(
        client_id, 100 * len(subset) / total, len(subset)))
    
    return subset

def recvall(sock, n):
    """Receive exactly n bytes from socket with timeout"""
    data = bytearray()
    start_time = time.time()
    timeout = 30.0  # 30 second timeout
    
    while len(data) < n:
        try:
            remaining = n - len(data)
            chunk = sock.recv(min(CHUNK_SIZE, remaining))
            if not chunk:
                if time.time() - start_time > timeout:
                    logger.error("[Client] Receive timeout after {} seconds".format(timeout))
                    return None
                time.sleep(0.1)
                continue
            data.extend(chunk)
        except socket.timeout:
            if time.time() - start_time > timeout:
                logger.error("[Client] Receive timeout after {} seconds".format(timeout))
                return None
            continue
        except Exception as e:
            logger.error("[Client] Error during recv: {}".format(str(e)))
            return None
    return data

def send_data(sock, data, max_retries=3):
    """Send data to server with chunking and retry logic"""
    for attempt in range(max_retries):
        try:
            serialized_data = pickle.dumps(data)
            length = struct.pack('!Q', len(serialized_data))
            
            # Send length first
            sock.sendall(length)
            
            # Send data in chunks with small delays
            for i in range(0, len(serialized_data), CHUNK_SIZE):
                chunk = serialized_data[i:i + CHUNK_SIZE]
                sock.sendall(chunk)
                time.sleep(0.01)  # Small delay between chunks
            
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning("[Client] Send attempt {} failed: {}. Retrying...".format(attempt + 1, str(e)))
                time.sleep(1)  # Wait before retry
            else:
                logger.error("[Client] Error sending data after {} attempts: {}".format(max_retries, str(e)))
                return False
    return False

def receive_data(sock, max_retries=3):
    """Receive data from server with chunking and retry logic"""
    for attempt in range(max_retries):
        try:
            # Receive length first
            length_data = recvall(sock, 8)
            if not length_data:
                if attempt < max_retries - 1:
                    logger.warning("[Client] Receive attempt {} failed: No length data. Retrying...".format(attempt + 1))
                    time.sleep(1)
                    continue
                return None
            
            total_length = struct.unpack('!Q', length_data)[0]
            data = recvall(sock, total_length)
            
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning("[Client] Receive attempt {} failed: {}. Retrying...".format(attempt + 1, str(e)))
                time.sleep(1)
            else:
                logger.error("[Client] Error receiving data after {} attempts: {}".format(max_retries, str(e)))
                return None
    return None

def evaluate_model(model, dataset, device):
    """Evaluate model on local validation set"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataset:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return f1_score(all_labels, all_preds, average='binary')

def train_model(model, dataset, device):
    """Train model for one round and return performance metrics"""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Training loop
    total_loss = 0.0
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    batch_num = 0
    for images, labels in loader:
        batch_num += 1
        logger.info("[Client] Training batch {}".format(batch_num))
        try:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
        except Exception as e:
            logger.error("[Client] Error in training batch: {}".format(str(e)))
            continue
    
    if batch_num == 0:
        logger.error("[Client] No batches processed! Check your dataset and DataLoader.")
        
    # Calculate metrics
    avg_loss = total_loss / batch_num if batch_num > 0 else float('nan')
    
    # Calculate F1 score
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            # Move tensors to CPU before converting to numpy
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    f1 = f1_score(all_labels, all_preds, average='binary')
    
    return {
        'loss': avg_loss,
        'f1_score': f1
    }

def select_best_model(global_models, dataset, device):
    """Select best performing model from global models using combined score"""
    best_f1 = -1
    best_model_type = None
    best_state_dict = None
    best_combined_score = -1
    
    # Create validation set
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Calculate data weight (normalized by total dataset size)
    data_weight = len(dataset) / len(dataset)  # This will be 1.0, actual weight is handled by server
    
    # Alpha and beta weights for combined score
    alpha = 0.5  # Performance weight
    beta = 0.5   # Data weight
    
    for model_type, model_data in global_models.items():
        # Initialize model
        model = PneumoniaCNN(adaptive_config={'model_type': model_type})
        if 'state' in model_data and model_data['state'] is not None:
            try:
                model.load_state_dict(model_data['state'])
                logger.info("[Client] Successfully loaded state_dict for model {}".format(model_type))
            except Exception as e:
                logger.error("[Client] Error loading state_dict for model {}: {}".format(model_type, str(e)))
                continue
        model.to(device)
        
        # Evaluate model
        f1 = evaluate_model(model, val_loader, device)
        
        # Calculate combined score using the formula: S_k,m(t) = α * P'_k,m(t) + β * W_Dk
        combined_score = (alpha * f1) + (beta * data_weight)
        
        logger.info("[Client] Model {} - Combined Score: {:.4f} (F1: {:.4f}, Data Weight: {:.4f})".format(
            model_type, combined_score, f1, data_weight))
        
        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_f1 = f1
            best_model_type = model_type
            best_state_dict = model_data.get('state')
    
    logger.info("[Client] Best model selected: {} with combined score: {:.4f}".format(
        best_model_type, best_combined_score))
    
    return best_model_type, best_state_dict, best_f1, best_combined_score

if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("Error: Client ID not provided")
        logger.error("Usage: python manual_scaffold_client.py CLIENT_ID")
        logger.error("where CLIENT_ID is 1, 2, or 3")
        sys.exit(1)
    
    client_id = sys.argv[1]
    if client_id not in ["1", "2", "3"]:
        logger.error("Error: CLIENT_ID must be 1, 2, or 3")
        logger.error("Usage: python manual_scaffold_client.py CLIENT_ID")
        sys.exit(1)
    
    logger.info("Starting client {}".format(client_id))
    dataset = get_client_dataset(client_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info("[Client {}] Dataset length: {}".format(client_id, len(dataset)))
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(120.0)  # Increased from 30.0 to 120.0 seconds
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)  # Enable keep-alive
        logger.info("[Client {}] Attempting to connect to server...".format(client_id))
        
        # Add connection retry logic
        max_connect_retries = 3
        for attempt in range(max_connect_retries):
            try:
                sock.connect((HOST, PORT))
                logger.info("[Client {}] Connected to server.".format(client_id))
                break
            except Exception as e:
                if attempt < max_connect_retries - 1:
                    logger.warning("[Client {}] Connection attempt {} failed: {}. Retrying...".format(
                        client_id, attempt + 1, str(e)))
                    time.sleep(2)
                else:
                    logger.error("[Client {}] Failed to connect after {} attempts".format(
                        client_id, max_connect_retries))
                    sys.exit(1)
        
        # Send client_id and dataset size as the first message
        if not send_data(sock, {
            'client_id': client_id,
            'dataset_size': len(dataset)
        }):
            logger.error("[Client {}] Failed to send client_id to server. Exiting.".format(client_id))
            sock.close()
            sys.exit(1)
            
        rnd = 0
        best_f1 = -1
        best_model_type = None
        best_state_dict = None
        best_combined_score = -1
        model_scores = {}
        
        while rnd < 6:
            logger.info("[Client {}] Waiting for model assignment for round {}".format(client_id, rnd+1))
            assignment_data = receive_data(sock)
            if assignment_data is None:
                logger.info("[Client {}] No assignment received. Server may have closed connection. Exiting.".format(client_id))
                break
                
            # Check if this is actually an extra rounds signal
            if isinstance(assignment_data, dict) and assignment_data.get('extra_rounds_start'):
                logger.info("[Client {}] Received extra rounds signal during regular rounds. Proceeding to extra rounds.".format(client_id))
                break
                
            model_type = assignment_data['model_type']
            state_dict = assignment_data.get('state_dict')
            
            if state_dict is None:
                logger.error("[Client {}] No state_dict received for model {}".format(client_id, model_type))
                break
                
            logger.info("[Client {}] Assigned {} for round {}".format(client_id, model_type, rnd+1))
            
            adaptive_config = {
                'model_type': model_type,
                'device_type': str(device)
            }
            model = PneumoniaCNN(adaptive_config=adaptive_config)
            try:
                model.load_state_dict(state_dict)
                logger.info("[Client {}] Successfully loaded state_dict for model {}".format(client_id, model_type))
            except Exception as e:
                logger.error("[Client {}] Error loading state_dict: {}".format(client_id, str(e)))
                break
                
            model.to(device)
            logger.info("[Client {}] Model loaded and moved to {}".format(client_id, device))
            
            if len(dataset) == 0:
                logger.error("[Client {}] Dataset is empty. Skipping training.".format(client_id))
                metrics = {'loss': float('nan'), 'f1_score': 0.0}
            else:
                metrics = train_model(model, dataset, device)
                
            logger.info("[Client {}] Completed training for round {} (loss: {:.4f}, F1: {:.4f})".format(
                client_id, rnd+1, metrics['loss'], metrics['f1_score']))
                
            model.cpu()
            logger.info("[Client {}] Model moved to CPU for sending update.".format(client_id))
            
            # Add retry logic for sending updates
            update_success = False
            for attempt in range(3):
                update = {
                    'state_dict': model.state_dict(),
                    'metrics': metrics
                }
                if send_data(sock, update):
                    update_success = True
                    logger.info("[Client {}] Sent model update to server".format(client_id))
                    break
                else:
                    logger.warning("[Client {}] Failed to send update (attempt {}). Retrying...".format(
                        client_id, attempt + 1))
                    time.sleep(1)
            
            if not update_success:
                logger.error("[Client {}] Failed to send update after all retries".format(client_id))
                break
            
            # Track best model using combined score
            f1 = metrics['f1_score']
            data_weight = len(dataset) / len(dataset)  # Normalized data weight
            combined_score = (0.5 * f1) + (0.5 * data_weight)
            model_scores[model_type] = {'f1': f1, 'combined_score': combined_score}
            
            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_f1 = f1
                best_model_type = model_type
                best_state_dict = model.state_dict()
                
            rnd += 1
            
        # After 6 rounds, continue with best model for 5 more rounds
        if best_model_type is not None and best_state_dict is not None:
            logger.info("[Client {}] Best model after 6 rounds: {} (Combined Score: {:.4f}, F1: {:.4f})".format(
                client_id, best_model_type, best_combined_score, best_f1))
            
            # After completing round 6, wait for server signal to start extra rounds
            logger.info("[Client {}] Waiting for server signal to start extra rounds...".format(client_id))
            
            # Send completion notification and wait for extra rounds signal
            if not send_data(sock, {'round_complete': True}):
                logger.error("[Client {}] Failed to notify server of round 6 completion".format(client_id))
                sock.close()
                sys.exit(1)
            
            # Wait for server signal to start extra rounds
            extra_rounds_started = False
            max_wait_time = 120  # 2 minutes max wait (increased from 5 minutes)
            start_time = time.time()
            heartbeat_interval = 10  # Send heartbeat every 10 seconds
            
            while not extra_rounds_started and (time.time() - start_time) < max_wait_time:
                try:
                    # Send periodic heartbeat to keep connection alive
                    if int(time.time() - start_time) % heartbeat_interval == 0:
                        send_data(sock, {'heartbeat': True})
                        logger.info("[Client {}] Sent heartbeat, waiting for extra rounds signal... ({}s elapsed)".format(
                            client_id, int(time.time() - start_time)))
                    
                    # Check for server response with shorter timeout
                    sock.settimeout(10.0)  # Increased from 5.0 to 10.0 seconds
                    response = receive_data(sock)
                    sock.settimeout(120.0)  # Reset to normal timeout (increased from 30.0)
                    
                    if response is None:
                        time.sleep(1)
                        continue
                    
                    if isinstance(response, dict):
                        if response.get('extra_rounds_start'):
                            extra_rounds_started = True
                            logger.info("[Client {}] Received signal to start extra rounds!".format(client_id))
                            break
                        elif response.get('heartbeat'):
                            continue  # Just a heartbeat, continue waiting
                        else:
                            logger.info("[Client {}] Received unexpected response: {}".format(client_id, response))
                    
                except socket.timeout:
                    continue  # Timeout is expected, continue waiting
                except Exception as e:
                    logger.error("[Client {}] Error while waiting for extra rounds signal: {}".format(client_id, str(e)))
                    break
            
            if not extra_rounds_started:
                logger.error("[Client {}] Timeout waiting for extra rounds signal after {} seconds. Exiting.".format(
                    client_id, max_wait_time))
                sock.close()
                sys.exit(1)
            
            # Now proceed with 5 extra rounds using the best model
            logger.info("[Client {}] Starting 5 extra rounds with best model: {}".format(client_id, best_model_type))
            
            extra_rounds_timeout = 300  # 5 minutes timeout for all extra rounds
            extra_rounds_start_time = time.time()
            
            for extra_round in range(5):
                # Check if we've exceeded the timeout
                if time.time() - extra_rounds_start_time > extra_rounds_timeout:
                    logger.error("[Client {}] Extra rounds timeout after {} seconds. Exiting.".format(
                        client_id, extra_rounds_timeout))
                    break
                    
                logger.info("[Client {}] Starting extra round {}/5".format(client_id, extra_round + 1))
                
                # Request the best model from server with retry logic
                request_success = False
                for attempt in range(3):
                    if send_data(sock, {'request_model_type': best_model_type}):
                        request_success = True
                        logger.info("[Client {}] Successfully requested model for extra round {}".format(client_id, extra_round + 1))
                        break
                    else:
                        logger.warning("[Client {}] Failed to request model (attempt {}). Retrying...".format(
                            client_id, attempt + 1))
                        time.sleep(1)
                
                if not request_success:
                    logger.error("[Client {}] Failed to request model for extra round {} after all retries".format(client_id, extra_round + 1))
                    break
                
                # Small delay after requesting model
                time.sleep(0.5)
                
                # Wait for a valid model assignment (ignore heartbeats)
                while True:
                    assignment_data = receive_data(sock)
                    if assignment_data is None:
                        logger.warning("[Client {}] No assignment received, retrying...".format(client_id))
                        time.sleep(1)
                        continue
                    if isinstance(assignment_data, dict) and assignment_data.get('heartbeat'):
                        logger.info("[Client {}] Received heartbeat during extra round, waiting for model assignment...".format(client_id))
                        continue
                    if not isinstance(assignment_data, dict) or 'model_type' not in assignment_data:
                        logger.error("[Client {}] Invalid assignment data received for extra round {}: {}".format(
                            client_id, extra_round + 1, assignment_data))
                        logger.warning("[Client {}] Skipping extra round {} due to invalid data".format(client_id, extra_round + 1))
                        continue
                    break  # Got a valid model assignment
                
                model_type = assignment_data['model_type']
                state_dict = assignment_data.get('state_dict')
                
                if state_dict is None:
                    logger.error("[Client {}] No state_dict received for extra round {}".format(client_id, extra_round + 1))
                    break
                
                logger.info("[Client {}] Received {} for extra round {}/5".format(client_id, model_type, extra_round + 1))
                
                # Load and train model
                adaptive_config = {
                    'model_type': model_type,
                    'device_type': str(device)
                }
                model = PneumoniaCNN(adaptive_config=adaptive_config)
                try:
                    model.load_state_dict(state_dict)
                    logger.info("[Client {}] Successfully loaded state_dict for extra round".format(client_id))
                except Exception as e:
                    logger.error("[Client {}] Error loading state_dict for extra round: {}".format(client_id, str(e)))
                    break
                
                model.to(device)
                
                if len(dataset) == 0:
                    logger.error("[Client {}] Dataset is empty. Skipping extra round training.".format(client_id))
                    metrics = {'loss': float('nan'), 'f1_score': 0.0}
                else:
                    try:
                        metrics = train_model(model, dataset, device)
                    except Exception as e:
                        logger.error("[Client {}] Error during extra round training: {}".format(client_id, str(e)))
                        metrics = {'loss': float('nan'), 'f1_score': 0.0}
                
                logger.info("[Client {}] Completed extra round {}/5 (loss: {:.4f}, F1: {:.4f})".format(
                    client_id, extra_round + 1, metrics['loss'], metrics['f1_score']))
                
                model.cpu()
                
                # Send update back to server with retry logic
                update_success = False
                for attempt in range(3):
                    update = {
                        'state_dict': model.state_dict(),
                        'metrics': metrics
                    }
                    
                    if send_data(sock, update):
                        update_success = True
                        logger.info("[Client {}] Sent update for extra round {}/5".format(client_id, extra_round + 1))
                        break
                    else:
                        logger.warning("[Client {}] Failed to send update (attempt {}). Retrying...".format(
                            client_id, attempt + 1))
                        time.sleep(1)
                
                if not update_success:
                    logger.error("[Client {}] Failed to send update for extra round {} after all retries".format(client_id, extra_round + 1))
                    break
                
                # Small delay between extra rounds to prevent connection overload
                time.sleep(1)
            
            logger.info("[Client {}] Completed all extra rounds!".format(client_id))
            
    except Exception as e:
        logger.error("[Client {}] Fatal error: {}".format(client_id, str(e)))
    finally:
        try:
            sock.close()
        except:
            pass
        logger.info("[Client {}] Disconnected.".format(client_id)) 