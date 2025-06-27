iimport socket
import pickle
import torch
import struct
import threading
import time
from collections import defaultdict
from model_and_data import PneumoniaCNN
import numpy as np
import logging
from typing import Dict, List, Set
import random
import torch.nn as nn
import torchvision.models as models
import math
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

HOST = '127.0.0.1'
PORT = 8081
MAX_CLIENTS = 3
CHUNK_SIZE = 8192

# Model architecture types
MODEL_TYPES = ['densenet121', 'resnet18', 'mobilenet_v2']

class ModelScheduler:
    def __init__(self, num_clients: int, model_types: List[str]):
        self.num_clients = num_clients
        self.model_types = model_types
        self.client_history = defaultdict(list)
        self.round_assignments = {}
        self.current_round = 0
        self.model_usage = defaultdict(int)
        self.client_model_counts = defaultdict(lambda: defaultdict(int))
        self.performance_scores = defaultdict(lambda: defaultdict(list))  # client -> model -> scores
        self.data_weights = {}  # client -> weight
        self.participation_counts = defaultdict(int)  # client -> count
        self.last_participation = defaultdict(int)  # client -> last round
        self.min_participation_rounds = 3  # Force participation every 3 rounds
        self.alpha = 0.5  # Performance weight (normalized)
        self.beta = 0.5   # Data weight (normalized)
        self.decay_rate = 0.1  # Decay factor for selection scores
        self.random_factor = 0.2  # Randomness factor for exploration
        # Initialize each client with a different starting model index
        self.last_model_assignment = {str(i): i % len(model_types) for i in range(1, num_clients + 1)}
        self.initial_rounds_completed = False  # Track if initial round-robin rounds are completed
        
    def update_performance_score(self, client_id: int, model_type: str, score: float):
        """Update performance score for a client-model pair"""
        self.performance_scores[client_id][model_type].append(score)
        # Keep only last 5 scores
        if len(self.performance_scores[client_id][model_type]) > 5:
            self.performance_scores[client_id][model_type].pop(0)
            
    def update_data_weight(self, client_id: int, data_size: int, total_data_size: int):
        """Update data weight for a client"""
        self.data_weights[client_id] = data_size / total_data_size
        
    def get_normalized_performance_score(self, client_id: int, model_type: str) -> float:
        """Get normalized performance score for a client-model pair"""
        scores = self.performance_scores[client_id][model_type]
        if not scores:
            return 0.5  # Default score if no history
            
        # Calculate average of recent scores
        avg_score = sum(scores) / len(scores)
        
        # Normalize across all client-model pairs
        all_scores = []
        for cid in self.performance_scores:
            for mtype in self.performance_scores[cid]:
                all_scores.extend(self.performance_scores[cid][mtype])
                
        if not all_scores:
            return avg_score
            
        min_score = min(all_scores)
        max_score = max(all_scores)
        if max_score == min_score:
            return 0.5
            
        return (avg_score - min_score) / (max_score - min_score)
        
    def get_combined_score(self, client_id: int, model_type: str) -> float:
        """Calculate combined selection score using the formula:
        S_k,m(t) = α * P'_k,m(t) + β * W_Dk
        where:
        - α is the performance weight (0.5)
        - β is the data weight (0.5)
        - P'_k,m(t) is the normalized performance score
        - W_Dk is the data weight for client k
        """
        # Get normalized performance score
        perf_score = self.get_normalized_performance_score(client_id, model_type)
        
        # Get data weight (normalized by total dataset size)
        data_weight = self.data_weights.get(client_id, 0.5)
        
        # Calculate combined score
        combined_score = (self.alpha * perf_score) + (self.beta * data_weight)
        
        logger.info("Client {} - Model {}: Combined Score = {:.4f} (perf: {:.4f}, data: {:.4f})".format(
            client_id, model_type, combined_score, perf_score, data_weight))
            
        return combined_score
        
    def get_assignment(self, client_id: int) -> str:
        """Get model assignment for a client based on round-robin first, then performance"""
        # First 6 rounds: Use round-robin assignment
        if self.current_round < 6:
            # Get the last model assigned to this client
            last_model_idx = self.last_model_assignment[str(client_id)]
            
            # Calculate next model index using round-robin
            next_model_idx = (last_model_idx + 1) % len(self.model_types)
            
            # Get the next model type
            selected_model = self.model_types[next_model_idx]
            
            # Update last model assignment
            self.last_model_assignment[str(client_id)] = next_model_idx
            self.round_assignments[client_id] = selected_model
            
            logger.info("Round {}: Client {} assigned model {} (round-robin index: {})".format(
                self.current_round + 1, client_id, selected_model, next_model_idx))
                
            return selected_model
            
        # After round 6: Use combined score-based assignment
        # Check if client needs forced participation
        if self.current_round - self.last_participation[client_id] >= self.min_participation_rounds:
            # Force participation with best performing model based on combined score
            best_model = max(
                self.model_types,
                key=lambda m: self.get_combined_score(client_id, m)
            )
            self.round_assignments[client_id] = best_model
            logger.info("Forced participation round {}: Client {} assigned best model {} (combined score: {:.4f})".format(
                self.current_round + 1, client_id, best_model, 
                self.get_combined_score(client_id, best_model)))
            return best_model
            
        # Calculate combined scores for all model types
        model_scores = {
            model_type: self.get_combined_score(client_id, model_type)
            for model_type in self.model_types
        }
        
        # Select model with highest combined score
        selected_model = max(model_scores.items(), key=lambda x: x[1])[0]
        self.round_assignments[client_id] = selected_model
        
        logger.info("Combined score round {}: Client {} assigned model {} (score: {:.4f})".format(
            self.current_round + 1, client_id, selected_model, model_scores[selected_model]))
            
        return selected_model
        
    def confirm_assignment(self, client_id: int, model_type: str, metrics: dict):
        """Confirm model assignment and update participation counts"""
        self.client_history[client_id].append(model_type)
        self.model_usage[model_type] += 1
        self.client_model_counts[client_id][model_type] += 1
        self.participation_counts[client_id] += 1
        self.last_participation[client_id] = self.current_round
        
        # Update performance score using F1 score
        if 'f1_score' in metrics:
            self.update_performance_score(client_id, model_type, metrics['f1_score'])
        
    def next_round(self):
        """Move to next round and update state"""
        self.current_round += 1
        self.round_assignments.clear()
        logger.info("Starting round {} with performance-aware scheduling".format(self.current_round))
        
    def get_round_stats(self) -> str:
        """Get detailed statistics for the current round"""
        stats = ["Round {} Assignments:".format(self.current_round)]
        for client_id, model_type in sorted(self.round_assignments.items()):
            perf_score = self.get_normalized_performance_score(client_id, model_type)
            data_weight = self.data_weights.get(client_id, 0.5)
            combined_score = self.get_combined_score(client_id, model_type)
            
            stats.append(
                "Client {}: {} (perf: {:.2f}, data: {:.2f}, score: {:.2f}, used {} times)".format(
                    client_id, model_type, perf_score, data_weight, combined_score,
                    self.client_model_counts[client_id][model_type]
                )
            )
        return "\n".join(stats)

# Global variables with lock
lock = threading.Lock()
round_condition = threading.Condition(lock)
client_updates = defaultdict(dict)
global_models = {}
scheduler = ModelScheduler(MAX_CLIENTS, MODEL_TYPES)
active_clients = set()
extra_round_barrier = set()
extra_round_barrier_condition = threading.Condition(lock)
client_connections = {}
extra_rounds_started = False  # Global flag to track if extra rounds have started
extra_rounds_start_time = None  # Track when extra rounds started

def initialize_global_models():
    for model_type in MODEL_TYPES:
        adaptive_config = {
            'model_type': model_type,
            'device_type': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        model = PneumoniaCNN(adaptive_config=adaptive_config)
        global_models[model_type] = {
            'state': model.state_dict(),
            'performance': defaultdict(list)
        }
    logger.info("Initialized models for all architecture types")

def receive_data(conn):
    try:
        length_data = recvall(conn, 8)
        if not length_data:
            return None
        total_length = struct.unpack('!Q', length_data)[0]
        data = recvall(conn, total_length)
        if data:
            return pickle.loads(data)
        return None
    except Exception as e:
        logger.error("Error receiving data: {}".format(str(e)))
        return None

def recvall(sock, n):
    data = bytearray()
    while len(data) < n:
        try:
            chunk = sock.recv(min(CHUNK_SIZE, n - len(data)))
            if not chunk:
                return None
            data.extend(chunk)
        except Exception as e:
            logger.error("Error during recv: {}".format(str(e)))
            return None
    return data

def send_data(conn, data):
    try:
        serialized_data = pickle.dumps(data)
        length = struct.pack('!Q', len(serialized_data))
        conn.sendall(length)
        for i in range(0, len(serialized_data), CHUNK_SIZE):
            chunk = serialized_data[i:i + CHUNK_SIZE]
            conn.sendall(chunk)
        return True
    except Exception as e:
        logger.error("Error sending data: {}".format(str(e)))
        return False

def handle_client(conn, addr, client_id):
    global active_clients, extra_round_barrier, client_connections, extra_rounds_started, extra_rounds_start_time
    try:
        # Set socket timeout
        conn.settimeout(120.0)  # Increased from 60.0 to 120.0 seconds
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)  # Enable keep-alive
        
        # Receive initial registration message
        registration_msg = receive_data(conn)
        if isinstance(registration_msg, dict) and 'client_id' in registration_msg:
            client_id = registration_msg['client_id']
            dataset_size = registration_msg.get('dataset_size', 0)
            logger.info("Registered Client {} from {} with dataset size {}".format(
                client_id, addr, dataset_size))
            
            # Update scheduler with dataset size
            total_data_size = sum(scheduler.data_weights.values()) + dataset_size
            scheduler.update_data_weight(int(client_id), dataset_size, total_data_size)
        else:
            logger.error("Invalid registration message format: {}".format(registration_msg))
            conn.close()
            return
            
        logger.info("Client {} connected from {}".format(client_id, addr))
        with lock:
            active_clients.add(client_id)
            
        rounds_completed = 0
        in_extra_rounds = False
        
        while True:
            try:
                # Check if we're in extra rounds mode
                if in_extra_rounds:
                    try:
                        conn.settimeout(5.0)  # Increased from 2.0 to 5.0 seconds
                        peek_msg = receive_data(conn)
                        conn.settimeout(120.0)  # Reset to normal timeout (increased from 60.0)
                        
                        if peek_msg is None:
                            # Send a heartbeat to keep the connection alive
                            send_data(conn, {'heartbeat': True})
                            continue
                        
                        if isinstance(peek_msg, dict) and 'request_model_type' in peek_msg:
                            requested_type = peek_msg['request_model_type']
                            if requested_type not in global_models:
                                logger.error("Client {} requested invalid model: {}".format(client_id, requested_type))
                                continue  # Changed from break to continue
                                
                            # Ensure we have a valid state_dict
                            if 'state' not in global_models[requested_type] or global_models[requested_type]['state'] is None:
                                logger.error("No state_dict for {}. Creating new model.".format(requested_type))
                                model = PneumoniaCNN(adaptive_config={'model_type': requested_type})
                                global_models[requested_type]['state'] = model.state_dict()
                                
                            assignment_data = {
                                'model_type': requested_type,
                                'state_dict': global_models[requested_type]['state']
                            }
                            
                            if not send_data(conn, assignment_data):
                                logger.error("Failed to send {} to client {}".format(requested_type, client_id))
                                continue  # Changed from break to continue
                                
                            logger.info("Sent requested model {} to Client {} for extra round".format(requested_type, client_id))
                            client_update = receive_data(conn)
                            
                            if client_update is None:
                                logger.error("Client {} disconnected during extra round training".format(client_id))
                                continue  # Changed from break to continue
                                
                            logger.info("Received extra round update from Client {} for model {}".format(client_id, requested_type))
                            with round_condition:
                                extra_round_id = "extra_" + str(rounds_completed - 6)
                                if extra_round_id not in client_updates:
                                    client_updates[extra_round_id] = {}
                                client_updates[extra_round_id][client_id] = {
                                    'model_type': requested_type,
                                    'state_dict': client_update['state_dict'],
                                    'metrics': client_update.get('metrics', {})
                                }
                                
                                # Update global model immediately for extra rounds
                                if 'state' not in global_models[requested_type] or global_models[requested_type]['state'] is None:
                                    global_models[requested_type]['state'] = client_update['state_dict']
                                else:
                                    # Simple averaging for extra rounds
                                    current_state = global_models[requested_type]['state']
                                    new_state = {}
                                    for key in current_state.keys():
                                        if key in client_update['state_dict']:
                                            new_state[key] = (current_state[key] + client_update['state_dict'][key]) / 2.0
                                    global_models[requested_type]['state'] = new_state
                                
                                logger.info("Updated global model for {} after extra round from client {}".format(requested_type, client_id))
                                
                                if len(client_updates[extra_round_id]) == len(active_clients) and len(active_clients) > 0:
                                    logger.info("All clients completed extra round {}".format(extra_round_id))
                                    # Save models after each extra round
                                    extra_round_num = int(rounds_completed - 6)
                                    save_models_per_extra_round(global_models, extra_round_num)
                                    client_updates[extra_round_id].clear()
                                    round_condition.notify_all()
                        elif isinstance(peek_msg, dict) and peek_msg.get('heartbeat'):
                            # Send heartbeat response
                            send_data(conn, {'heartbeat': True})
                            continue
                        else:
                            # Unknown message, continue
                            continue
                                    
                    except socket.timeout:
                        # Send a heartbeat to keep the connection alive
                        send_data(conn, {'heartbeat': True})
                        continue
                    except Exception as e:
                        logger.error("Error in extra rounds for Client {}: {}".format(client_id, str(e)))
                        continue  # Changed from break to continue
                    continue
                
                # Regular rounds logic
                with round_condition:
                    while True:
                        if client_id not in client_updates[scheduler.current_round]:
                            break
                        round_condition.wait()
                    assigned_model = scheduler.get_assignment(client_id)
                    logger.info("Assigned {} to Client {}".format(assigned_model, client_id))
                    
                    # Ensure we have a valid state_dict
                    if 'state' not in global_models[assigned_model] or global_models[assigned_model]['state'] is None:
                        logger.error("No state_dict for {}. Creating new model.".format(assigned_model))
                        model = PneumoniaCNN(adaptive_config={'model_type': assigned_model})
                        global_models[assigned_model]['state'] = model.state_dict()
                    
                    assignment_data = {
                        'model_type': assigned_model,
                        'state_dict': global_models[assigned_model]['state']
                    }
                    
                    if not send_data(conn, assignment_data):
                        logger.error("Failed to send assignment to Client {}".format(client_id))
                        break
                    logger.info("Sent {} model to Client {}".format(assigned_model, client_id))
                
                client_update = receive_data(conn)
                if client_update is None:
                    logger.error("No update received from Client {}".format(client_id))
                    break
                    
                if not isinstance(client_update, dict) or 'state_dict' not in client_update:
                    logger.info("Received non-model update from Client {}: {}".format(client_id, client_update))
                    if isinstance(client_update, dict):
                        if client_update.get('round_complete'):
                            # Client completed round 6, check if extra rounds have started
                            if extra_rounds_started:
                                logger.info("Client {} completed round 6, sending extra rounds signal".format(client_id))
                                send_data(conn, {'extra_rounds_start': True})
                            continue
                        elif client_update.get('heartbeat'):
                            # Send heartbeat response
                            send_data(conn, {'heartbeat': True})
                            # If extra rounds have started and client is waiting, send the signal
                            if extra_rounds_started and not in_extra_rounds:
                                logger.info("Client {} sent heartbeat, sending extra rounds signal".format(client_id))
                                send_data(conn, {'extra_rounds_start': True})
                            continue
                        else:
                            break
                    else:
                        break
                        
                with round_condition:
                    client_updates[scheduler.current_round][client_id] = {
                        'model_type': assigned_model,
                        'state_dict': client_update['state_dict'],
                        'metrics': client_update.get('metrics', {})
                    }
                    scheduler.confirm_assignment(client_id, assigned_model, client_update.get('metrics', {}))
                    logger.info("Received {} update from Client {}".format(assigned_model, client_id))
                    rounds_completed += 1
                    
                    if len(client_updates[scheduler.current_round]) == len(active_clients) and len(active_clients) > 0:
                        logger.info("\n" + scheduler.get_round_stats())
                        round_updates = defaultdict(list)
                        for cid in active_clients:
                            update = client_updates[scheduler.current_round].get(cid)
                            if update:
                                round_updates[update['model_type']].append(update['state_dict'])
                                
                        for model_type, updates in round_updates.items():
                            if updates:
                                new_state = {}
                                first_update = updates[0]
                                for key in first_update.keys():
                                    stacked = torch.stack([update[key].float() for update in updates])
                                    new_state[key] = torch.mean(stacked, dim=0)
                                global_models[model_type]['state'] = new_state
                                
                        client_updates[scheduler.current_round].clear()
                        # Save models after each round
                        save_models_per_round(global_models, scheduler.current_round)
                        scheduler.next_round()
                        round_condition.notify_all()
                        
                # Handle transition to extra rounds
                if rounds_completed == 6:
                    with extra_round_barrier_condition:
                        # Check if extra rounds have already started
                        if extra_rounds_started:
                            logger.info("Client {} completed round 6 but extra rounds already started. Proceeding directly.".format(client_id))
                            in_extra_rounds = True
                        else:
                            extra_round_barrier.add(client_id)
                            client_connections[client_id] = conn
                            logger.info("Client {} ready for extra rounds ({}/{})".format(
                                client_id, len(extra_round_barrier), len(active_clients)))
                            
                            # Wait for all active clients to complete round 6, with timeout
                            timeout_start = time.time()
                            timeout_duration = 60.0  # 60 seconds timeout
                            
                            while len(extra_round_barrier) < len(active_clients) and len(active_clients) > 0:
                                # Check if we should timeout
                                if time.time() - timeout_start > timeout_duration:
                                    logger.warning("Timeout waiting for all clients. Proceeding with {} clients".format(
                                        len(extra_round_barrier)))
                                    break
                                extra_round_barrier_condition.wait(timeout=1.0)  # Wait with timeout
                                
                            # If this is the last client to arrive or timeout occurred, notify all clients to start extra rounds
                            if len(extra_round_barrier) >= len(active_clients) or extra_rounds_started:
                                if not extra_rounds_started:
                                    logger.info("All active clients synchronized. Starting extra rounds...")
                                    extra_rounds_started = True
                                    extra_rounds_start_time = time.time()
                                    
                                # Send multiple notifications to ensure all clients receive it
                                for _ in range(3):  # Send 3 times to ensure delivery
                                    for cid, client_conn in client_connections.items():
                                        try:
                                            send_data(client_conn, {'extra_rounds_start': True})
                                            logger.info("Notified client {} to start extra rounds".format(cid))
                                            # Add a small delay between notifications
                                            time.sleep(0.5)
                                        except Exception as e:
                                            logger.error("Failed to notify client {}: {}".format(cid, str(e)))
                                    extra_round_barrier.clear()
                                    client_connections.clear()
                                    extra_round_barrier_condition.notify_all()
                                    in_extra_rounds = True  # Set the flag to enter extra rounds mode
                                    break
                                
                            # Wait for the barrier to be cleared before proceeding
                            while len(extra_round_barrier) > 0 and not extra_rounds_started:
                                extra_round_barrier_condition.wait(timeout=1.0)
                                
                            # If extra rounds have already started, set the flag for this client
                            if extra_rounds_started:
                                in_extra_rounds = True
                            
            except socket.timeout:
                logger.warning("Socket timeout for Client {}".format(client_id))
                continue
            except Exception as e:
                logger.error("Error in client loop for Client {}: {}".format(client_id, str(e)))
                break
                
    except Exception as e:
        logger.error("Error handling client {}: {}".format(client_id, str(e)))
    finally:
        with lock:
            if client_id in active_clients:
                active_clients.remove(client_id)
        conn.close()
        logger.info("Client {} disconnected".format(client_id))

def save_final_models(global_models, save_dir="saved_models"):
    os.makedirs(save_dir, exist_ok=True)
    for model_type, model_data in global_models.items():
        state = model_data.get('state')
        if state is not None:
            torch.save(state, os.path.join(save_dir, "{}_final.pth".format(model_type)))
            logger.info("Saved {} final model to {}/{}_final.pth".format(model_type, save_dir, model_type))

def save_models_per_round(global_models, round_num, save_dir="saved_models_rounds"):
    os.makedirs(save_dir, exist_ok=True)
    for model_type, model_data in global_models.items():
        state = model_data.get('state')
        if state is not None:
            path = os.path.join(save_dir, "{}_round_{}.pth".format(model_type, round_num))
            torch.save(state, path)
            logger.info("Saved {} model for round {} to {}".format(model_type, round_num, path))
            print("[SAVE] {} model for round {} saved to {}".format(model_type, round_num, path))

def save_models_per_extra_round(global_models, extra_round_num, save_dir="saved_models_rounds"):
    os.makedirs(save_dir, exist_ok=True)
    for model_type, model_data in global_models.items():
        state = model_data.get('state')
        if state is not None:
            path = os.path.join(save_dir, "{}_extra_{}.pth".format(model_type, extra_round_num))
            torch.save(state, path)
            logger.info("Saved {} model for extra round {} to {}".format(model_type, extra_round_num, path))
            print("[SAVE] {} model for extra round {} saved to {}".format(model_type, extra_round_num, path))

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
            # Exit if all expected clients have connected and all have disconnected
            if client_id_counter > MAX_CLIENTS and len(active_clients) == 0:
                logger.info("All clients have finished. Shutting down server.")
                break
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
    finally:
        server_sock.close()
        save_final_models(global_models)

if __name__ == "__main__":
    main()