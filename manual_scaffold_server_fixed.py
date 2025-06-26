import socket
import pickle
import threading
import torch
import struct
from model_and_data import PneumoniaCNN
from fl_client import init_control_variate, average_state_dicts
import time

HOST = '127.0.0.1'
PORT = 8081
NUM_CLIENTS = 3
ROUNDS = 20
SOCKET_TIMEOUT = 300  # increased timeout to 5 minutes
MAX_RETRIES = 5

model = PneumoniaCNN()
global_weights = model.state_dict()
global_c = init_control_variate(model)

client_updates = []
client_cs = []
client_weights = []
client_losses = []
lock = threading.Lock()

def recvall(sock, n):
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def handle_client(conn, addr, round_num):
    global global_weights, global_c
    print("[Server] Handling client {} for round {}".format(addr, round_num+1))
    
    try:
        payload = pickle.dumps({
            'weights': global_weights,
            'c': global_c,
            'num_samples': 500,  # increased to 500 samples
            'round': round_num + 1
        })
        conn.sendall(struct.pack('!I', len(payload)))
        conn.sendall(payload)
        print("[Server] Sent model to client {} for round {}".format(addr, round_num+1))
        
        raw_len = recvall(conn, 4)
        if not raw_len:
            print("[Server] No update received from client {}, closing connection.".format(addr))
            conn.close()
            return False
            
        msg_len = struct.unpack('!I', raw_len)[0]
        received = recvall(conn, msg_len)
        if not received:
            print("[Server] No data received from client {}, closing connection.".format(addr))
            conn.close()
            return False
            
        print("[Server] Received update from client {} for round {}".format(addr, round_num+1))
        update = pickle.loads(received)
        
        with lock:
            client_updates.append(update['weights'])
            client_cs.append(update['c'])
            client_weights.append(update['num_samples'])
            if 'loss' in update:
                client_losses.append(update['loss'])
        
        conn.close()
        print("[Server] Closed connection with client {} for round {}".format(addr, round_num+1))
        return True
        
    except Exception as e:
        print("[Server] Error handling client {}: {}".format(addr, str(e)))
        try:
            conn.close()
        except:
            pass
        return False

def aggregate_and_update():
    global global_weights, global_c
    if len(client_updates) < NUM_CLIENTS:
        print("[Server] Not all clients provided updates. Received {}/{} updates.".format(
            len(client_updates), NUM_CLIENTS))
        return False
        
    global_weights = average_state_dicts(client_updates, client_weights)
    global_c = average_state_dicts(client_cs, client_weights)
    
    if client_losses:
        # Filter out any invalid loss values
        valid_losses = [loss for loss in client_losses if isinstance(loss, (int, float)) and not torch.isnan(torch.tensor(loss))]
        if valid_losses:
            avg_loss = sum(valid_losses) / len(valid_losses)
            print("[Server] Average loss for this round: {:.4f}".format(avg_loss))
        else:
            print("[Server] Warning: No valid loss values received for this round")
    
    return True

def main():
    global client_updates, client_cs, client_weights, client_losses
    
    for rnd in range(ROUNDS):
        print("--- Round {} ---".format(rnd+1))
        retry_count = 0
        
        while retry_count < MAX_RETRIES:
            client_updates = []
            client_cs = []
            client_weights = []
            client_losses = []
            successful_clients = 0
            
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.bind((HOST, PORT))
            server_socket.listen(NUM_CLIENTS)
            server_socket.settimeout(SOCKET_TIMEOUT)
            
            try:
                # Wait for all clients to connect and complete
                for i in range(NUM_CLIENTS):
                    try:
                        print("[Server] Waiting for client {} of {} for round {}".format(
                            i+1, NUM_CLIENTS, rnd+1))
                        conn, addr = server_socket.accept()
                        if handle_client(conn, addr, rnd):
                            successful_clients += 1
                    except socket.timeout:
                        print("[Server] Timeout waiting for client {} in round {}".format(i+1, rnd+1))
                        break
                
                # Check if all clients completed successfully
                if successful_clients == NUM_CLIENTS:
                    if aggregate_and_update():
                        print("[Server] Successfully aggregated updates from all clients for round {}".format(rnd+1))
                        break
                
                retry_count += 1
                if retry_count < MAX_RETRIES:
                    print("[Server] Not all clients completed. Retrying round {} (attempt {}/{})".format(
                        rnd+1, retry_count+1, MAX_RETRIES))
                    time.sleep(5)  # Wait before retrying
                
            except Exception as e:
                print("[Server] Error in round {}: {}".format(rnd+1, str(e)))
                retry_count += 1
            finally:
                server_socket.close()
        
        if retry_count == MAX_RETRIES:
            print("[Server] Failed to complete round {} after {} attempts. Exiting.".format(
                rnd+1, MAX_RETRIES))
            break
            
        print("[Server] Completed round {}".format(rnd+1))
    
    print("Training complete.")
    # Save the final global model weights
    torch.save(global_weights, "global_model_final.pth")

if __name__ == "__main__":
    main() 