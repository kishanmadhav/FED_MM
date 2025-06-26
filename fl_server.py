import flwr as fl
from typing import List, Tuple, Dict, Optional
from flwr.common import Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
import numpy as np

class SynchronousStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 3,
        min_evaluate_clients: int = 3,
        min_available_clients: int = 3,
        evaluate_fn=None,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        accept_failures: bool = False,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
        )
        self.current_round = 0
        self.completed_clients_in_round = set()
        self.current_parameters = None

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return self.current_parameters, {}

        # Check if all clients have completed this round
        for _, fit_res in results:
            client_id = fit_res.metrics["client_id"]
            round_number = fit_res.metrics["round_number"]
            if round_number == self.current_round:
                self.completed_clients_in_round.add(client_id)
                # Store the parameters from the first result as current
                if self.current_parameters is None:
                    self.current_parameters = fit_res.parameters

        if len(self.completed_clients_in_round) < self.min_fit_clients:
            print("Waiting for more clients to complete round {}".format(self.current_round))
            # Return current parameters while waiting
            return self.current_parameters, {
                "round": self.current_round,
                "clients_completed": len(self.completed_clients_in_round),
                "clients_remaining": self.min_fit_clients - len(self.completed_clients_in_round)
            }

        # All clients have completed, reset for next round
        self.completed_clients_in_round.clear()
        self.current_round += 1

        # Aggregate weights
        weights_results = [
            (parameters, num_examples)
            for _, fit_res in results
            for parameters, num_examples in [(fit_res.parameters, fit_res.num_examples)]
        ]
        
        aggregated_parameters = self.aggregate_fit_results(weights_results)
        self.current_parameters = aggregated_parameters  # Store the new parameters
        
        # Calculate average loss if available
        total_examples = sum(num_ex for _, num_ex in weights_results)
        losses = [fit_res.metrics["loss"] * fit_res.num_examples for _, fit_res in results]
        average_loss = sum(losses) / total_examples if total_examples > 0 else 0.0

        return aggregated_parameters, {
            "round": self.current_round - 1,
            "clients_completed": self.min_fit_clients,
            "loss": average_loss
        }

    def aggregate_fit_results(
        self, results: List[Tuple[fl.common.Parameters, int]]
    ) -> fl.common.Parameters:
        """Aggregate fit results using weighted average."""
        if not results:
            return self.current_parameters

        # Calculate the total number of examples used during training
        total_examples = sum([num_examples for _, num_examples in results])

        # Create a list of weights, each multiplied by the number of examples
        weighted_weights = [
            [layer * num_examples for layer in weights] 
            for weights, num_examples in results
        ]

        # Aggregate and return the weighted average
        median_weights = [
            np.sum(layer_updates, axis=0) / total_examples
            for layer_updates in zip(*weighted_weights)
        ]

        return fl.common.ndarrays_to_parameters(median_weights)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientProxy
    ) -> List[Tuple[ClientProxy, Dict]]:
        """Configure the next round of training."""
        # Store initial parameters if not set
        if self.current_parameters is None:
            self.current_parameters = parameters
            
        config = {
            "round_number": self.current_round,
            "min_fit_clients": self.min_fit_clients,
        }
        if self.on_fit_config_fn is not None:
            config.update(self.on_fit_config_fn(server_round))

        clients = client_manager.sample(
            num_clients=self.min_fit_clients,
            min_num_clients=self.min_fit_clients,
        )
        
        return [(client, config) for client in clients]

def main():
    strategy = SynchronousStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        accept_failures=False,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=strategy,
    )

if __name__ == "__main__":
    main() 