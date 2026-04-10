import flwr as fl
import numpy as np
from pathlib import Path

from utils import parameter_stats


class FedAvgSave(fl.server.strategy.FedAvg):
    def __init__(self, save_dir="reports/params", *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.param_rows = []
        self.comm_rows = []
        self.total_comm_bytes = 0

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        # If aggregation failed, just return what Flower returned
        if aggregated_parameters is None:
            return aggregated_parameters, aggregated_metrics

        ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)

        # Save global model parameters for this round
        np.savez(self.save_dir / f"global_round_{server_round}.npz", *ndarrays)

        # Parameter statistics (optional)
        stats = parameter_stats(ndarrays)
        stats["round"] = int(server_round)
        stats["num_clients"] = int(len(results))
        self.param_rows.append(stats)

        # Communication accounting (bytes)
        global_model_bytes = int(sum(arr.nbytes for arr in ndarrays))

        bytes_up = 0
        for _, fitres in results:
            client_nd = fl.common.parameters_to_ndarrays(fitres.parameters)
            bytes_up += int(sum(a.nbytes for a in client_nd))

        bytes_down = global_model_bytes * int(len(results))
        bytes_round = bytes_up + bytes_down
        self.total_comm_bytes += bytes_round

        self.comm_rows.append({
            "round": int(server_round),
            "num_clients": int(len(results)),
            "global_model_bytes": int(global_model_bytes),
            "bytes_up": int(bytes_up),
            "bytes_down": int(bytes_down),
            "bytes_per_round": int(bytes_round),
            "total_comm_bytes": int(self.total_comm_bytes),
        })

        return aggregated_parameters, aggregated_metrics
