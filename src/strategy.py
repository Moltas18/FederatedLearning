from flwr.server.strategy import FedAvg

from src.utils import write_to_file

class CustomFedAvg(FedAvg):
    
    def __init__(self, *, fraction_fit = 1, fraction_evaluate = 1, min_fit_clients = 2, min_evaluate_clients = 2, min_available_clients = 2, evaluate_fn = None, on_fit_config_fn = None, on_evaluate_config_fn = None, accept_failures = True, initial_parameters = None, fit_metrics_aggregation_fn = None, evaluate_metrics_aggregation_fn = None, inplace = True):
        super().__init__(fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate, min_fit_clients=min_fit_clients, min_evaluate_clients=min_evaluate_clients, min_available_clients=min_available_clients, evaluate_fn=evaluate_fn, on_fit_config_fn=on_fit_config_fn, on_evaluate_config_fn=on_evaluate_config_fn, accept_failures=accept_failures, initial_parameters=initial_parameters, fit_metrics_aggregation_fn=fit_metrics_aggregation_fn, evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn, inplace=inplace)

        self.save_path = None

    def _check_save_path(self):
        if not self.save_path:
            NameError('A path for saving results must be initialized in self.path')

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate results from federated evaluation."""
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        
        # Add loss to metrics
        metrics['validation_loss'] = loss
        
        self._check_save_path()
        write_to_file(data=metrics,
                      path=self.save_path,
                      filename='metrics')
        
        return loss, metrics
    
    def aggregate_fit(self, server_round, results, failures):
        """Aggregate results from federated evaluation."""
        parameters_aggregated, metrics = super().aggregate_fit(server_round, results, failures)
        
        self._check_save_path()
        write_to_file(data=metrics,
                      path=self.save_path,
                      filename='metrics')
        
        return parameters_aggregated, metrics