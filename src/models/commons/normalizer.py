import torch

from torch import nn

from commons.config import Config


class Normalizer(nn.Module):
    def __init__(
            self,
            config: Config,
            feature_dim: int, # Need-normalizing feature dimension
            max_accumulations: int = 10 ** 10, # Maximum number of accumulations
            epsilon: float = 1e-8
    ):
        super().__init__()
        self._max_accumulations = max_accumulations
        self._epsilon = epsilon

        # Register variable to store accumulative values
        self.register_buffer("_acc_sum", torch.zeros(feature_dim).to(config.device)) # Summation of feature vectors
        self.register_buffer("_acc_sum_squared", torch.zeros(feature_dim).to(config.device)) # Squared summation of feature vectors
        self.register_buffer("_acc_count", torch.zeros(1).to(config.device)) # Amount of feature vectors
        self.register_buffer("_n_accumulations", torch.zeros(1).to(config.device)) # Number of accumulations carried out

    def forward(self, feature_batch: torch.Tensor, is_training: bool = False) -> torch.Tensor:
        # Do statistics accumulation only when the model is run.py and number of accumulations does not exceed maximum number
        if is_training and self._n_accumulations < self._max_accumulations:
            self._accumulate(feature_batch)

        return self._normalize(feature_batch)

    def inverse(self, normalized_feature_batch: torch.Tensor) -> torch.Tensor:
        mean, std_dev = self._get_statistics()
        return normalized_feature_batch * std_dev + mean

    def _accumulate(self, feature_batch: torch.Tensor):
        cur_count = feature_batch.shape[0] # Number of feature vectors in current (considered) feature batch
        cur_sum = torch.sum(feature_batch, dim=0) # Summation of all feature vectors in current (considered) feature batch
        cur_squared_sum = torch.sum(feature_batch ** 2, dim=0) # Squared summation of all feature vectors in current (considered) feature batch

        self._acc_sum += cur_sum # Update accumulated feature vector summation
        self._acc_sum_squared += cur_squared_sum # Update accumulated feature vector squared summation
        self._acc_count += cur_count # Update accumulated feature vector counting
        self._n_accumulations += 1 # Increase number of accumulations carried out so far

    def _normalize(self, batch: torch.Tensor) -> torch.Tensor:
        mean, std_dev = self._get_statistics()
        return (batch - mean) / std_dev

    def _get_statistics(self) -> tuple:
        count = self._acc_count if self._acc_count > 1.0 else 1.0 # Prevent divided by zero error
        mean = self._acc_sum / count # Compute mean
        # Compute standard deviation, always ensure that the value is greater than or equal to a positive epsilon value
        var = self._acc_sum_squared / count - mean ** 2 # Compute variance
        if isinstance(var, float): # This code block is used to prevent (annoying) type warning
            var = torch.tensor(var)
        std_dev = torch.maximum(
            torch.sqrt(torch.maximum(
                var,
                torch.tensor(0.0)
            )), # Ensure that term inside squared root is non-negative
            torch.tensor(self._epsilon)
        )
        return mean, std_dev
