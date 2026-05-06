from torch import nn


class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dims: list[int],
            hidden_activation: nn.Module | None = None,
            output_activation: nn.Module | None = None,
            bias: bool = True,
            is_normalized: bool = False,
    ):
        super().__init__()

        # Define MLP layers
        self.mlp_layers = nn.Sequential()

        # Define layers
        if len(hidden_dims) == 0:
            self.mlp_layers.append(nn.Linear(
                in_features=input_dim,
                out_features=output_dim,
                bias=bias,
            ))
        else:
            self.mlp_layers.append(nn.Linear(
                in_features=input_dim,
                out_features=hidden_dims[0],
                bias=bias,
            ))
            for i in range(len(hidden_dims[1:])):
                self.mlp_layers.append(nn.Linear(
                    in_features=hidden_dims[i],
                    out_features=hidden_dims[i+1],
                    bias=bias,
                ))
                if hidden_activation is not None:
                    self.mlp_layers.append(hidden_activation)
            self.mlp_layers.append(nn.Linear(
                in_features=hidden_dims[-1],
                out_features=output_dim,
                bias=bias,
            ))
        if is_normalized:
            self.mlp_layers.append(nn.LayerNorm(
                normalized_shape=output_dim
            ))
        if output_activation is not None:
            self.mlp_layers.append(output_activation)

    def forward(self, x):
        return self.mlp_layers(x)
