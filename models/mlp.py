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

        # Initialize layers and attributes
        self.input_layer = None
        self.hidden_layers = []
        self.output_layer = None
        self.normalization_layer = None
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        # Define layers
        if len(hidden_dims) == 0:
            self.input_layer = nn.Linear(
                in_features=input_dim,
                out_features=output_dim,
                bias=bias,
            )
        else:
            self.input_layer = nn.Linear(
                in_features=input_dim,
                out_features=hidden_dims[0],
                bias=bias,
            )
            for i in range(len(hidden_dims[1:])):
                self.hidden_layers.append(nn.Linear(
                        in_features=hidden_dims[i],
                        out_features=hidden_dims[i+1],
                        bias=bias,
                    )
                )
            self.output_layer = nn.Linear(
                in_features=hidden_dims[-1],
                out_features=output_dim,
                bias=bias,
            )

        # Define normalization layer
        if is_normalized:
            self.normalization_layer = nn.LayerNorm(normalized_shape=output_dim)

    def forward(self, x):
        x = self.input_layer(x)

        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            if self.hidden_activation is not None:
                x = self.hidden_activation(x)

        if self.output_layer is not None:
            x = self.output_layer(x)
        if self.normalization_layer is not None:
            x = self.normalization_layer(x)
        if self.output_activation is not None:
            x = self.output_activation(x)

        return x
