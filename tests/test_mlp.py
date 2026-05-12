import torch
from torch import nn

from models.mlp import MLP


def test_mlp():
    B = 64 # Batch size
    I = 12 # Dimension of input
    O = 128 # Dimension of output
    batch = torch.randn(B, I)

    # Test MLP without hidden layer
    model = MLP(
        input_dim=I,
        output_dim=O,
        hidden_dims=[],
        hidden_activation=nn.ReLU(),
        is_output_normalized=True,
    )
    prediction = model(batch)
    assert prediction.shape == (B, O)

    # Test MLP with hidden layers
    model = MLP(
        input_dim=I,
        output_dim=O,
        hidden_dims=[3, 17, 5, 77, 35],
        output_activation=nn.Softmax(dim=-1),
    )
    prediction = model(batch)
    assert prediction.shape == (B, O)

if __name__ == "__main__":
    test_mlp()
    print("Test passed.")
