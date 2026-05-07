import torch
from torch import nn

from models.mlp import MLP



def test_mlp():
    B = 64
    I = 12
    O = 128
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
        output_activation=nn.Softmax(),
    )
    prediction = model(batch)
    assert prediction.shape == (B, O)
