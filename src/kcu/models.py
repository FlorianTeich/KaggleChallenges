import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast
from torch import Tensor


class MNIST_CNN_01(nn.Module):
    """
    MNIST CNN Try 01
    """
    def __init__(self):
        super(MNIST_CNN_01, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


ModuleType = Union[str, Callable[..., nn.Module]]

def _make_nn_module(module_type, *args) -> nn.Module:
    return (
        (
            ReGLU()
            if module_type == 'ReGLU'
            else GEGLU()
            if module_type == 'GEGLU'
            else getattr(nn, module_type)(*args)
        )
        if isinstance(module_type, str)
        else module_type(*args)
    )


class ResNet(nn.Module):
    """The ResNet model from [gorishniy2021revisiting].
    References:
        [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
    """

    class Block(nn.Module):
        """The main building block of `ResNet`."""

        def __init__(
            self,
            *,
            d: int,
            d_intermidiate: int,
            bias_first: bool,
            bias_second: bool,
            dropout_first: float,
            dropout_second: float,
            normalization: ModuleType,
            activation: ModuleType,
            skip_connection: bool,
        ) -> None:
            """Initialize self."""
            super().__init__()
            self.normalization = _make_nn_module(normalization, d)
            self.linear_first = nn.Linear(d, d_intermidiate, bias_first)
            self.activation = _make_nn_module(activation)
            self.dropout_first = nn.Dropout(dropout_first)
            self.linear_second = nn.Linear(d_intermidiate, d, bias_second)
            self.dropout_second = nn.Dropout(dropout_second)
            self.skip_connection = skip_connection

        def forward(self, x: Tensor) -> Tensor:
            """Perform the forward pass."""
            x_input = x
            x = self.normalization(x)
            x = self.linear_first(x)
            x = self.activation(x)
            x = self.dropout_first(x)
            x = self.linear_second(x)
            x = self.dropout_second(x)
            if self.skip_connection:
                x = x_input + x
            return x

    class Head(nn.Module):
        """The final module of `ResNet`."""

        def __init__(
            self,
            *,
            d_in: int,
            d_out: int,
            bias: bool,
            normalization: ModuleType,
            activation: ModuleType,
        ) -> None:
            """Initialize self."""
            super().__init__()
            self.normalization = _make_nn_module(normalization, d_in)
            self.activation = _make_nn_module(activation)
            self.linear = nn.Linear(d_in, d_out, bias)

        def forward(self, x: Tensor) -> Tensor:
            """Perform the forward pass."""
            if self.normalization is not None:
                x = self.normalization(x)
            x = self.activation(x)
            x = self.linear(x)
            return x

    def __init__(
        self,
        *,
        d_in: int,
        n_blocks: int,
        d: int,
        d_intermidiate: int,
        dropout_first: float,
        dropout_second: float,
        normalization: ModuleType,
        activation: ModuleType,
        d_out: int,
    ) -> None:
        """Initialize self.
        Warning:
            The `make_baseline` method is the recommended constructor. Use `__init__`
            only if you are sure that you need it.
        """
        super().__init__()

        self.first_layer = nn.Linear(d_in, d)
        if d is None:
            d = d_in
        self.blocks = nn.Sequential(
            *[
                ResNet.Block(
                    d=d,
                    d_intermidiate=d_intermidiate,
                    bias_first=True,
                    bias_second=True,
                    dropout_first=dropout_first,
                    dropout_second=dropout_second,
                    normalization=normalization,
                    activation=activation,
                    skip_connection=True,
                )
                for _ in range(n_blocks)
            ]
        )
        self.head = ResNet.Head(
            d_in=d,
            d_out=d_out,
            bias=True,
            normalization=normalization,
            activation=activation,
        )

    @classmethod
    def make_baseline(
        cls: Type['ResNet'],
        *,
        d_in: int,
        d: int,
        d_intermidiate: int,
        dropout_first: float,
        dropout_second: float,
        n_blocks: int,
        d_out: int,
    ) -> 'ResNet':
        """Create a "baseline" `ResNet`.
        It is a user-friendly alternative to `__init__`. This variation of ResNet was
        used in the original paper.
        """
        return cls(
            d_in=d_in,
            n_blocks=n_blocks,
            d=d,
            d_intermidiate=d_intermidiate,
            dropout_first=dropout_first,
            dropout_second=dropout_second,
            normalization='BatchNorm1d',
            activation='ReLU',
            d_out=d_out,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Perform the forward pass."""
        x = self.first_layer(x)
        x = self.blocks(x)
        x = self.head(x)
        return x
