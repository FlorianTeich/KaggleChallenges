"""
Capsules
"""
import torch
import torch.nn.functional as F
from torch import nn

TRAIN_ON_GPU = torch.cuda.is_available()


class ConvLayer(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, in_channels=1, out_channels=256):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=9, stride=1, padding=0
        )

    def forward(self, input_x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        input_x = F.relu(self.conv(input_x))
        return input_x


class PrimaryCaps(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, num_capsules=8, in_channels=256, out_channels=32):
        super(PrimaryCaps, self).__init__()
        self.capsules = nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_channels, kernel_size=9, stride=2, padding=0)
                for _ in range(num_capsules)
            ]
        )

    def forward(self, input_x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        batch_size = input_x.size(0)
        u_data = [
            capsule(input_x).view(batch_size, 32 * 6 * 6, 1)
            for capsule in self.capsules
        ]
        u_data = torch.cat(u_data, dim=-1)
        u_squashed = self.squash(u_data)
        return u_squashed

    def squash(self, input_x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        squared_norm = (input_x**2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        output = scale * input_x / torch.sqrt(squared_norm)
        return output


def softmax(input_x, dim=1):
    """_summary_

    Args:
        x (_type_): _description_
        dim (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    transposed_inp = input_x.transpose(dim, len(input_x.size()) - 1)
    softmaxed = F.softmax(
        transposed_inp.contiguous().view(-1, transposed_inp.size(-1)), dim=-1
    )
    return softmaxed.view(*transposed_inp.size()).transpose(
        dim, len(input_x.size()) - 1
    )


def dynamic_routing(b_ij, u_hat, squash, routing_iterations=3):
    """_summary_

    Args:
        b_ij (_type_): _description_
        u_hat (_type_): _description_
        squash (_type_): _description_
        routing_iterations (int, optional): _description_. Defaults to 3.

    Returns:
        _type_: _description_
    """
    for iterations in range(routing_iterations):
        c_ij = softmax(b_ij, dim=2)
        s_j = (c_ij * u_hat).sum(dim=2, keepdim=True)
        v_j = squash(s_j)
        if iterations < routing_iterations - 1:
            a_ij = (u_hat * v_j).sum(dim=-1, keepdim=True)
            b_ij = b_ij + a_ij
    return v_j


class DigitCaps(nn.Module):
    """
    DigitCaps
    """

    def __init__(
        self,
        num_caps=10,
        previous_layer_nodes=32 * 6 * 6,
        in_channels=8,
        out_channels=16,
    ):
        super(DigitCaps, self).__init__()
        self.num_caps = num_caps
        self.previous_layer_nodes = previous_layer_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.data_w = nn.Parameter(
            torch.randn(num_caps, previous_layer_nodes, in_channels, out_channels)
        )

    def forward(self, input_x):
        """_summary_

        Args:
            input_x (_type_): _description_

        Returns:
            _type_: _description_
        """
        input_x = input_x[None, :, :, None, :]
        data_w = self.data_W[:, None, :, :, :]
        x_hat = torch.matmul(input_x, data_w)
        b_ij = torch.zeros(*x_hat.size())
        if TRAIN_ON_GPU:
            b_ij = b_ij.cuda()
        v_j = dynamic_routing(b_ij, x_hat, self.squash, routing_iterations=3)
        return v_j

    def squash(self, input_x):
        """_summary_

        Args:
            input_x (_type_): _description_

        Returns:
            _type_: _description_
        """
        squared_norm = (input_x**2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        out = scale * input_x / torch.sqrt(squared_norm)
        return out


class Decoder(nn.Module):
    """
    Decoder
    """

    def __init__(self, input_vector_length=16, input_capsules=10, hidden_dim=512):
        super(Decoder, self).__init__()
        input_dim = input_vector_length * input_capsules
        self.lin_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, 28 * 28),
            nn.Sigmoid(),
        )

    def forward(self, input_x):
        """_summary_

        Args:
            input_x (_type_): _description_

        Returns:
            _type_: _description_
        """
        classes = (input_x**2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)
        _, max_length_indices = classes.max(dim=1)
        sparse_matrix = torch.eye(10)
        if TRAIN_ON_GPU:
            sparse_matrix = sparse_matrix.cuda()
        data_y = sparse_matrix.index_select(dim=0, index=max_length_indices.data)
        input_x = input_x * data_y[:, :, None]
        input_x = input_x.contiguous()
        flattened_x = input_x.view(input_x.size(0), -1)
        reconstructed = self.lin_layers(flattened_x)
        return reconstructed, data_y


class CapsuleNetwork(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self):
        super(CapsuleNetwork, self).__init__()
        self.conv_layer = ConvLayer()
        self.primary_capsule = PrimaryCaps()
        self.digit_capsule = DigitCaps()
        self.decoder = Decoder()

    def forward(self, input_x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        primary_caps_out = self.primary_capsule(self.conv_layer(input_x))
        caps_out = self.digit_capsule(primary_caps_out).squeeze().transpose(0, 1)
        reconstructed, data_y = self.decoder(caps_out)
        return caps_out, reconstructed, data_y


class CapsuleLoss(nn.Module):
    """
    Capsule Loss
    """

    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=False)

    def forward(self, input_x, labels, images, reconstructions):
        """_summary_

        Args:
            x (_type_): _description_
            labels (_type_): _description_
            images (_type_): _description_
            reconstructions (_type_): _description_

        Returns:
            _type_: _description_
        """
        batch_size = input_x.size(0)
        v_c = torch.sqrt((input_x**2).sum(dim=2, keepdim=True))
        left = F.relu(0.9 - v_c).view(batch_size, -1)
        right = F.relu(v_c - 0.1).view(batch_size, -1)
        margin_loss = labels * left + 0.5 * (1.0 - labels) * right
        margin_loss = margin_loss.sum()
        images = images.view(reconstructions.size()[0], -1)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)
        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)
