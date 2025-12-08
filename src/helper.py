import torch
import torch.nn as nn


def MLP(dims, act=nn.ReLU, last_act=None):
    layers = []
    for i in range(len(dims) - 1):
        layers += [nn.Linear(dims[i], dims[i - 1])]
        if i < len(dims) - 2:  # prelast layer
            layers += [act()]
        elif last_act is not None:
            layers += [last_act()]
    return nn.Sequential(*layers)


def comp_grid(y, num_grid):
    """
    Compute grid indices and interpolation weights for treatment values

    Args:
        y: Treatment values in [0,1], shape (batch_size,)
        num_grid: Number of grid intervals (B)

    Returns:
        L: Lower grid indices
        U: Upper grid indices
        inter: Interpolation weights (distance from lower to upper)
    """
    # L gives the lower index
    # U gives the upper index
    # inter gives the distance to the lower int
    U = torch.ceil(y * num_grid)
    inter = 1 - (U - y * num_grid)
    L = U - 1
    L += (L < 0).int()
    return L.int().tolist(), U.int().tolist(), inter


class Density_Block(nn.Module):
    """
    Grid-based density estimation block

    Assumes the treatment variable is bounded by [0,1]
    Output grid: 0, 1/B, 2/B, ..., B/B; output dim = B + 1; num_grid = B
    """

    def __init__(self, num_grid, ind, isbias=1):
        super(Density_Block, self).__init__()
        self.ind = ind  # Input dimension
        self.num_grid = num_grid  # Number of grid intervals
        self.outd = num_grid + 1  # Number of grid points
        self.isbias = isbias

        # Learnable parameters
        self.weight = nn.Parameter(torch.rand(self.ind, self.outd), requires_grad=True)
        if self.isbias:
            self.bias = nn.Parameter(torch.rand(self.outd), requires_grad=True)
        else:
            self.bias = None

        self.softmax = nn.Softmax(dim=1)

    def forward(self, t, x):
        """
        Forward pass

        Args:
            t: Treatment values, shape (batch_size,) - assumed in [0,1]
            x: Representation input, shape (batch_size, ind)

        Returns:
            Interpolated density values at treatment positions
        """
        # Linear transformation to get logits for each grid point
        out = torch.matmul(x, self.weight)
        if self.isbias:
            out += self.bias

        # Softmax to get probability distribution over grid
        out = self.softmax(out)

        # Get interpolated density at actual treatment values
        x1 = list(torch.arange(0, x.shape[0]))
        L, U, inter = comp_grid(t, self.num_grid)
        L_out = out[x1, L]
        U_out = out[x1, U]

        # Linear interpolation between grid points
        out = L_out + (U_out - L_out) * inter
        return out


class Truncated_power:
    """
    Truncated power basis functions for spline approximation.
    Data is assumed to be in [0,1].

    Creates basis functions:
    - Polynomial terms: 1, t, t², ..., t^degree
    - Truncated terms: (t - knot_i)_+^degree for each knot
    where (·)_+ = max(0, ·)
    """

    def __init__(self, degree, knots):
        """
        Args:
            degree: int, degree of truncated power basis
            knots: list, knots for spline basis (should not include 0 and 1)
        """
        self.degree = degree
        self.knots = knots
        self.num_of_basis = self.degree + 1 + len(self.knots)
        self.relu = nn.ReLU(inplace=True)

        if self.degree == 0:
            print("Degree should not be set to 0!")
            raise ValueError
        if not isinstance(self.degree, int):
            print("Degree should be int")
            raise ValueError

    def forward(self, x):
        """
        Args:
            x: torch.tensor, shape (batch_size, 1) or (batch_size,)

        Returns:
            Basis function values, shape (batch_size, num_of_basis)
        """
        x = x.squeeze()
        out = torch.zeros(x.shape[0], self.num_of_basis)

        for i in range(self.num_of_basis):
            if i <= self.degree:
                # Polynomial terms
                if i == 0:
                    out[:, i] = 1.0
                else:
                    out[:, i] = x**i
            else:
                # Truncated power terms
                if self.degree == 1:
                    out[:, i] = self.relu(x - self.knots[i - self.degree])
                else:
                    out[:, i] = (
                        self.relu(x - self.knots[i - self.degree - 1])
                    ) ** self.degree

        return out  # (batch_size, num_of_basis)
