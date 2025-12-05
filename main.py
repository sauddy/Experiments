import torch
import torch.nn as nn


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


class Dynamic_FC(nn.Module):
    """
    Dynamic Fully Connected Layer with treatment-varying coefficients.

    Instead of fixed weights W, this layer learns W(t) that varies smoothly
    with treatment t using spline basis functions:
    output = W(t) · x + b(t)
    where W(t) = Σᵢ Wᵢ · basis_i(t)
    """

    def __init__(self, ind, outd, degree, knots, act="relu", isbias=1, islastlayer=0):
        super(Dynamic_FC, self).__init__()
        self.ind = ind
        self.outd = outd
        self.degree = degree
        self.knots = knots
        self.islastlayer = islastlayer
        self.isbias = isbias

        # Spline basis
        self.spb = Truncated_power(degree, knots)
        self.d = self.spb.num_of_basis  # number of basis functions

        # Weight tensor: (input_dim, output_dim, num_basis)
        self.weight = nn.Parameter(
            torch.rand(self.ind, self.outd, self.d), requires_grad=True
        )

        if self.isbias:
            self.bias = nn.Parameter(torch.rand(self.outd, self.d), requires_grad=True)
        else:
            self.bias = None

        # Activation function
        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "tanh":
            self.act = nn.Tanh()
        elif act == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            self.act = None

    def forward(self, x):
        """
        Args:
            x: Input tensor, shape (batch_size, ind)
               First column is treatment, rest are features

        Returns:
            Output tensor, shape (batch_size, outd) if last layer
                          or (batch_size, outd+1) if intermediate layer
        """
        # Split treatment and features
        x_treat = x[:, 0]
        x_feature = x[:, 1:]

        # Compute treatment-varying weights: W(t) = Σ weight_i * basis_i(t)
        # weight: (ind, outd, d), x_feature: (bs, ind-1)
        x_feature_weight = torch.matmul(self.weight.T, x_feature.T).T  # (bs, outd, d)

        # Get basis function values for treatment
        x_treat_basis = self.spb.forward(x_treat)  # (bs, d)
        x_treat_basis_ = torch.unsqueeze(x_treat_basis, 1)  # (bs, 1, d)

        # Apply treatment-varying transformation
        out = torch.sum(x_feature_weight * x_treat_basis_, dim=2)  # (bs, outd)

        # Add bias term
        if self.isbias:
            out_bias = torch.matmul(self.bias, x_treat_basis.T).T
            out = out + out_bias

        # Apply activation
        if self.act is not None:
            out = self.act(out)

        # Concatenate treatment for intermediate layers
        # This allows subsequent layers to also have treatment-varying behavior
        if not self.islastlayer:
            out = torch.cat((torch.unsqueeze(x_treat, 1), out), 1)

        return out

    """Multi-layer perceptron with ReLU activations"""

    def __init__(self, input_dim, hidden_dims):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class DTRnet(nn.Module):
    """
    Dynamic Treatment Regime Network (DTRnet)

    DTRnet learns disentangled representations to precisely control selection bias
    for estimating individual treatment effects with continuous treatments.

    Architecture Overview:
    =====================

    X (covariates)
        ↓
        ├→ φ_γ(X) (gamma) ─┐
        ├→ φ_δ(X) (delta) ─┼─→ [φ_γ, φ_δ] → Density_Block → g (balancing density)
        └→ φ_ψ(X) (psi)   ─┤
                            ├─→ φ_ψ → Density_Block → g_psi (outcome density)
                            └─→ [φ_δ, φ_ψ] ─┐
                                             ├→ [t, φ_δ, φ_ψ] → Q (varying coef network) → Ŷ
    t (treatment) ──────────────────────────┘

    Components:
    -----------
    1. Three Embedding Networks:
       - φ_γ (gamma): Captures confounding factors (treatment assignment mechanism)
       - φ_δ (delta): Shared representation for both balancing and outcome
       - φ_ψ (psi): Focuses on outcome prediction

    2. Two Density Estimators:
       - g: Balancing density p(t|φ_γ,φ_δ) - corrects selection bias via reweighting
       - g_psi: Outcome-focused density p(t|φ_ψ) - used in outcome prediction

    3. Varying Coefficient Network (Q):
       - Multi-layer network with treatment-varying weights
       - Each layer has coefficients W(t) that vary smoothly with treatment
       - Predicts outcome: Ŷ = Q(t, φ_δ, φ_ψ)

    Forward Pass Returns:
    --------------------
    - g: Balancing density for inverse propensity weighting
    - Q: Predicted outcome
    - gamma, delta, psi: Learned representations (for regularization)
    - g_psi: Outcome-focused density
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=200,
        num_grid=10,
        degree=2,
        knots=[0.33, 0.67],
        outcome_hidden_dims=[200, 200],
    ):
        super(DTRnet, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_grid = num_grid
        self.degree = degree
        self.knots = knots
        self.outcome_hidden_dims = outcome_hidden_dims

        # Three embedding networks (disentangled representations)
        self.hidden_features_gamma = MLP(input_dim, [hidden_dim, hidden_dim])  # φ_γ
        self.hidden_features_delta = MLP(input_dim, [hidden_dim, hidden_dim])  # φ_δ
        self.hidden_features_psi = MLP(input_dim, [hidden_dim, hidden_dim])  # φ_ψ

        # Balancing density estimator: p(t | φ_γ, φ_δ)
        # Input: concatenation of gamma and delta representations
        self.density_estimator_head = Density_Block(
            num_grid=num_grid, ind=2 * hidden_dim, isbias=1  # [φ_γ, φ_δ]
        )

        # Outcome-focused density estimator: p(t | φ_ψ)
        # Input: psi representation only
        self.density_estimator_head_psi = Density_Block(
            num_grid=num_grid, ind=hidden_dim, isbias=1  # φ_ψ only
        )

        # Varying coefficient network for outcome prediction
        # Builds a multi-layer network with treatment-varying weights
        blocks = []

        # First layer input: [t, φ_δ, φ_ψ] -> dimension is 1 + 2*hidden_dim
        input_size = 1 + 2 * hidden_dim

        for i, out_size in enumerate(outcome_hidden_dims):
            is_last = i == len(outcome_hidden_dims) - 1
            blocks.append(
                Dynamic_FC(
                    ind=input_size,
                    outd=out_size,
                    degree=degree,
                    knots=knots,
                    act="relu",
                    isbias=1,
                    islastlayer=0,  # Intermediate layers concatenate treatment
                )
            )
            # Next layer input includes treatment + previous output
            input_size = 1 + out_size

        # Final layer: predicts scalar outcome
        final_layer = Dynamic_FC(
            ind=input_size,
            outd=1,
            degree=degree,
            knots=knots,
            act=None,  # No activation for regression output
            isbias=1,
            islastlayer=1,  # Last layer doesn't concatenate treatment
        )
        blocks.append(final_layer)

        self.Q = nn.Sequential(*blocks)

    def forward(self, t, x):
        """
        Forward pass

        Args:
            t: Treatment values, shape (batch_size,) - should be in [0,1]
            x: Covariates, shape (batch_size, input_dim)

        Returns:
            Tuple of (g, Q, gamma, delta, psi, g_psi):
            - g: Balancing density p(t|φ_γ,φ_δ), shape (batch_size,)
            - Q: Predicted outcome, shape (batch_size,)
            - gamma: φ_γ representation, shape (batch_size, hidden_dim)
            - delta: φ_δ representation, shape (batch_size, hidden_dim)
            - psi: φ_ψ representation, shape (batch_size, hidden_dim)
            - g_psi: Outcome density p(t|φ_ψ), shape (batch_size,)
        """
        # Step 1: Get three disentangled representations
        gamma = self.hidden_features_gamma(x)  # φ_γ(X)
        delta = self.hidden_features_delta(x)  # φ_δ(X)
        psi = self.hidden_features_psi(x)  # φ_ψ(X)

        # Step 2: Concatenate representations for different purposes
        gamma_delta = torch.cat((gamma, delta), dim=1)  # [φ_γ, φ_δ] for balancing
        delta_psi = torch.cat((delta, psi), dim=1)  # [φ_δ, φ_ψ] for outcome

        # Step 3: Prepare input for varying coefficient network
        t_hidden = torch.cat((torch.unsqueeze(t, 1), delta_psi), dim=1)  # [t, φ_δ, φ_ψ]

        # Step 4: Compute balancing density (for reweighting)
        g = self.density_estimator_head(t, gamma_delta)  # p(t | φ_γ, φ_δ)

        # Step 5: Compute outcome-focused density
        g_psi = self.density_estimator_head_psi(t, psi)  # p(t | φ_ψ)

        # Step 6: Predict outcome through varying coefficient network
        Q = self.Q(t_hidden).squeeze(-1)  # Outcome prediction

        return g, Q, gamma, delta, psi, g_psi


# Test the embeddings
if __name__ == "__main__":
    # Example configuration
    input_dim = 10
    hidden_dim = 200
    num_grid = 10
    degree = 2
    knots = [0.33, 0.67]
    outcome_hidden_dims = [200, 200]
    batch_size = 32

    model = DTRnet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_grid=num_grid,
        degree=degree,
        knots=knots,
        outcome_hidden_dims=outcome_hidden_dims,
    )

    # Create dummy data
    x = torch.randn(batch_size, input_dim)
    t = torch.rand(batch_size)  # Continuous treatment in [0,1]

    # Forward pass
    g, Q, gamma, delta, psi, g_psi = model(t, x)

    print("=" * 60)
    print("DTRnet - Complete Architecture")
    print("=" * 60)
    print(f"\nInput:")
    print(f"  Covariates (X) shape: {x.shape}")
    print(f"  Treatment (t) shape: {t.shape}")

    print(f"\nRepresentations:")
    print(f"  Gamma (φ_γ) shape: {gamma.shape}")
    print(f"  Delta (φ_δ) shape: {delta.shape}")
    print(f"  Psi (φ_ψ) shape: {psi.shape}")

    print(f"\nDensity Estimates:")
    print(f"  Balancing density (g) shape: {g.shape}")
    print(f"  Balancing density values (sample): {g[:5]}")
    print(f"  Outcome density (g_psi) shape: {g_psi.shape}")
    print(f"  Outcome density values (sample): {g_psi[:5]}")

    print(f"\nOutcome Prediction:")
    print(f"  Predicted outcome (Q) shape: {Q.shape}")
    print(f"  Predicted outcome values (sample): {Q[:5]}")
    print("=" * 60)
