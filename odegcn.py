import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


adjoint = False
if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


class PhysicsDrift(nn.Module):
    """
    Physics-inspired drift capturing temporal change and conservation of flow.
    Operates on tensors of shape (B, N, T, F) and returns the same shape.
    """

    def __init__(self, feature_dim, adj, delta_t=1.0):
        super().__init__()
        self.delta_t = delta_t
        self.register_buffer('adj', adj.clone().detach())
        self.flow_proj = nn.Linear(feature_dim, 1)
        self.combine_weight = nn.Parameter(torch.randn(2, feature_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(feature_dim))
        self.last_penalty = None

    def forward(self, x):
        # x: (B, N, T, F)
        flow = self.flow_proj(x).squeeze(-1)  # (B, N, T)

        adj = self.adj.to(x.dtype).to(x.device)
        neighbour_flow = torch.einsum('ij,bjt->bit', adj, flow)
        flux = neighbour_flow - flow  # conservation residual

        # temporal derivative via finite difference
        dt = flow[..., 1:] - flow[..., :-1]
        dt = torch.cat([torch.zeros_like(flow[..., :1]), dt], dim=-1) / max(self.delta_t, 1e-6)

        features = torch.stack([flux, dt], dim=-1)  # (B, N, T, 2)
        weights = torch.tanh(self.combine_weight)  # keep bounded
        physics = torch.einsum('bntk,kf->bntf', features, weights) + self.bias

        self.last_penalty = 0.5 * (flux.pow(2).mean() + dt.pow(2).mean())
        return physics


class DelayPatternModule(nn.Module):
    """
    Learnable delay-aware pattern mixer.
    For each node we extract the last `horizon` hidden states (most recent time slice)
    and match them against `num_patterns` learnable prototypes.
    """

    def __init__(self, feature_dim, horizon=3, num_patterns=8):
        super().__init__()
        self.horizon = horizon
        self.num_patterns = num_patterns
        self.feature_dim = feature_dim
        self.scale = math.sqrt(max(horizon * feature_dim, 1))

        self.pattern_keys = nn.Parameter(torch.randn(num_patterns, horizon, feature_dim) * 0.01)
        self.pattern_values = nn.Parameter(torch.randn(num_patterns, horizon, feature_dim) * 0.01)
        self.query_norm = nn.LayerNorm(horizon * feature_dim)
        self.key_norm = nn.LayerNorm(horizon * feature_dim)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, history, current):
        device = current.device
        dtype = current.dtype
        horizon = self.horizon

        if len(history) == 0:
            history = [current.detach()] * horizon
        selected = history[-horizon:]
        if len(selected) < horizon:
            pad = [history[0].detach()] * (horizon - len(selected))
            selected = pad + selected

        stack = torch.stack(selected, dim=0)  # (H, B, N, T, F)
        # take the most recent time slice from each cached state
        stack_last = stack[..., -1, :]  # (H, B, N, F)
        stack_last = stack_last.permute(1, 2, 0, 3)  # (B, N, H, F)
        B, N, H, F = stack_last.shape
        hist_flat = stack_last.reshape(B * N, H * F)
        hist_flat = self.query_norm(hist_flat)

        keys = self.pattern_keys.reshape(self.num_patterns, -1)
        keys = self.key_norm(keys).transpose(0, 1)  # (H*F, K)
        logits = (hist_flat @ keys) / (self.scale + 1e-6)
        logits = logits * self.temperature
        weights = torch.softmax(logits, dim=-1)  # (B*N, K)

        values = self.pattern_values.reshape(self.num_patterns, -1)  # (K, H*F)
        agg = torch.matmul(weights, values).reshape(B, N, H, F)
        agg = agg.mean(dim=2)  # (B, N, F)
        agg = agg.unsqueeze(2).expand(-1, -1, current.shape[2], -1)

        uniform = 1.0 / self.num_patterns
        penalty = (weights.mean(dim=0) - uniform).pow(2).mean()
        return agg, penalty


# Define the ODE function.
# Input:
# --- t: A tensor with shape [], meaning the current time.
# --- x: A tensor with shape [#batches, dims], meaning the value of x at t.
# Output:
# --- dx/dt: A tensor with shape [#batches, dims], meaning the derivative of x at t.
class ODEFunc(nn.Module):

    def __init__(self, feature_dim, temporal_dim, adj, use_physics=False, physics_config=None,
                 use_delay=False, delay_config=None):
        super(ODEFunc, self).__init__()
        self.adj = adj
        self.use_physics = use_physics
        self.physics_config = physics_config or {}
        self.use_delay = use_delay
        self.delay_config = delay_config or {}
        self.x0 = None
        self.alpha = nn.Parameter(0.8 * torch.ones(adj.shape[1]))
        self.beta = 0.6
        self.w = nn.Parameter(torch.eye(feature_dim))
        self.d = nn.Parameter(torch.zeros(feature_dim) + 1)
        self.w2 = nn.Parameter(torch.eye(temporal_dim))
        self.d2 = nn.Parameter(torch.zeros(temporal_dim) + 1)
        if self.use_physics:
            delta_t = self.physics_config.get('delta_t', 1.0)
            self.physics = PhysicsDrift(feature_dim, adj, delta_t=delta_t)
            self.max_scale = self.physics_config.get('max_scale', 1.0)
            self.physics_scale = nn.Parameter(torch.zeros(feature_dim))
        else:
            self.physics = None
        if self.use_delay:
            horizon = self.delay_config.get('horizon', 3)
            num_patterns = self.delay_config.get('num_patterns', 8)
            self.delay_module = DelayPatternModule(feature_dim, horizon=horizon, num_patterns=num_patterns)
        else:
            self.delay_module = None
        self.last_penalty = None
        self.last_physics = None
        self.last_delay_penalty = None
        self.history = []

    def forward(self, t, x):
        alpha = torch.sigmoid(self.alpha).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        xa = torch.einsum('ij, kjlm->kilm', self.adj, x)

        # ensure the eigenvalues to be less than 1
        d = torch.clamp(self.d, min=0, max=1)
        w = torch.mm(self.w * d, torch.t(self.w))
        xw = torch.einsum('ijkl, lm->ijkm', x, w)

        d2 = torch.clamp(self.d2, min=0, max=1)
        w2 = torch.mm(self.w2 * d2, torch.t(self.w2))
        xw2 = torch.einsum('ijkl, km->ijml', x, w2)

        f = alpha / 2 * xa - x + xw - x + xw2 - x + self.x0
        if self.use_physics:
            physics_term = self.physics(x)
            scale = torch.sigmoid(self.physics_scale).view(1, 1, 1, -1) * self.max_scale
            physics_term = physics_term * scale
            f = f + physics_term
            self.last_penalty = self.physics.last_penalty
            self.last_physics = physics_term
        else:
            self.last_penalty = None
            self.last_physics = None
        horizon = self.delay_config.get('horizon', 3)
        if self.use_delay and self.delay_module is not None:
            delay_out, penalty = self.delay_module(self.history, x)
            f = f + delay_out
            self.last_delay_penalty = penalty
        else:
            self.last_delay_penalty = None
        self.history.append(x.detach())
        horizon = self.delay_config.get('horizon', 3)
        if len(self.history) > horizon:
            self.history = self.history[-horizon:]
        return f

    def reset_history(self):
        self.history = []

    def update_adjacency(self, adj):
        self.adj = adj
        if self.use_physics and hasattr(self, 'physics') and self.physics is not None:
            self.physics.adj = adj.clone().detach()


class ODEblock(nn.Module):
    def __init__(self, odefunc, t=torch.tensor([0,1])):
        super(ODEblock, self).__init__()
        self.t = t
        self.odefunc = odefunc

    def set_x0(self, x0):
        self.odefunc.x0 = x0.clone().detach()

    def reset_history(self):
        if hasattr(self.odefunc, 'reset_history'):
            self.odefunc.reset_history()

    def forward(self, x):
        t = self.t.type_as(x)
        z = odeint(self.odefunc, x, t, method='euler')[1]
        return z


# Define the ODEGCN model.
class ODEG(nn.Module):
    def __init__(self, feature_dim, temporal_dim, adj, time, use_physics=False, physics_config=None,
                 use_delay=False, delay_config=None):
        super(ODEG, self).__init__()
        self.use_physics = use_physics
        self.use_delay = use_delay
        self.odeblock = ODEblock(ODEFunc(feature_dim, temporal_dim, adj, use_physics=use_physics,
                                         physics_config=physics_config,
                                         use_delay=use_delay, delay_config=delay_config),
                                 t=torch.tensor([0, time]))
        self.last_physics_loss = None
        self.last_delay_loss = None

    def forward(self, x):
        self.odeblock.set_x0(x)
        z = self.odeblock(x)
        if self.use_physics:
            self.last_physics_loss = self.odeblock.odefunc.last_penalty
        else:
            self.last_physics_loss = None
        if self.use_delay:
            self.last_delay_loss = self.odeblock.odefunc.last_delay_penalty
        else:
            self.last_delay_loss = None
        return F.relu(z)

    def update_adjacency(self, adj):
        self.odeblock.odefunc.update_adjacency(adj)

    def reset_history(self):
        self.odeblock.reset_history()
