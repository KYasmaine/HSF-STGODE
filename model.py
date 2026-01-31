import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from odegcn import ODEG


class Chomp1d(nn.Module):
    """
    extra dimension will be added by padding, remove it
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()


class TemporalConvNet(nn.Module):
    """
    time dilation convolution
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        Args:
            num_inputs : channel's number of input data's feature
            num_channels : numbers of data feature tranform channels, the last is the output channel
            kernel_size : using 1d convolution, so the real kernel is (1, kernel_size) 
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            self.conv = nn.Conv2d(in_channels, out_channels, (1, kernel_size), dilation=(1, dilation_size), padding=(0, padding))
            self.conv.weight.data.normal_(0, 0.01)
            self.chomp = Chomp1d(padding)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

            layers += [nn.Sequential(self.conv, self.chomp, self.relu, self.dropout)]

        self.network = nn.Sequential(*layers)
        self.downsample = nn.Conv2d(num_inputs, num_channels[-1], (1, 1)) if num_inputs != num_channels[-1] else None
        if self.downsample:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """ 
        like ResNet
        Args:
            X : input data of shape (B, N, T, F) 
        """
        # permute shape to (B, F, N, T)
        y = x.permute(0, 3, 1, 2)
        y = F.relu(self.network(y) + self.downsample(y) if self.downsample else y)
        y = y.permute(0, 2, 3, 1)
        return y


class GCN(nn.Module):
    def __init__(self, A_hat, in_channels, out_channels,):
        super(GCN, self).__init__()
        self.A_hat = A_hat
        self.theta = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.reset()
    
    def reset(self):
        stdv = 1. / math.sqrt(self.theta.shape[1])
        self.theta.data.uniform_(-stdv, stdv)

    def forward(self, X):
        y = torch.einsum('ij, kjlm-> kilm', self.A_hat, X)
        return F.relu(torch.einsum('kjlm, mn->kjln', y, self.theta))


class ContinuousReadout(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_queries):
        super().__init__()
        self.num_queries = num_queries
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, base, query_steps):
        """
        base: (B, N, D)
        query_steps: (T,)
        return: (B, N, T)
        """
        B, N, D = base.shape
        T = query_steps.shape[0]
        queries = query_steps.view(1, 1, T, 1).expand(B, N, T, 1)
        base_expanded = base.unsqueeze(2).expand(-1, -1, T, -1)
        inp = torch.cat([base_expanded, queries], dim=-1)
        out = self.mlp(inp).squeeze(-1)
        return out


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, A_hat, use_physics=False,
                 physics_config=None, use_multiscale=False, num_regions=8,
                 use_delay=False, delay_config=None):

        super(STGCNBlock, self).__init__()
        self.A_hat = A_hat
        self.use_multiscale = use_multiscale
        self.num_regions = num_regions
        self.use_delay = use_delay
        self.temporal1 = TemporalConvNet(num_inputs=in_channels,
                                   num_channels=out_channels)
        self.slow_proj = nn.Conv2d(in_channels, out_channels[-1], (1, 1))
        self.odeg = ODEG(out_channels[-1], 12, A_hat, time=6, use_physics=use_physics,
                         physics_config=physics_config,
                         use_delay=use_delay, delay_config=delay_config)
        if self.use_multiscale:
            self.region_assign = nn.Parameter(torch.randn(num_regions, num_nodes))
            slow_adj = torch.eye(num_regions)
            self.slow_ode = ODEG(out_channels[-1], 12, slow_adj, time=6, use_physics=False)
            self.region_norm_eps = 1e-6
        self.temporal2 = TemporalConvNet(num_inputs=out_channels[-1],
                                   num_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self._physics_loss = None
        self._delay_loss = None

    def forward(self, inputs):
        """
        Args:
            inputs: Tensor or tuple (fast, slow) each of shape (batch_size, num_nodes, num_timesteps, num_features)
        Return:
            Tuple (fast_out, slow_out)
        """
        if isinstance(inputs, (list, tuple)):
            X_fast, X_slow = inputs
        else:
            X_fast = X_slow = inputs

        t = self.temporal1(X_fast)
        slow_feature = self.slow_proj(X_slow.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        t = self.odeg(t)
        if self.odeg.use_physics:
            self._physics_loss = self.odeg.last_physics_loss
        else:
            self._physics_loss = None
        if self.use_delay:
            self._delay_loss = self.odeg.last_delay_loss
        else:
            self._delay_loss = None
        if self.use_multiscale:
            assign = torch.softmax(self.region_assign, dim=1).to(t.device)  # (R, N)
            slow_from_fast = torch.einsum('rn,bntf->brtf', assign, t)
            slow_from_stream = torch.einsum('rn,bntf->brtf', assign, slow_feature)
            slow_input = slow_from_fast + slow_from_stream
            with torch.no_grad():
                slow_adj = torch.matmul(assign, torch.matmul(self.A_hat.to(t.device), assign.t()))
                slow_deg = slow_adj.sum(dim=1, keepdim=True).clamp(min=self.region_norm_eps)
                slow_adj = slow_adj / slow_deg
            self.slow_ode.update_adjacency(slow_adj)
            slow_out = self.slow_ode(slow_input)
            t = t + torch.einsum('nr,brtf->bntf', assign.t(), slow_out)
        t = self.temporal2(F.relu(t))

        return self.batch_norm(t), slow_feature

    @property
    def physics_loss(self):
        return self._physics_loss

    @property
    def delay_loss(self):
        return self._delay_loss

    def reset_delay_history(self):
        if hasattr(self, 'odeg') and hasattr(self.odeg, 'reset_history'):
            self.odeg.reset_history()
        if self.use_multiscale and hasattr(self, 'slow_ode'):
            self.slow_ode.reset_history()


class ODEGCN(nn.Module):
    """ the overall network framework """
    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, A_sp_hat, A_se_hat, use_physics=False, physics_config=None,
                 use_multiscale=False, num_regions=8, use_delay=False, delay_config=None,
                 use_continuous_readout=False):


        super(ODEGCN, self).__init__()
        self.use_physics = use_physics
        self.physics_config = physics_config or {}
        self.use_multiscale = use_multiscale
        self.num_regions = num_regions
        self.use_delay = use_delay
        self.delay_config = delay_config or {}
        self.use_continuous_readout = use_continuous_readout
        # spatial graph
        self.sp_blocks = nn.ModuleList(
            [nn.Sequential(
                STGCNBlock(in_channels=num_features, out_channels=[64, 32, 64],
                num_nodes=num_nodes, A_hat=A_sp_hat, use_physics=use_physics,
                physics_config=self.physics_config,
                use_multiscale=self.use_multiscale, num_regions=self.num_regions,
                use_delay=self.use_delay, delay_config=self.delay_config),
                STGCNBlock(in_channels=64, out_channels=[64, 32, 64],
                num_nodes=num_nodes, A_hat=A_sp_hat, use_physics=use_physics,
                physics_config=self.physics_config,
                use_multiscale=self.use_multiscale, num_regions=self.num_regions,
                use_delay=self.use_delay, delay_config=self.delay_config)) for _ in range(3)
            ])
        # semantic graph
        self.se_blocks = nn.ModuleList([nn.Sequential(
                STGCNBlock(in_channels=num_features, out_channels=[64, 32, 64],
                num_nodes=num_nodes, A_hat=A_se_hat, use_physics=use_physics,
                physics_config=self.physics_config,
                use_multiscale=self.use_multiscale, num_regions=self.num_regions,
                use_delay=self.use_delay, delay_config=self.delay_config),
                STGCNBlock(in_channels=64, out_channels=[64, 32, 64],
                num_nodes=num_nodes, A_hat=A_se_hat, use_physics=use_physics,
                physics_config=self.physics_config,
                use_multiscale=self.use_multiscale, num_regions=self.num_regions,
                use_delay=self.use_delay, delay_config=self.delay_config)) for _ in range(3)
            ]) 

        input_dim = num_timesteps_input * 64
        if self.use_continuous_readout:
            self.pred = None
            self.cont_readout = ContinuousReadout(input_dim, hidden_dim=128, num_queries=num_timesteps_output)
            self.register_buffer('query_steps', torch.linspace(0, 1, steps=num_timesteps_output))
        else:
            self.pred = nn.Sequential(
                nn.Linear(input_dim, num_timesteps_output * 32), 
                nn.ReLU(),
                nn.Linear(num_timesteps_output * 32, num_timesteps_output)
            )

    def forward(self, x):
        """
        Args:
            x : tensor or tuple/list (fast, slow) each of shape (B, N, T, F)
        Returns:
            prediction for future of shape (batch_size, num_nodes, num_timesteps_output)
        """
        if isinstance(x, (list, tuple)):
            x_fast, x_slow = x
        else:
            x_fast = x_slow = x

        outs = []
        input_tuple = (x_fast, x_slow)
        # spatial graph
        for blk in self.sp_blocks:
            fast_out, _ = blk(input_tuple)
            outs.append(fast_out)
        # semantic graph
        for blk in self.se_blocks:
            fast_out, _ = blk(input_tuple)
            outs.append(fast_out)
        outs = torch.stack(outs)
        x = torch.max(outs, dim=0)[0]
        x = x.reshape((x.shape[0], x.shape[1], -1))

        if self.use_continuous_readout:
            return self.cont_readout(x, self.query_steps.to(x.device))
        return self.pred(x)

    def physics_regularizer(self):
        if not self.use_physics:
            return None
        total = None
        count = 0
        for module_list in [self.sp_blocks, self.se_blocks]:
            for seq in module_list:
                for block in seq:
                    if hasattr(block, 'physics_loss') and block.physics_loss is not None:
                        total = block.physics_loss if total is None else total + block.physics_loss
                        count += 1
        if total is None or count == 0:
            return None
        return total / count

    def delay_regularizer(self):
        total = None
        count = 0
        for module_list in [self.sp_blocks, self.se_blocks]:
            for seq in module_list:
                for block in seq:
                    if hasattr(block, 'delay_loss') and block.delay_loss is not None:
                        total = block.delay_loss if total is None else total + block.delay_loss
                        count += 1
        if total is None or count == 0:
            return None
        return total / count

    def reset_delay_history(self):
        for module_list in [self.sp_blocks, self.se_blocks]:
            for seq in module_list:
                for block in seq:
                    if hasattr(block, 'reset_delay_history'):
                        block.reset_delay_history()


class StreamEncoder(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.proj = nn.Linear(feature_dim, feature_dim)
        # LayerNorm 在单特征场景下会把所有值归零，这里改成跳过归一化
        self.norm = nn.LayerNorm(feature_dim) if feature_dim > 1 else nn.Identity()

    def forward(self, x):
        # x: (B, N, T, F)
        out = self.proj(x)
        return self.norm(out)


class DualStreamGate(nn.Module):
    def __init__(self, num_streams, hidden_dim=64):
        super().__init__()
        hidden = max(hidden_dim, num_streams * 4)
        self.mlp = nn.Sequential(
            nn.Linear(num_streams, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_streams * 2)
        )
        self.last_fast = None
        self.last_slow = None
        self.fast_hist = []
        self.slow_hist = []

    def forward(self, summary):
        logits = self.mlp(summary)
        fast_logits, slow_logits = torch.chunk(logits, 2, dim=-1)
        fast = torch.softmax(fast_logits, dim=-1)
        slow = torch.softmax(slow_logits, dim=-1)
        self.last_fast = fast.detach()
        self.last_slow = slow.detach()
        self.fast_hist.append(self.last_fast.mean(dim=0).cpu())
        self.slow_hist.append(self.last_slow.mean(dim=0).cpu())
        return fast, slow

    def pop_history(self):
        if self.fast_hist:
            fast_avg = torch.stack(self.fast_hist, dim=0).mean(dim=0)
        else:
            fast_avg = None
        if self.slow_hist:
            slow_avg = torch.stack(self.slow_hist, dim=0).mean(dim=0)
        else:
            slow_avg = None
        self.fast_hist.clear()
        self.slow_hist.clear()
        return fast_avg, slow_avg


class MultiStreamWrapper(nn.Module):
    def __init__(self, base_model, stream_names, feature_dim):
        super().__init__()
        self.base_model = base_model
        self.stream_names = stream_names
        self.encoders = nn.ModuleDict({
            name: StreamEncoder(feature_dim) for name in stream_names
        })
        self.gate = DualStreamGate(len(stream_names))

    def forward(self, stream_inputs):
        if len(stream_inputs) != len(self.stream_names):
            raise ValueError('Number of provided streams does not match configuration.')
        encoded = []
        summaries = []
        for tensor, name in zip(stream_inputs, self.stream_names):
            enc = self.encoders[name](tensor)
            encoded.append(enc)
            summaries.append(enc.mean(dim=(1, 2, 3)))
        summary = torch.stack(summaries, dim=-1)
        fast_weights, slow_weights = self.gate(summary)
        fused_fast = None
        fused_slow = None
        for idx, enc in enumerate(encoded):
            fast_w = fast_weights[:, idx].view(-1, 1, 1, 1)
            slow_w = slow_weights[:, idx].view(-1, 1, 1, 1)
            fused_fast = enc * fast_w if fused_fast is None else fused_fast + enc * fast_w
            fused_slow = enc * slow_w if fused_slow is None else fused_slow + enc * slow_w
        return self.base_model((fused_fast, fused_slow))

    def physics_regularizer(self):
        return self.base_model.physics_regularizer()

    def delay_regularizer(self):
        return self.base_model.delay_regularizer()

    def reset_delay_history(self):
        if hasattr(self.base_model, 'reset_delay_history'):
            self.base_model.reset_delay_history()

    def pop_gate_history(self):
        return self.gate.pop_history()
