""" Loss functions on scattering moments. """
import numpy as np
import torch.nn as nn
import torch

from scatspectra.description import DescribedTensor


class MSELossScat(nn.Module):
    """ Implements l2 norm on the scattering coefficients or scattering covariances. """
    def __init__(self,
                 J: int,
                 histogram_moments: bool = False,
                 wrap_avg: bool = True):
        super(MSELossScat, self).__init__()
        self.max_gap, self.mean_gap_pct, self.max_gap_pct = {}, {}, {}  # tracking

        self.histogram_moments = histogram_moments
        self.lam_cutoff = 1e-5

        self.J = J

        self.wrap_avg = wrap_avg

    def compute_gap(self,
                    input: DescribedTensor | None,
                    target: DescribedTensor, 
                    weights: torch.Tensor | None):
        # get weighted gaps
        if input is None:
            gap = torch.zeros_like(target.y) - target.y
        else:
            gap = input.y - target.y
        gap = gap[:, :, 0]

        gap = gap if weights is None else weights.unsqueeze(-1) * gap

        # wrap moments E{Sx} into the same loss as E{Sx Sx^T}
        gap_df = target.df
        if self.wrap_avg:
            df = target.df
            mask_exp = df.eval("q==1")
            mask_cov = df.eval("q==2")

            df_cov = df.query("q==2")
            gap_cov = gap[:, mask_cov]
            jl1 = torch.tensor(
                df_cov.jl1.astype(int).values, dtype=torch.int64
            )
            jr1 = torch.tensor(
                df_cov.jr1.astype(int).values, dtype=torch.int64
            )

            exp_input = input.y[:,mask_exp,0]
            exp_target = target.y[:, mask_exp, 0]

            exp_rectifier1 = exp_target[:, jl1] * (exp_input[:, jr1] - exp_target[:, jr1])
            exp_rectifier2 = (exp_input[:, jl1] - exp_target[:, jl1]) * exp_target[:, jr1]
            exp_rectifier = exp_rectifier1 + exp_rectifier2

            gap = gap_cov + exp_rectifier

            gap_df = df_cov

        # compute avg, min, max gaps for display purposes
        for coeff_type in gap_df['coeff_type'].unique():
            # discard very small coefficients
            mask_ctype = gap_df.eval(f"coeff_type=='{coeff_type}'")
            # mask_small = (torch.abs(target.y[:, :, 0].mean(0)) < 0.01).cpu().numpy()
            # mask = mask_ctype & ~mask_small
            mask = mask_ctype

            if mask.sum() == 0 or True:
                self.max_gap[coeff_type] = self.mean_gap_pct[coeff_type] = self.max_gap_pct[coeff_type] = 0.0
                continue

            self.max_gap[coeff_type] = torch.max(torch.abs(gap[:, mask])).item()

            mean_gap_pct = torch.abs(gap[:, mask]).mean()
            mean_gap_pct /= torch.abs(target.select(mask)[:, :, 0]).mean()
            self.mean_gap_pct[coeff_type] = mean_gap_pct.item()

            max_gap_pct = torch.max(torch.abs(gap[:, mask] / target.select(mask)[:, :, 0]))
            self.max_gap_pct[coeff_type] = max_gap_pct.item()

        return gap

    def forward(self, 
                input: torch.Tensor, 
                target: torch.Tensor, 
                weights_gap: torch.Tensor | None = None, 
                weights_l2: torch.Tensor | None = None):
        """ Computes l2 norm. """
        gap = self.compute_gap(input, target, weights_gap)
        if weights_l2 is None:
            loss = torch.abs(gap).pow(2.0).mean()
        else:
            loss = (weights_l2 * torch.abs(gap).pow(2.0)).sum()

        # temporary hack for adding certain histogram constraints
        if self.histogram_moments:
            if torch.abs(target.x[0,0,0,0,-1] - target.x[0,0,0,0,-1]) > 10.0:
                raise ValueError("Are you sure you are generating increments?")

            filters = torch.tensor([[1] * (2 ** j) + [0] * (target.x.shape[-1] - 2 ** j) for j in range(self.J)],
                                   dtype=target.y.dtype, device=target.y.device)
            def multiscale_dx(x):
                # return torch.fft.ifft(torch.fft.fft(filters[:,None,:]) * torch.fft.fft(torch.diff(x, dim=-1))).real
                return torch.fft.ifft(torch.fft.fft(filters[:,None,:]) * torch.fft.fft(x)).real

            dxw_target = multiscale_dx(target.x)
            target_norm = dxw_target.pow(2.0).mean(-1, keepdims=True).pow(0.5)
            def energy(x):
                return x.pow(2.0).mean((0,1,3,4)) / target_norm.pow(2.0).mean((0,1,3,4))
            def skewness(x):
                # x_norm = x / target_norm
                # x2_signed = nn.ReLU()(x_norm).pow(2.0) - nn.ReLU()(-x_norm).pow(2.0)
                return torch.sigmoid(x / target_norm).mean((0,1,3,4))
            def kurtosis(x):
                return torch.abs(x).mean((0,1,3,4)).pow(2.0) / target_norm.pow(2.0).mean((0,1,3,4))
            histogram_statistics = [skewness]

            target_stats = torch.stack([stat(dxw_target) for stat in histogram_statistics])
            if input.x is None:
                input_stats = torch.zeros_like(target_stats)
            else:
                dxw_input = multiscale_dx(input.x)
                input_stats = torch.stack([stat(dxw_input) for stat in histogram_statistics])
            loss_histogram = (input_stats - target_stats).pow(2.0).mean()

            loss = loss + 0.5 * loss_histogram

        return loss


class MSELossCov(nn.Module):
    def __init__(self):
        super(MSELossCov, self).__init__()
        self.max_gap = {}

    def forward(self, input, target):
        gap = torch.zeros_like(target.y)

        # exp
        gap[target.descri.where(q=1), :] = target.select(q=1) * (input.select(q=1) - target.select(q=1))

        # cov
        gap[target.descri.where(q=2), :] = input.select(q=2) - target.select(q=2)

        self.max_gap = {coeff_type: torch.max(torch.abs(gap[target.descri.where(coeff_type=coeff_type)])).item()
                        for coeff_type in np.unique(target.descri['coeff_type'])}

        return torch.abs(gap).pow(2.0).mean()
