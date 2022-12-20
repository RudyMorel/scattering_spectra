""" Manage generation algorithm. """
from typing import *
from termcolor import colored
from time import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable, grad

from scatcov.layers.described_tensor import DescribedTensor


def compute_w_l2(weights: Dict, model: nn.Module, w_gap: Dict, nchunks: int) -> List[torch.tensor]:
    # normalize by the number of coeffs at each order
    # weights = {c_type: 1 / (scat * nbs[c_type]) for c_type in R.moment_types}
    # # weights = {c_type: 1.0 for c_type in moments.moment_types}
    # weights = {'marginal': 100, 'mixed': 10, 'tenv': 10, 'senv': 10, 'tsenv': 5}
    # # weights = {'norm1': 10, 'norm2': 5, 'norm2lpcross': 5, 'norm2lpmixed': 5, 'lev': 7, 'zum': 1e-12}# for perfect L

    nbs = {c_type: model.count_coefficients(c_type=c_type) for c_type in model.c_types}

    weights_sum = sum([weights[c_type] * nbs[c_type] for c_type in model.c_types])
    weights = {key: weights[key] / weights_sum for key in weights.keys()}
    w_l2 = [None] * nchunks
    for i_chunk in range(nchunks):
        w_l2[i_chunk] = torch.ones_like(w_gap[i_chunk])
        for c_type in model.c_types:
            coeff_type_mask = model.descri_chunked[-1][i_chunk].where(c_type=c_type)
            w_l2[i_chunk][coeff_type_mask] *= weights[c_type]

    assert abs(sum([w.sum() for w in w_l2]) - 1.0) < 1e-6

    return w_l2


class Solver(nn.Module):
    """ A class that contains all information necessary for generation. """
    def __init__(self,
                 model: nn.Module, loss: nn.Module,
                 xf: Optional[np.ndarray] = None,
                 Rxf: Optional[DescribedTensor] = None,
                 x0: Optional[np.ndarray] = None,
                 cuda: bool = False) -> None:
        super(Solver, self).__init__()

        self.model = model
        self.loss = loss

        self.B = xf.shape[0]
        self.N = xf.shape[1]
        self.nchunks = 1
        self.is_cuda = cuda
        self.x0 = torch.DoubleTensor(x0)

        self.res = None, None

        if cuda:
            self.cuda()
            Rxf = Rxf.cuda()

        # compute target representation Rxf
        self.Rxf = Rxf.mean_batch()

        # compute initial loss
        Rx0 = self.model(self.format(x0, requires_grad=False)).mean_batch()
        self.loss0 = self.loss(Rx0, self.Rxf, None, None).detach().cpu().numpy()

    def format(self, x: np.ndarray, requires_grad: Optional[bool] = True) -> torch.tensor:
        """ Transforms x into a compatible format for the embedding. """
        x = torch.tensor(x.reshape(self.B, self.N, -1)).unsqueeze(-2).unsqueeze(-2)
        if self.is_cuda:
            x = x.cuda()
        x = Variable(x, requires_grad=requires_grad)
        return x

    def joint(self, x: np.ndarray) -> Tuple[torch.tensor, torch.tensor]:
        """ Computes the loss on current vector. """

        # format x and set gradient to 0
        x_torch = self.format(x)

        res_max = {c_type: 0.0 for c_type in self.model.module.c_types}
        res_max_pct = {c_type: 0.0 for c_type in self.model.module.c_types}

        # clear gradient
        if x_torch.grad is not None:
            x_torch.grad.x.zero_()

        # compute moments
        # self.time_tracker.start('forward')
        Rxt = self.model(x_torch).mean_batch()

        # compute loss function
        # self.time_tracker.start('loss')
        loss = self.loss(Rxt, self.Rxf, None, None)
        res_max = {c_type: max(res_max[c_type], self.loss.max_gap[c_type] if c_type in self.loss.max_gap else 0.0)
                   for c_type in self.model.module.c_types}
        res_mean_pct = {c_type: max(res_max_pct[c_type], self.loss.mean_gap_pct[c_type] if c_type in self.loss.mean_gap_pct else 0.0)
                        for c_type in self.model.module.c_types}
        res_max_pct = {c_type: max(res_max_pct[c_type], self.loss.max_gap_pct[c_type] if c_type in self.loss.max_gap_pct else 0.0)
                       for c_type in self.model.module.c_types}

        # compute gradient
        # self.time_tracker.start('backward')
        grad_x, = grad([loss], [x_torch], retain_graph=True)

        # move to numpy
        # self.time_tracker.start('move_np')
        grad_x = grad_x.contiguous().detach().cpu().numpy()
        loss = loss.detach().cpu().numpy()

        self.res = loss, grad_x.ravel(), res_max, res_mean_pct, res_max_pct

        return loss, grad_x.ravel()


class SmallEnoughException(Exception):
    pass


class CheckConvCriterion:
    """ A callback function given to the optimizer. """
    def __init__(self,
                 solver: Solver,
                 tol: float,
                 max_wait: Optional[int] = 1000,
                 save_data_evolution_p: Optional[bool] = False):
        self.solver = solver
        self.tol = tol
        self.result = None
        self.next_milestone = None
        self.counter = 0
        self.err = None
        self.max_gap = None
        self.gerr = None
        self.tic = time()

        self.max_wait, self.wait = max_wait, 0
        self.save_data_evolution_p = save_data_evolution_p

        self.logs_loss = []
        self.logs_grad = []
        self.logs_x = []

    def __call__(self, xk: np.ndarray) -> None:
        err, grad_xk, max_gap, mean_gap_pct, max_gap_pct = self.solver.res

        gerr = np.max(np.abs(grad_xk))
        err, gerr = float(err), float(gerr)
        self.err = err
        self.max_gap = max_gap
        self.mean_gap_pct = mean_gap_pct
        self.max_gap_pct = max_gap_pct
        self.gerr = gerr
        self.counter += 1

        self.logs_loss.append(err)
        self.logs_grad.append(gerr)

        if self.next_milestone is None:
            self.next_milestone = 10 ** (np.floor(np.log10(gerr)))

        info_already_printed_p = False
        if self.save_data_evolution_p and not np.log2(self.counter) % 1:
            self.logs_x.append(xk)
            self.print_info_line('SAVED X')
            info_already_printed_p = True

        if np.sqrt(err) <= self.tol:
            self.result = xk
            raise SmallEnoughException()
        elif gerr <= self.next_milestone or self.wait >= self.max_wait:
            if not info_already_printed_p:
                self.print_info_line()
            if gerr <= self.next_milestone:
                self.next_milestone /= 10
            self.wait = 0
        else:
            self.wait += 1

    def print_info_line(self, msg: Optional[str] = '') -> None:
        delta_t = time() - self.tic

        def cap(pct):
            return pct if pct < 1e3 else np.inf

        print(colored(
            f"{self.counter:6}it in {self.hms_string(delta_t)} ( {self.counter / delta_t:.2f}it/s )"
            + " .... "
            + f"err {np.sqrt(self.err):.2E} -- max {max(self.max_gap.values()):.2E}"
            + f" -- maxpct {cap(max(self.max_gap_pct.values())):.3%} -- gerr {self.gerr:.2E}",
            'cyan'))
        print(colored(
            "".join([f"\n -- {c_type:<15} max {value:.2e} -- meanpct {cap(self.mean_gap_pct[c_type]):.2%} "
                     + f"-- maxpct {cap(self.max_gap_pct[c_type]):.1%}, "
                     for c_type, value in self.max_gap.items()])
            + msg,
            'green'))

    @staticmethod
    def hms_string(sec_elapsed: float) -> str:
        """ Format  """
        h = int(sec_elapsed / (60 * 60))
        m = int((sec_elapsed % (60 * 60)) / 60)
        s = sec_elapsed % 60.
        return f"{h}:{m:>02}:{s:>05.2f}"
