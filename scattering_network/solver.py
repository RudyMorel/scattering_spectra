""" Manage generation algorithm. """
from typing import *
from termcolor import colored
from time import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch.autograd import Variable, grad

import utils.complex_utils as cplx
from scattering_network.module_chunk import ModuleChunk
from scattering_network.described_tensor import DescribedTensor


def compute_w_l2(weights, model, w_gap, nchunks):
    # normalize by the number of coeffs at each order
    # weights = {m_type: 1 / (scat * nbs[m_type]) for m_type in R.moment_types}
    # # weights = {m_type: 1.0 for m_type in moments.moment_types}
    # weights = {'marginal': 100, 'mixed': 10, 'tenv': 10, 'senv': 10, 'tsenv': 5}
    # # weights = {'norm1': 10, 'norm2': 5, 'norm2lpcross': 5, 'norm2lpmixed': 5, 'lev': 7, 'zum': 1e-12}# for perfect L

    nbs = {m_type: model.count_coefficients(m_type=m_type) for m_type in model.m_types}

    weights_sum = sum([weights[m_type] * nbs[m_type] for m_type in model.m_types])
    weights = {key: weights[key] / weights_sum for key in weights.keys()}
    w_l2 = [None] * nchunks
    for i_chunk in range(nchunks):
        w_l2[i_chunk] = torch.ones_like(w_gap[i_chunk])
        for m_type in model.m_types:
            coeff_type_mask = model.descri_chunked[-1][i_chunk].where(m_type=m_type)
            w_l2[i_chunk][coeff_type_mask] *= weights[m_type]

    assert abs(sum([w.sum() for w in w_l2]) - 1.0) < 1e-6

    return w_l2


class Solver(nn.Module):
    """ A class that contains all information necessary for generation. """
    def __init__(self, model: ModuleChunk, loss: nn.Module,
                 xf: Optional[torch.Tensor] = None, Rxf: Optional[DescribedTensor] = None,
                 x0: Optional[torch.Tensor] = None,
                 weights: Optional[Dict[str, float]] = None, cuda: bool = False, relative_optim: bool = False):
        super(Solver, self).__init__()

        self.model = model
        self.loss = loss

        self.N = xf.shape[2]
        self.nchunks = self.model.nchunks
        self.is_cuda = cuda
        self.xf = cplx.from_np(xf)

        self.res = None, None

        if cuda:
            self.cuda()
            self.xf = self.xf.cuda()
            if Rxf is not None:
                Rxf = Rxf.cuda()

        # compute target representation RXf
        self.Rxf = [Rxf]

        # compute weights from target representation
        self.w_gap, self.w_l2 = [None] * self.nchunks, [None] * self.nchunks
        if relative_optim:
            eps = 1e-7
            lRxf = [-self.loss.compute_gap(None, Rxf_chunk, None) for Rxf_chunk in self.Rxf]
            self.w_gap = [(relu(cplx.modulus(lRxf_chunk) - eps) + eps) ** (-1) for lRxf_chunk in lRxf]

            nbs = {m_type: model.count_coefficients(m_type=m_type) for m_type in model.m_types}
            N_total, scat = sum(nbs.values()), len(model.m_types)
            if weights is None:
                weights = {m_type: 1 / (scat * nbs[m_type]) for m_type in model.m_types}
            self.w_l2 = compute_w_l2(weights, model, self.w_gap, self.nchunks)

        # compute initial loss
        self.loss0 = 0.0
        for i_chunk in range(self.nchunks):
            Rx0_chunk = self.model(self.format(x0, requires_grad=False), i_chunk)
            self.loss0 += self.loss(Rx0_chunk, self.Rxf[i_chunk], self.w_gap[i_chunk], self.w_l2[i_chunk]).detach().cpu().numpy()

    def format(self, x: np.ndarray, requires_grad=True):
        """ Transforms x into a compatible format for the embedding. """
        x = cplx.from_np(x.reshape(1, self.N, -1), tensor=torch.DoubleTensor)
        if self.is_cuda:
            x = x.cuda()
        x = Variable(x, requires_grad=requires_grad)
        return x

    def joint(self, x: np.ndarray):
        """ Computes the loss on current vector. """
        # format x and set gradient to 0
        x_torch = self.format(x)

        total_loss = np.array([0.0], dtype=np.float64)
        total_grad_x = np.zeros(x_torch.shape, dtype=np.float64)[0, ..., 0]

        res_max = {m_type: 0.0 for m_type in self.model.module_list[-1].m_types}
        for i_chunk in range(self.nchunks):
            # clear gradient
            if x_torch.grad is not None:
                x_torch.grad.X.zero_()

            # compute moments
            Rxt_chunk = self.model(x_torch, i_chunk)

            # compute loss function
            loss = self.loss(Rxt_chunk, self.Rxf[i_chunk], self.w_gap[i_chunk], self.w_l2[i_chunk])
            res_max = {m_type: max(res_max[m_type], self.loss.max_gap[m_type] if m_type in self.loss.max_gap else 0.0)
                       for m_type in self.model.module_list[-1].m_types}

            # compute gradient
            grad_x, = grad([loss], [x_torch], retain_graph=True)

            # only get the real part
            grad_x = grad_x[0, ..., 0]

            # move to numpy
            grad_x = grad_x.contiguous().detach().cpu().numpy()
            loss = loss.detach().cpu().numpy()

            total_loss += loss
            total_grad_x += grad_x

        self.res = total_loss, total_grad_x.ravel(), res_max

        return total_loss, total_grad_x.ravel()


class SmallEnoughException(Exception):
    pass


class CheckConvCriterion:
    """ A callback function given to the optimizer. """
    def __init__(self, solver: Solver, tol: float,
                 max_wait: Optional[int] = 1000, save_data_evolution_p: Optional[bool] = False):
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

        self.curr_xk = cplx.to_np(solver.xf).real[0, 0, 0, :]

    def __call__(self, xk: np.ndarray):
        err, grad_xk, max_gap = self.solver.res
        gerr = np.max(np.abs(grad_xk))
        err, gerr = float(err), float(gerr)
        self.err = err
        self.max_gap = max_gap
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

    def print_info_line(self, msg: Optional[str] = ''):
        delta_t = time() - self.tic
        print(colored(
            f"{self.counter:6}it in {self.hms_string(delta_t)} ( {self.counter / delta_t:.2f}it/s )"
            + " ........ "
            + f"{np.sqrt(self.err):.2E} -- {max(self.max_gap.values()):.2E} -- {self.gerr:.2E}",
            'cyan'))
        print(colored(
            "".join([f"\n ----- {m_type} {value:.2e}, " for m_type, value in self.max_gap.items()])
            + msg,
            'green'))

    @staticmethod
    def hms_string(sec_elapsed: float):
        """ Format  """
        h = int(sec_elapsed / (60 * 60))
        m = int((sec_elapsed % (60 * 60)) / 60)
        s = sec_elapsed % 60.
        return f"{h}:{m:>02}:{s:>05.2f}"
