""" Manage generation algorithm. """
from typing import Tuple
from termcolor import colored
from time import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable, grad

from scatspectra.description import DescribedTensor
from scatspectra.utils import resolve_device, device_supports_float64


class Solver(nn.Module):
    """ A class that contains all information necessary for generation. """
    def __init__(
        self,
        shape: torch.Size,
        model: nn.Module,
        loss: nn.Module,
        Rx_target: DescribedTensor,
        x0: np.ndarray,
        cuda: bool = False,
        device: str | torch.device | None = None
    ):
        """
        :param cuda: (DEPRECATED, use ``device`` instead) run generation on gpu
        :param device: compute device, accepts None, 'cpu'/'cuda'/'mps'/'auto'
            or a torch.device; takes precedence over ``cuda`` when provided
        """
        super(Solver, self).__init__()

        self.model = model
        self.loss = loss

        self.shape = shape
        self.nchunks = 1
        self.device = resolve_device(device, cuda)
        self.is_cuda = self.device.type == 'cuda'
        # MPS does not support float64. The optimization runs in double
        # precision on cpu/cuda; on mps the time-series tensor is held in
        # float32 (numpy arrays default to float64, so they must be cast
        # before being moved to the device). generate's frontend already
        # falls back to cpu for float64 targets, so float32-on-mps is the
        # only case that reaches this branch.
        self.dtype = (
            torch.float64 if device_supports_float64(self.device)
            else torch.float32
        )
        self.x0 = self.format(x0, requires_grad=False)

        self.result = np.inf, np.inf

        self.Rx_target = Rx_target
        self.to(self.device)
        if self.Rx_target is not None:
            self.Rx_target = self.Rx_target.to(self.device)

        # compute initial loss
        Rx0 = self.model(self.x0).mean_batch()
        self.loss0 = self.loss(Rx0, self.Rx_target).detach().cpu().numpy()
        Rnull = self.Rx_target.clone()
        Rnull.y.fill_(0)
        self.loss_norm = self.loss(Rnull, self.Rx_target)

    def format(self, x: np.ndarray, requires_grad: bool = True) -> Variable:
        """ Transforms x into a compatible format for the embedding. """
        x_torch = torch.tensor(x.reshape(self.shape))
        # cast to the device-compatible precision before the device move
        # (numpy float64 cannot be moved to an mps device)
        if x_torch.dtype == torch.float64 and self.dtype != torch.float64:
            x_torch = x_torch.to(self.dtype)
        x_torch = x_torch.to(self.device)
        x_torch = Variable(x_torch, requires_grad=requires_grad)
        return x_torch

    def joint(self, x: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Computes the loss on current vector. """

        # format x and set gradient to 0
        x_torch = self.format(x)

        # clear gradient
        if x_torch.grad is not None:
            x_torch.grad.data.zero_()

        # compute moments
        Rx = self.model(x_torch).mean_batch()

        # compute loss function
        loss = self.loss(Rx, self.Rx_target, None, None) / self.loss_norm

        # compute gradient
        grad_x, = grad([loss], [x_torch], retain_graph=True)

        # move to numpy; scipy's L-BFGS-B requires float64 for the value and
        # gradient, so up-cast (a no-op on cpu/cuda which already run float64,
        # needed on mps where the computation runs in float32)
        grad_x = grad_x.contiguous().detach().cpu().numpy().astype(np.float64)
        loss = loss.detach().cpu().numpy().astype(np.float64)

        self.result = loss, grad_x.ravel()

        return loss, grad_x.ravel()


class SmallEnoughException(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class MaxIteration(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class CheckConvCriterion:
    """ A callback function given to the optimizer. """
    def __init__(
        self,
        solver: Solver,
        tol: float,
        max_wait: int = 1000,
        save_interval_data: int | None = None,
        verbose: bool = True
    ):
        self.solver = solver
        self.tol = tol  # stops when |Rx-Rx_target| / |Rx_target|   <  tol
        self.result = None
        self.next_milestone = None
        self.counter = 0
        self.err = np.inf
        self.max_gap = None
        self.gerr = None
        self.tic = time()

        self.verbose = verbose
        self.max_wait, self.wait = max_wait, 0
        self.save_interval_data = save_interval_data

        self.logs_loss = []
        self.logs_grad = []
        self.logs_x = []

    def __call__(self, xk: np.ndarray) -> None:
        err, grad_xk = self.solver.result

        gerr = np.max(np.abs(grad_xk))
        err, gerr = float(err), float(gerr)
        self.err = err
        self.gerr = gerr
        self.counter += 1

        self.logs_loss.append(err)
        self.logs_grad.append(gerr)

        if self.next_milestone is None:
            self.next_milestone = 10 ** (np.floor(np.log10(gerr)))

        info_already_printed_p = False
        if self.save_interval_data is not None and self.counter % self.save_interval_data == 0:
            self.logs_x.append(xk)

        if np.sqrt(err) <= self.tol:
            self.result = xk
            raise SmallEnoughException("Small enough error.")
        elif gerr <= self.next_milestone or self.wait >= self.max_wait:
            if not info_already_printed_p:
                self.print_info_line()
            if gerr <= self.next_milestone:
                self.next_milestone /= 10
            self.wait = 0
        else:
            self.wait += 1

    def print_info_line(self) -> None:
        delta_t = time() - self.tic

        if self.verbose:
            print(colored(
                f"{self.counter:6}it in {self.hms_string(delta_t)} "
                + f"( {self.counter / delta_t:.2f} it/s )"
                + " .... "
                + f"err {np.sqrt(self.err):.2E}",
                'cyan'
            ))

    @staticmethod
    def hms_string(sec_elapsed: float) -> str:
        """ Format  """
        h = int(sec_elapsed / (60 * 60))
        m = int((sec_elapsed % (60 * 60)) / 60)
        s = sec_elapsed % 60.
        return f"{h}:{m:>02}:{s:>05.2f}"
