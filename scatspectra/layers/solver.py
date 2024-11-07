""" Manage generation algorithm. """
from typing import Tuple
from termcolor import colored
from time import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable, grad

from scatspectra.description import DescribedTensor


class Solver(nn.Module):
    """ A class that contains all information necessary for generation. """
    def __init__(
        self,
        shape: torch.Size,
        model: nn.Module,
        loss: nn.Module,
        Rx_target: DescribedTensor,
        x0: np.ndarray,
        cuda: bool
    ):
        super(Solver, self).__init__()

        self.model = model
        self.loss = loss

        self.shape = shape
        self.nchunks = 1
        self.is_cuda = cuda
        self.x0 = self.format(x0, requires_grad=False)

        self.result = np.inf, np.inf

        self.Rx_target = Rx_target
        if cuda:
            self.cuda()
            if self.Rx_target is not None:
                self.Rx_target = self.Rx_target.cuda()

        # compute initial loss
        Rx0 = self.model(self.x0).mean_batch()
        self.loss0 = self.loss(Rx0, self.Rx_target).detach().cpu().numpy()
        Rnull = self.Rx_target.clone()
        Rnull.y.fill_(0)
        self.loss_norm = self.loss(Rnull, self.Rx_target)

    def format(self, x: np.ndarray, requires_grad: bool = True) -> Variable:
        """ Transforms x into a compatible format for the embedding. """
        x_torch = torch.tensor(x.reshape(self.shape))
        if self.is_cuda:
            x_torch = x_torch.cuda()
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

        # move to numpy
        grad_x = grad_x.contiguous().detach().cpu().numpy()
        loss = loss.detach().cpu().numpy()

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
