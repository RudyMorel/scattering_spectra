from itertools import product
import numpy as np
import matplotlib.pyplot as plt

from global_const import *
import utils.complex_utils as cplx
from stochastic_classical_models import mrw
from scattering_network import Scattering, Cov, CovStat, ModuleChunk


def init_model(N, T, J, Q, r_max, A, wav_type, high_freq, rm_high, wav_norm, normalize,
               moments, m_types, chunk_method, nchunks):
    module_list = [Scattering(J, Q, r_max, T, A, N, wav_type, high_freq, rm_high, wav_norm, normalize)]

    if moments in ['cov', 'covstat']:
        module_list.append(Cov(J, Q, r_max, N, m_types))
    if moments == 'covstat':
        module_list.append(CovStat(J*Q, m_types))

    return ModuleChunk(module_list, chunk_method, nchunks)


def main():

    # perform analysis of a MRW
    # data
    R, T = 32, 2 ** 15  # 128, 2 **15 to get the results in the paper
    lam = 0.1

    X = mrw(R=R, T=T, L=T, H=0.5, lam=lam)
    X = cplx.from_np(X.reshape(1, R, -1))

    # scattering model
    moments = 'cov'  # either cov or covstat
    J, Q, r = 10, 1, 2
    wav_type, high_freq = 'battle_lemarie', 0.425 / 2
    model = init_model(R, T, J, Q, r, None, wav_type, high_freq, False, 'l1', True,
                       moments, ['m00', 'm10', 'm11'], 'quotient_n', 1)
    model.init_chunks()

    # compute scattering covariance representation
    RX = model(X)

    # plotting
    # scattering cross-spectrum
    C_mW = torch.zeros(J-1, J-1, 2)
    for (a, b) in product(range(J-1), range(-J+1, 0)):
        if a - b >= J:
            continue

        C_mW_j = torch.stack([RX.select(pivot='n1', m_type='m11', j1=j, jp1=j-a, j2=j-b, lp=False)[:, 0, :].mean(0)
                             for j in range(a, J+b)])
        C_mW[a, J-1+b] = C_mW_j.mean(0)  # average on j

    C_mW = cplx.to_np(C_mW)
    C_mod = np.abs(C_mW)
    C_arg = np.angle(C_mW)

    bs = np.arange(-J + 1, 0)[None, :]
    C_mod /= (2.0 ** bs)

    fig, axes = plt.subplots(2, 1)
    for a in range(J-1):
        bs = np.arange(-J+a+1, 0)
        axes[0].plot(bs, C_mod[a, a:J-1])
        # axes[0].plot(bs, C_mod[a, a:J-1], marker='+')

    for a in range(J-1):
        bs = np.arange(-J+a+1, 0)
        axes[1].plot(bs, C_arg[a, a:J-1])
        # axes[1].plot(bs, C_arg[a, a:J-1], marker='+')
    plt.yticks(np.arange(-2, 3) * np.pi / 8,
               [r'$-\frac{\pi}{4}$', r'$-\frac{\pi}{8}$', r'$0$', r'$\frac{\pi}{8}$', r'$\frac{\pi}{4}$'])

    plt.show()


if __name__ == "__main__":
    main()
