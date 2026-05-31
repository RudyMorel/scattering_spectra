""" Device resolution helpers supporting CPU, CUDA and Apple MPS backends. """
import torch


def resolve_device(
    device: str | torch.device | None = None,
    cuda: bool = False
) -> torch.device:
    """ Resolve a torch.device from a new-style ``device`` argument and the
    deprecated boolean ``cuda`` alias.

    Resolution rules:
    - If ``device`` is not None:
        - 'auto': pick the best available backend, i.e. cuda if available,
          else mps if available, else cpu.
        - otherwise use the given device (str or torch.device).
    - elif ``cuda`` is True: use 'cuda' (preserves the old behavior).
    - else: use 'cpu'.

    A clear ValueError is raised if the user explicitly requests a backend
    ('cuda' or 'mps') that is not available on this machine.

    :param device: None, 'cpu'|'cuda'|'mps'|'auto', or a torch.device
    :param cuda: deprecated boolean alias, equivalent to device='cuda'
    :return: a torch.device
    """
    if device is not None:
        if isinstance(device, str) and device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            if torch.backends.mps.is_available():
                return torch.device('mps')
            return torch.device('cpu')
        device = torch.device(device)
    elif cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # validate availability for explicitly requested accelerators
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError(
            "CUDA device requested but torch.cuda.is_available() is False. "
            "Use device='cpu', device='mps' (Apple Silicon) or device='auto'."
        )
    if device.type == 'mps' and not torch.backends.mps.is_available():
        raise ValueError(
            "MPS device requested but torch.backends.mps.is_available() is "
            "False. MPS requires an Apple Silicon Mac and a recent PyTorch "
            "build. Use device='cpu' or device='auto'."
        )

    return device


def device_supports_float64(device: str | torch.device | None) -> bool:
    """ Whether a device supports float64/complex128.

    Apple's MPS backend does NOT support double precision (float64 /
    complex128). All other backends (cpu, cuda) do.

    :param device: a torch.device or a value accepted by torch.device(...)
    :return: False for an 'mps' device, True otherwise
    """
    if device is None:
        return True
    if not isinstance(device, torch.device):
        device = torch.device(device)
    return device.type != 'mps'
