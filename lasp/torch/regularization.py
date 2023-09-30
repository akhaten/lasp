from typing import Any
import torch

import lasp.torch.differential

def tv_anisotropic(tensor: torch.Tensor, detach: bool = True) -> torch.Tensor:
    """
    Params:
        - tensor : Size([1, N, M])
    """
    dx = lasp.torch.differential.dx2D_circ(tensor, detach)
    dy = lasp.torch.differential.dy2D_circ(tensor, detach)
    return torch.sum( torch.abs(dx) + torch.abs(dy) )


def tv_isotropic(tensor: torch.Tensor, detach: bool = True) -> torch.Tensor:
    """
    Params:
        - tensor : Size([1, N, M])
    """
    dx = lasp.torch.differential.dx2D_circ(tensor, detach)
    dy = lasp.torch.differential.dy2D_circ(tensor, detach)
    return torch.sum(torch.sqrt( dx**2 + dy**2 ))


def dirichlet_energy(tensor: torch.Tensor, detach: bool = True) -> torch.Tensor:
    """
    Params:
        - tensor : Size([1, N, M])
    """
    dx = lasp.torch.differential.dx2D_circ(tensor, detach)
    dy = lasp.torch.differential.dy2D_circ(tensor, detach)
    return (1/2) * torch.sum(dx**2 + dy**2)

      
