import torch

import lasp.torch.autograd_ops
import lasp.torch.differential
import lasp.torch.utils


def dx(tensor: torch.Tensor) -> torch.Tensor:

    if tensor.requires_grad:
        grad_x = lasp.torch.autograd_ops.GradX2DCirc.apply(
            tensor
        )
    else:
        grad_x = lasp.torch.differential.dx2D_circ(
            tensor,
            detach = True
        )

    return grad_x

def dxT(tensor: torch.Tensor) -> torch.Tensor:

    if tensor.requires_grad:
        gradT_x = lasp.torch.autograd_ops.TransposedGradX2DCirc.apply(
            tensor
        )
    else:
        gradT_x = lasp.torch.differential.dxT2D_circ(
            tensor,
            detach = True
        )

    return gradT_x

def dy(tensor: torch.Tensor) -> torch.Tensor:

    if tensor.requires_grad:
        grad_y = lasp.torch.autograd_ops.GradY2DCirc.apply(
            tensor
        )
    else:
        grad_y = lasp.torch.differential.dy2D_circ(
            tensor,
            detach = True
        )
        
    return grad_y

def dyT(tensor: torch.Tensor) -> torch.Tensor:

    if tensor.requires_grad:
        gradT_y = lasp.torch.autograd_ops.TransposedGradY2DCirc.apply(
            tensor
        )
    else:
        gradT_y = lasp.torch.differential.dyT2D_circ(
            tensor,
            detach = True
        )
        
    return gradT_y

def multidimensional_soft(d: torch.Tensor, epsilon: float, gamma_zero: float=1e-12):
    """ Thresholding soft for multidimensional array
    Use generalization of sign function
    
    Params:
        - d : multidimensional array
        - epsilon : threshold
        - gamma_zero : for zero value (prevent "Error detected in DivBackward0")

    Return:
        Array thresholded with dimesion equal to d
    """
    s = torch.sqrt(torch.sum(d**2, axis=0)+gamma_zero)
    ss = torch.where(s > epsilon, (s-epsilon)/s, 0)
    output = torch.concat([(ss*d[i]).unsqueeze(0) for i in range(0, d.size()[0])], 0)
    return output