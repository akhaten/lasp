import torch
import torch.autograd


def dx2D_circ(tensor: torch.Tensor, detach: bool = True) -> torch.Tensor:
    """Derivation by column with circular boundary condition

    Params:
        - tensor : Size([1, N, M])
    
    Return:
        - first element of gradient
    """
    _, _, nb_cols = tensor.size()
    tensor_derivated = tensor.detach().clone() if detach else tensor.clone()
    tensor_derivated[:, :, 1:nb_cols] -= tensor[:, :, 0:nb_cols-1]
    tensor_derivated[:, :, 0] -= tensor[:, :, nb_cols-1]
    return tensor_derivated


def dxT2D_circ(tensor: torch.Tensor, detach: bool = True) -> torch.Tensor:
    """Transposed Derivation by column with circular boundary condition

    Params:
        - tensor : Size([1, N, M])
    
    Return:
        - first element of gradient transposed
    """
    _, _, nb_cols = tensor.size()
    tensor_derivated = tensor.detach().clone() if detach else tensor.clone()
    tensor_derivated[:, :, 0:nb_cols-1] -= tensor[:, :, 1:nb_cols]
    tensor_derivated[:, :, nb_cols-1] -= tensor[:, :, 0]
    return tensor_derivated


def dy2D_circ(tensor: torch.Tensor, detach: bool = True) -> torch.Tensor:
    """Derivation by line with circular boundary condition

    Params:
        - image
    
    Return:
        - second element of gradient
    """
    _, nb_rows, _ = tensor.size()
    tensor_derivated = tensor.detach().clone() if detach else tensor.clone()
    tensor_derivated[:, 1:nb_rows, :] -= tensor[:, 0:nb_rows-1, :]
    tensor_derivated[:, 0, :] -= tensor[:, nb_rows-1, :]
    return tensor_derivated

def dyT2D_circ(tensor: torch.Tensor, detach: bool = True) -> torch.Tensor:
    """Transposed Derivation by line with circular boundary condition

    Params:
        - image
    
    Return:
        - second element of gradient transposed
    """
    _, nb_rows, _ = tensor.size()
    tensor_derivated = tensor.detach().clone() if detach else tensor.clone()
    tensor_derivated[:, 0:nb_rows-1, :] -= tensor[:, 1:nb_rows, :]
    tensor_derivated[:, nb_rows-1, :] -= tensor[:, 0, :]
    return tensor_derivated


def laplacian2D_circ(tensor: torch.Tensor, detach: bool = True) -> torch.Tensor:
    """Laplacian with circular boundary condition on gradient
    """
    d_dx = dx2D_circ(tensor, detach)
    d_dy = dy2D_circ(tensor, detach)
    d2_d2x = dxT2D_circ(d_dx, detach)
    d2_d2y = dyT2D_circ(d_dy, detach)
    lap = d2_d2x + d2_d2y
    return lap

    


    



        