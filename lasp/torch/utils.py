from typing import Any

import torch
import torch.fft


def decimation2D(tensor: torch.Tensor, decim_row: int, decim_col: int) -> torch.Tensor:
    """Apply decimation on a batch
    Params:
        - tensor : Size([batch_length, N, M])
    """
    return tensor[:, ::decim_row, ::decim_col].clone()

def decimation2D_adjoint(tensor: torch.Tensor, decim_row: int, decim_col: int) -> torch.Tensor:
    batch_length, nb_row, nb_col = tensor.size()
    out = torch.zeros(
        size = (batch_length, decim_row*nb_row, decim_col*nb_col),
        device = tensor.device,
        requires_grad=tensor.requires_grad
    )
    out[:, ::decim_row, ::decim_col] = tensor.clone()
    return out


# class Decimation2D:

#     def __init__(self, factor_rows: int, factor_cols: int) -> None:
#         self.factor_rows = factor_rows
#         self.factor_cols = factor_cols

#     def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
#         return decimation2D(tensor, self.factor_rows, self.factor_cols)
    
#     def T(self, tensor: torch.Tensor) -> torch.Tensor:
#         return decimation2D_adjoint(tensor, self.factor_rows, self.factor_cols)


def pad(tensor: torch.Tensor, shape_out: tuple) -> torch.Tensor:
    
    nb_rows, nb_cols = tensor.size()

    padded = torch.zeros(size = shape_out)
    padded[:nb_rows, :nb_cols] = tensor.clone()
       
    return padded

          
def circshift(tensor: torch.Tensor, shift: tuple) -> torch.Tensor:
    """Circular Shift
    Similary to ocatave/matlab function.

    Params:
        - tensor : tensor
        - shift : shift 

    Returns:
        - Circulary shifted matrix
    """
    return torch.roll(tensor, shift, [0, 1])


def compute_center(tensor: torch.Tensor) -> torch.Size:
    center = tensor.size() // 2
    return center

def pad_circshift_center(tensor: torch.Tensor, shape_out: torch.Tensor | tuple) -> torch.Tensor:
    padded = pad(tensor, shape_out)
    center = compute_center(tensor)
    circshifted = circshift(padded, -center)
    return circshifted


def fourier_diagonalization(kernel: torch.Tensor, shape_out: torch.Tensor | tuple) -> torch.Tensor:
    """Diagonalize input in Fourier space

    Params:
        - kernel: filter/kernel for diagonalization
        - shape_out: dimension of output

    Returns:
        Diagonalisation in Fourier space of kernel with dimension shape out
    """
    return torch.fft.fftn(pad_circshift_center(kernel, shape_out))