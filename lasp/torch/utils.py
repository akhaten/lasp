from typing import Any
import torch


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


class Decimation2D:

    def __init__(self, factor_rows: int, factor_cols: int) -> None:
        self.factor_rows = factor_rows
        self.factor_cols = factor_cols

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return decimation2D(tensor, self.factor_rows, self.factor_cols)
    
    def T(self, tensor: torch.Tensor) -> torch.Tensor:
        return decimation2D_adjoint(tensor, self.factor_rows, self.factor_cols)