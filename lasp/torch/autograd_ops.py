import torch
import torch.autograd

import lasp.torch.differential

class GradX2DCirc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor: torch.Tensor) -> torch.Tensor:
        return lasp.torch.differential.dx2D_circ(tensor, detach=True)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return lasp.torch.differential.dxT2D_circ(grad_output, detach=False)
    
class GradY2DCirc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor: torch.Tensor) -> torch.Tensor:
        return lasp.torch.differential.dy2D_circ(tensor, detach=True)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return lasp.torch.differential.dyT2D_circ(grad_output, detach=False)
    
class TransposedGradX2DCirc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor: torch.Tensor) -> torch.Tensor:
        return lasp.torch.differential.dxT2D_circ(tensor, detach=True)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return lasp.torch.differential.dx2D_circ(grad_output, detach=False)
    
class TransposedGradY2DCirc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor: torch.Tensor) -> torch.Tensor:
        return lasp.torch.differential.dyT2D_circ(tensor, detach=True)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return lasp.torch.differential.dy2D_circ(grad_output, detach=False)
    
    
class Lap2DCirc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor: torch.Tensor) -> torch.Tensor:
        return lasp.torch.differential.laplacian2D_circ(tensor, detach=True)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return lasp.torch.differential.laplacian2D_circ(grad_output, detach=False)