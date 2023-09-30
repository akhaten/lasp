import torch

import lasp.torch.autograd_ops
import lasp.torch.differential
import lasp.torch.utils

from utils import \
    dx, dy, dxT, dyT, multidimensional_soft


class CustomConv2D(torch.nn.Module):

    def __init__(self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: tuple[int, int],
        padding: str ='same', 
        bias: bool = False
    ) -> None:
        super(CustomConv2D, self).__init__()
        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding = padding,
            bias = bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
 
class Cell(torch.nn.Module):

    def __init__(self,
        alpha: torch.Tensor,
        beta1: torch.Tensor,
        sigma: torch.Tensor,
        kernel_size: tuple[int, int],
        channels_intermediate: int
    ):
        
        super(Cell, self).__init__()

        self.alpha = alpha
        self.beta1 = beta1
        self.sigma = sigma

        self.HT = CustomConv2D(1, channels_intermediate, kernel_size)
        self.inv = CustomConv2D(channels_intermediate, 1, kernel_size)

    def forward(self, 
        STg: torch.Tensor, # (1, N, M)
        d_x: torch.Tensor, # (1, N, M)
        d_y: torch.Tensor, # (1, N, M)
        b_x: torch.Tensor, # (1, N, M)
        b_y: torch.Tensor  # (1, N, M)
    ) -> torch.Tensor:
        
        # print(d_x.size())

        gradT_x = dxT(d_x - b_x) # (1, N, M)
        gradT_y = dyT(d_y - b_y) # (1, N, M)
        sigma_expr = self.sigma * ( gradT_x + gradT_y ) # (1, N, M)
        alpha_expr = self.alpha * self.HT(STg) # (channels_intermediate, N, M)

        f = self.inv(sigma_expr + alpha_expr) # (1, N, M)

        dx_f = dx(f) # (1, N, M)
        dy_f = dy(f) # (1, N, M)
     
        d_x, d_y = multidimensional_soft(
            torch.concat([ (dx_f + b_x), (dy_f + b_y) ], dim = 0),
            self.beta1 / self.sigma
        ) # (1, N, M), (1, N, M)
      
        b_x += (dx_f - d_x) # (1, N, M)
        b_y += (dy_f - d_y) # (1, N, M)

        return [ f, d_x, d_y, b_x, b_y ]



class Unfolding(torch.nn.Module):

    def __init__(self, 
        alpha: tuple[torch.float, bool],
        beta1: tuple[torch.float, bool],
        sigma: tuple[torch.float, bool],
        nb_iterations: int,
        kernel_size: tuple[int, int],
        channels_intermediate: int
    ) -> None:
        
        super(Unfolding, self).__init__()

        self.nb_iterations = nb_iterations

        self.alphas = torch.nn.Parameter(
            data = torch.fill(
                input = torch.zeros(
                    size=(self.nb_iterations,), 
                    dtype=torch.float
                ), 
                value=alpha[0]
            ),
            requires_grad=alpha[1]
        )

        self.beta1s = torch.nn.Parameter(
            data = torch.fill(
                input = torch.zeros(
                    size=(self.nb_iterations,), 
                    dtype=torch.float
                ), 
                value=beta1[0]
            ),
            requires_grad=beta1[1]
        )

        self.sigmas = torch.nn.Parameter(
            data = torch.fill(
                input = torch.zeros(
                    size=(self.nb_iterations,), 
                    dtype=torch.float
                ), 
                value=sigma[0]
            ),
            requires_grad=sigma[1]
        )

        self.cells = torch.nn.ModuleList(
            [
                Cell(
                    alpha = self.alphas[i],
                    beta1 = self.beta1s[i],
                    sigma = self.sigmas[i],
                    kernel_size = kernel_size,
                    channels_intermediate = channels_intermediate
                )
                for i in range(0, self.nb_iterations)
            ]
        )
            

    def forward(self, low_resolution: torch.Tensor, decim_row: int, decim_col: int) -> torch.Tensor:

        """
        
            Params:
                - low_resolution : image low-resolution; Size([1, N, M])
                - decim_row : decimation on line
                - decim_col : decimation on col

            Return:
                Image high-resolution of size.
                If size of low_resolution is (N, M), Image high-resolution
                will be (N*decim_row, M*decim_col)

        """


        # Initialize static attribute / shared attribute

        g = low_resolution
        S = lasp.torch.utils.Decimation2D(decim_row, decim_col)
       

        STg = S.T(g)

        d_x = torch.zeros_like(STg)
        d_y = torch.zeros_like(STg)
        b_x = torch.zeros_like(STg)
        b_y = torch.zeros_like(STg)

        for cell in self.cells:
            # STg, d_x, d_y, b_x, b_y = iter_layer(STg, d_x, d_y, b_x, b_y)
            f, d_x, d_y, b_x, b_y = cell(STg, d_x, d_y, b_x, b_y)

        # f_approx = Iteration.f
        f_approx = f
        # Normalize f
        mini = torch.min(f_approx)
        maxi = torch.max(f_approx)
        normalized = (f_approx - mini) / (maxi - mini)

        return normalized
    
    def from_config(config: dict) -> 'Unfolding':

        return Unfolding(
            alpha = (
                config['model']['alpha']['initialize'],
                config['model']['alpha']['is_trainable']
            ),
            beta1 = (
                config['model']['beta1']['initialize'],
                config['model']['beta1']['is_trainable']
            ),
            sigma = (
                config['model']['sigma']['initialize'],
                config['model']['sigma']['is_trainable']
            ),
            nb_iterations = config['model']['nb_iterations'],
            kernel_size = config['model']['kernel_size'],
            channels_intermediate = config['model']['channel_intermediate']
        )