import numpy
import torch


def gaussian_filter(size: int, sigma: float, normalize: bool = False) -> torch.Tensor:
    """Create square gaussian filter

    Params:
        - size
        - sigma
        - normalize

    Return:
        - Gaussian filter
    """

    def gaussian2d_psf(sigma: float, x: float | torch.Tensor, y: float | torch.Tensor) -> float | torch.Tensor:
        exp = torch.exp( - (x**2+y**2) / (2*sigma**2) )
        return ( 1 / (2*torch.pi*sigma**2) ) * exp

    x, y = torch.meshgrid(
        *[
            torch.tensor([ i for i in range(-size//2 + 1, size//2 + 1) ]),
            torch.tensor([ i for i in range(-size//2 + 1, size//2 + 1) ])
        ]
    )
    
    filter = gaussian2d_psf(sigma, x, y)
    
    if normalize:
        filter /= torch.sum(filter)

    return filter

def north() -> torch.Tensor:
    return torch.tensor(
        [
            [0, 1, 0], 
            [0, -1, 0], 
            [0, 0, 0]
        ]
    )

def south() -> torch.Tensor:
    return torch.tensor(
        [
            [0, 0, 0], 
            [0, -1, 0], 
            [0, 1, 0]
        ]
    )

def west() -> torch.Tensor:
    return torch.tensor(
        [
            [0, 0, 0], 
            [1, -1, 0], 
            [0, 0, 0]
        ]
    )

def est() -> torch.Tensor:
    return torch.tensor(
        [
            [0, 0, 0], 
            [0, -1, 1], 
            [0, 0, 0]
        ]
    )

def laplacian() -> torch.Tensor:
    return torch.tensor(
        [
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ]
    )


def mean_filter(size: int) -> torch.Tensor:
    """Create mean filter

    Params:
        - size

    Return:
        - Mean filter
    """
    filter = torch.ones(size=(size, size))
    filter /= size*size
    return filter


def roberts_masks() -> torch.Tensor:
    #TODO : TEST
 
    return torch.tensor(
        [
            torch.tensor([[-1, 0], [0, 1]]),
            torch.tensor([[0, -1], [1, 0]])
        ]
    )


def sobel_masks() -> torch.Tensor:
    #TODO : TEST
 
    return torch.tensor(
        [
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        ]
    )


def kirsh_masks() -> torch.Tensor:
    #TODO
    # return torch.Tensor(
    #     [
    #         torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    #         torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    #     ]
    # )
    pass


def robinson_masks() -> torch.Tensor:
    #TODO
    # return torch.Tensor(
    #     [
    #         torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    #         torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    #     ]
    # )
    pass
