import pathlib
import numpy

class Parameters:

    def __init__(self):
        self.image: pathlib.Path = None
        self.alpha: float = None
        self.beta: float = None
        self.sigma: float = None
        self.tol: float = -1
        self.iterations: int = None
        self.blur: tuple[int, float] = numpy.nan
        self.noise: tuple[float] = numpy.nan

    def to_dict(self) -> dict:

        if(self.image is None):
            raise AssertionError('image is None')

        if(self.alpha is None):
            raise AssertionError('alpha is None')

        if(self.beta is None):
            raise AssertionError('beta is None')

        if(self.sigma is None):
            raise AssertionError('sigma is None')

        if(self.iterations is None):
            raise AssertionError('iterations is None')

        return {
            'image' : self.image,
            'alpha' : self.alpha,
            'beta' : self.beta,
            'sigma' : self.sigma,
            'tol' : self.tol,
            'iterations' : self.iterations,
            'blur' : self.blur,
            'noise' : self.noise
        }


import lasp.io
import lasp.filters.linear
import lasp.noise
import scipy.signal

def make_image(params: Parameters) -> numpy.ndarray:
    
    img = lasp.io.read(params.image)
    blur: tuple[int, float] = params.blur
    snr: float = params.noise

    if blur != numpy.nan:
       kernel = lasp.filters.linear.gaussian_filter(size=blur[0], sigma=blur[1])
       img =  scipy.signal.convolve2d(img, kernel, mode='same')

    if snr != numpy.nan:
        img = lasp.noise.awgn(img, snr)

    return img


import pandas

def add(dataset: pandas.DataFrame, params: Parameters) -> pandas.DataFrame:
    return pandas.concat(
        [ dataset, pandas.DataFrame(params.to_dict()) ], 
        ignore_index=True
    )
    
def process_dataset(dataset: pandas.DataFrame, output: pathlib.Path) -> None:
    
    for i in range(0, dataset.index):

        
        dataset.iloc[i]['image']







