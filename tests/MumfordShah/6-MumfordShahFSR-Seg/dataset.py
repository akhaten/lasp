import sys
sys.path.append('../../..')

import pandas
import pathlib
import scipy.signal

import lasp.io
import lasp.filters.linear
import lasp.utils
import lasp.noise


def add_image(
    df_imgs: pandas.DataFrame,
    original: pathlib.Path, 
    blur: tuple[int, int] = (0, 0), 
    decimation: int = 1, 
    noise: float = -1
) -> pandas.DataFrame :
    to_add = pandas.DataFrame(
        { 
            'original': [original], 
            'blur' : [blur], 
            'decimation': [decimation], 
            'noise': [noise]
        }
    )
    return pandas.concat([df_imgs,  to_add], ignore_index=True)

def make_images(df_imgs: pandas.DataFrame, save_path: pathlib.Path) -> None:


    for index in df_imgs.index:

        CURRENT = save_path / str(index)
        if not(CURRENT.exists()):
            CURRENT.mkdir()

        
        img_datas = df_imgs.loc[index]

        img_path = img_datas['original']
        blur = img_datas['blur']
        decim = img_datas['decimation']
        noise = img_datas['noise']


        out = lasp.io.read(img_path) # x
        lasp.io.save(out, CURRENT / 'original.npy')
        lasp.io.save(out, CURRENT / 'original.png')
        

        if (blur[0] > 0) and (blur[1] > 0):
            kernel = lasp.filters.linear.gaussian_filter(size=blur[0], sigma=blur[1]) # Hx
            out = scipy.signal.convolve2d(out, kernel, mode='same')
            lasp.io.save(out, CURRENT / 'blurred.npy')
            lasp.io.save(out, CURRENT / 'blurred.png')
            

        if decim > 0:
            out = lasp.utils.decimation(out, decim) # SHx
            lasp.io.save(out, CURRENT / 'decimed.npy')
            lasp.io.save(out, CURRENT / 'decimed.png')

        
        if noise >= 0:
            out = lasp.noise.awgn(out, noise) # SHx + n
            lasp.io.save(out, CURRENT / 'noised.npy')
            lasp.io.save(out, CURRENT / 'noised.png')


        normalized = lasp.utils.normalize(out)
        lasp.io.save(normalized, CURRENT / 'input_normalized.npy')
        lasp.io.save(normalized, CURRENT / 'input_normalized.png')

        

def add_params(
    df_params: pandas.DataFrame,
    df_imgs_index: int,
    deblur_kernel: tuple[int, int],
    alpha: float,
    beta0: float,
    beta1: float,
    sigma: float,
    iterations: int,
    tol: float 
) -> pandas.DataFrame:

    to_add = pandas.DataFrame(
        {
            'df_imgs_index': [df_imgs_index],
            'deblur_kernel': [deblur_kernel],
            'alpha': [alpha],
            'beta0': [beta0],
            'beta1': [beta1],
            'sigma': [sigma],
            'iterations': [iterations],
            'tol': [tol]
        }
    )
    
    return pandas.concat([df_params, to_add], ignore_index=True)