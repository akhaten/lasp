import pandas
import pathlib
import matplotlib.pyplot
import sys
sys.path.append('../../..')

import lasp.io

import lasp.metrics

def compute_metrics(
    df_to_process: pandas.DataFrame, 
    index: int
) -> None:

    imgs_gen_path = pathlib.Path(df_to_process.attrs['imgs_gen_path'])
    output_path = pathlib.Path(df_to_process.attrs['output_path'])
    params = df_to_process.loc[index]
    # print(params.keys())

    df_imgs_index = params['index']

    original = lasp.io.read(
        imgs_gen_path / str(df_imgs_index) / 'original.npy'
    )

    # print(imgs_gen_path / str(df_imgs_index) / 'original.npy')
    # print(output_path / str(index) / 'output.npy')
    output = lasp.io.read(
        output_path / str(index) / 'output.npy'
    )
    

    return lasp.metrics.PSNR(original, output, intensity_max=255)

def title_from(params: pandas.Series) -> str:

    alpha = params['alpha']
    beta0 = params['beta0']
    beta1 = params['beta1']
    sigma = params['sigma']
    d = params['decimation']
    tol = params['tol']
    iterations = params['iterations']
    blur = params['blur']
    deblur = params['deblur_kernel']
    noise = params['noise']

    params_algo_str = 'Params : ($\\alpha$={}, $\\beta_0$={}, $\\beta_1$={}, $\sigma$={}, $d$={}, tol={}, nb_iters={})\n'.format(
        alpha, beta0, beta1, sigma, d, tol, iterations
    )

    params_blur_str = ''
    if not(pandas.isna(blur)):
        params_blur_str = 'Blur filter : {}x{}, $\sigma$={}\n'.format(blur[0], blur[0], blur[1])

    params_noise_str = ''
    if not(pandas.isna(noise)):
        params_noise_str = 'Noise : {} (not dB)\n'.format(noise)
    
    params_deblur_str = 'Deblur filter : {}x{}, $\sigma$={}\n'.format(deblur[0], deblur[0], deblur[1])

    title = params_algo_str \
        + params_blur_str \
        + params_noise_str \
        + params_deblur_str    
    return title
    

def plot1x3(dataset_params: pandas.DataFrame, index: int, dataset_path: pathlib.Path) -> None:
    
    
    CURRENT = dataset_path / str(index)
    params = dataset_params.loc[index]
    
    original = lasp.io.read(params['image'])
    input = lasp.io.read(CURRENT / 'input.png')
    output = lasp.io.read(CURRENT / 'output.png')

    figure = matplotlib.pyplot.figure(
        figsize=(10, 10)
    )
    
    figure.subplots_adjust(top=1.4)
    
    title = title_from(params)

    psnr_str = str(lasp.metrics.PSNR(original, output, intensity_max=255))
    title += psnr_str
    
    figure.suptitle('Mumford-Shah \n'+title)
    # figure.tight_layout()
    
    # plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)

    ax = figure.subplots(1, 3)
    
    ax[0].axis('off')
    ax[0].set_title('Original')
    ax[0].imshow(original, cmap='gray')


    ax[1].axis('off')
    ax[1].set_title('Img')
    ax[1].imshow(input, cmap='gray')

    ax[2].axis('off')
    ax[2].set_title('Result')
    ax[2].imshow(output, cmap='gray')

    figure.savefig(CURRENT / ('figure1x3.png'))


def plot_inputs1x2(dataset: pandas.DataFrame, index: int, dataset_path: pathlib.Path) -> None:

    CURRENT = dataset_path / str(index)
    params = dataset_params.loc[index]
    
    original = lasp.io.read(params['image'])
    input = lasp.io.read(CURRENT / 'input.png')
    # output = lasp.io.read(CURRENT / 'output.png')

    params = dataset.loc[index]
    

    figure = matplotlib.pyplot.figure(
        figsize=(10, 10)
    )
    
    figure.subplots_adjust(top=1.4)
    
    # figure.tight_layout()
    
    # plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)

    ax = figure.subplots(1, 2)

    
    ax[0].axis('off')
    ax[0].set_title('Original')
    ax[0].imshow(original, cmap='gray')


    ax[1].axis('off')
    ax[1].set_title('Img')
    ax[1].imshow(input, cmap='gray')