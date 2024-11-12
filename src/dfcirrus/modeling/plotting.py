import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from astropy.stats import sigma_clip

from skimage.color import rgb2gray
from skimage import img_as_float

from ..io import logger

try:
    from PIL import Image
    PIL_installed = True
except ModuleNotFoundError:
    PIL_installed = False

try:
    from elderflower.plotting import AsinhNorm, LogNorm, colorbar
except:
    logger.error('elderflower is not installed. Missing some utilities.')
    
# Temperature table for RGB display
kelvin_table = {
    1000: (255,56,0),
    1500: (255,109,0),
    2000: (255,137,18),
    2500: (255,161,72),
    3000: (255,180,107),
    3500: (255,196,137),
    4000: (255,209,163),
    4500: (255,219,186),
    5000: (255,228,206),
    5500: (255,236,224),
    6000: (255,243,239),
    6500: (255,249,253),
    7000: (245,243,255),
    7500: (235,238,255),
    8000: (227,233,255),
    8500: (220,229,255),
    9000: (214,225,255),
    9500: (208,222,255),
    10000: (204,219,255)}
    

def display(image, mask=None,
            k_std=10, cmap="gray_r",
            a=0.1, fig=None, ax=None):
    """ Visualize an image """
    
    if mask is not None:
        sky_vals = image[(mask==0)].ravel()
    else:
        sky_vals = sigma_clip(image, 3).compressed()
        
    sky_mean, sky_std = np.median(sky_vals), np.std(sky_vals)
    
    if fig is None: fig = plt.figure(figsize=(12,8))
    if ax is None: ax = plt.subplot(111)
    ax.imshow(image, cmap="gray_r",
              norm=AsinhNorm(a, vmin=sky_mean-2*sky_std,
                                vmax=sky_mean+k_std*sky_std))
    ax.axis('off')
    return ax
    
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
    
def draw_reference_residual(Image):
    """ Plotting function for visualizing output """
    
    mask = Image.mask
    norm = Image.norm
    norm0 = Image.norm0
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(26, 6))
    ax1.imshow(Image.image_ref, norm=norm)
    ax1.set_title('Reference Image')

    ax2.imshow(Image.image_extend_conv, norm=norm)
    ax2.set_title('smoothed')

    ax3.imshow(Image.image_extend_ds, norm=norm)
    ax3.set_title('downsampled')

    ax4.imshow(Image.image_extend_us, norm=norm)
    ax4.set_title('upsampled')
    
    plt.tight_layout()
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    img_ref_res_ = Image.img_ref_res.copy()
    img_ref_res_[mask] = np.nan
    m = ax1.imshow(img_ref_res_, norm=norm)
    colorbar(m, ax = ax1)
    ax1.set_title('Reference Residual')

    img_out_ = Image.image_output.copy()
    img_out_[mask] = np.nan
    m = ax2.imshow(img_out_, norm=norm0)
    colorbar(m, ax = ax2)
    ax2.set_title('Output Residual')
    
    plt.tight_layout()
    plt.show()


def convert_temp(image, temp=7500):
    """ Implement a temperatue correction for displaying RGB image.
    
    Parameters
    ----------
    image: numpy array of WxHx3 of uint8 [0,255]
        Input color image.
    temp: int
        Display temperatue.
    
    Returns
    -------
    image_rgb: numpy array of WxHx3 of float [0,1]
        Output image with adjusted temperatue.
        
    """
    
    r, g, b = kelvin_table[temp]
    matrix = ( r / 255.0, 0.0, 0.0, 0.0,
               0.0, g / 255.0, 0.0, 0.0,
               0.0, 0.0, b / 255.0, 0.0 )
    if PIL_installed:
        im = Image.fromarray(image)
        im_convert = image.convert('RGB', matrix)
        image_rgb = np.array(im_convet)
    else:
        logger.error('PIL is not installed. No temperature correction applied.')
        image_rgb = None
        
    return image_rgb
    

def correct_colors(image, w=[1,1,1],verbose=False):
    '''
    ---------------------------------------------------------------------------
                    Correct image colors (remove color casts)
    ---------------------------------------------------------------------------
    
    Implements a simple color correction using the Gray World Color Assumption
    and White Point Correction. Snippets from Vonikakis
    
    Snippets from  Vonikakis, V., Arapakis, I. & Andreadis, I. (2011).
    
    Parameters
    ----------
    image: numpy array of WxHx3 of uint8 [0,255]
        Input color image.
    verbose: boolean
        Display outputs.
    
    Returns
    -------
    image_out: numpy array of WxHx3 of float [0,1]
        Output image with adjusted colors.
        
    '''
    
    image_out = img_as_float(image.copy())  # [0,1]
    
    # mean of all channels
    image_mean = (image_out[:,:,0].mean() +
                  image_out[:,:,1].mean() +
                  image_out[:,:,2].mean()) / 3
                  
    # logarithm base to which each channel will be raised
    base_r = w[0]*image_out[:,:,0].mean() / image_out[:,:,0].max()
    base_g = w[1]*image_out[:,:,1].mean() / image_out[:,:,1].max()
    base_b = w[2]*image_out[:,:,2].mean() / image_out[:,:,2].max()
    
    # the power to which each channel will be raised
    power_r = np.math.log(image_mean, base_r)
    power_g = np.math.log(image_mean, base_g)
    power_b = np.math.log(image_mean, base_b)
    
    # separately applying different color correction powers to each channel
    image_out[:,:,0] = (image_out[:,:,0] / image_out[:,:,0].max()) ** power_r
    image_out[:,:,1] = (image_out[:,:,1] / image_out[:,:,1].max()) ** power_g
    image_out[:,:,2] = (image_out[:,:,2] / image_out[:,:,2].max()) ** power_b
    
    if verbose is True:
        
        plt.figure(figsize=(18,7),dpi=100)
        plt.subplot(1,2,1)
        plt.imshow(image)
        plt.title('Input image')
        plt.axis('off')
        
        plt.subplot(1,2,2)
        plt.imshow(image_out, vmin=0, vmax=1)
        plt.title('Corrected colors')
        plt.axis('off')
        
        plt.tight_layout()
        plt.suptitle('Gray world color correction')
        plt.show()
    
    return image_out
