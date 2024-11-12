import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, ndimage

import astropy.units as u
from astropy.table import Table
from astropy.stats import SigmaClip
from astropy.convolution import convolve_fft, Gaussian2DKernel, interpolate_replace_nans

from photutils.background import Background2D, SExtractorBackground
from photutils.segmentation import detect_sources, deblend_sources, SourceCatalog

from maskfill import maskfill

from ..io import logger

def mode(x):
    return 2.5*np.nanmedian(x) - 1.5*np.nanmean(x)

def background_extraction(field, mask=None, return_rms=True,
                          b_size=64, f_size=3, n_iter=5, **kwargs):
    """ Extract background & rms image using SE estimator with mask """
    
    try:
        Bkg = Background2D(field, mask=mask,
                           bkg_estimator=SExtractorBackground(),
                           box_size=b_size, filter_size=f_size,
                           sigma_clip=SigmaClip(sigma=3., maxiters=n_iter),
                           **kwargs)
        back = Bkg.background
        back_rms = Bkg.background_rms
        
    except ValueError:
        img = field.copy()
        if mask is not None:
            img[mask] = np.nan
        back = np.nanmedian(field) * np.ones_like(field)
        back_rms = np.nanstd(field) * np.ones_like(field)
        
    if return_rms:
        return back, back_rms
    else:
        return back

def photutils_source_detection(data, mask=None, n_threshold=3, b_size=64, npixels=10):
    
    """ Source detection using photuils """
    
    back, back_rms = background_extraction(data, b_size=b_size)
    threshold = back + (n_threshold * back_rms)
    
    segm_sm = detect_sources(data, threshold, npixels=npixels, mask=mask)
    ma = segm_sm.data!=0
    data_ma = np.ma.array(data - back, mask=ma)

    return data_ma, segm_sm


default_kernel = np.array([[1,2,1], [2,4,2], [1,2,1]])

def _byteswap(arr):
    """
    If array is in big-endian byte order (as astropy.io.fits
    always returns), swap to little-endian for SEP.
    """
    if arr is not None and arr.dtype.byteorder=='>':
        arr = arr.byteswap().newbyteorder()
    return arr

def sep_extract_sources(image, 
    thresh = 3, 
    minarea=10, 
    filter_kernel=default_kernel, 
    filter_type='matched', 
    deblend_nthresh=64, 
    deblend_cont=0.001, 
    clean=True, 
    clean_param=1.0,
    bw=64, 
    bh=None, 
    fw=3, 
    fh=None, 
    mask=None,
    subtract_sky=True,
    **kwargs):
    """
    Extract sources using sep.
    
    https://sep.readthedocs.io

    Parameters
    ----------
    path_or_pixels : pathlib.Path or str or np.ndarray
        Image path or pixels.
    thresh : float, optional
        Threshold pixel value for detection., by default 2.5.
    minarea : int, optional
        Minimum number of pixels required for an object, by default 5.
    filter_kernel : np.ndarray, optional
        Filter kernel used for on-the-fly filtering (used to enhance detection). 
        Default is a 3x3 array: [[1,2,1], [2,4,2], [1,2,1]].
    filter_type : str, optional
        Filter treatment. This affects filtering behavior when a noise array is 
        supplied. 'matched' (default) accounts for pixel-to-pixel noise in the filter 
        kernel. 'conv' is simple convolution of the data array, ignoring pixel-to-pixel 
        noise across the kernel. 'matched' should yield better detection of faint 
        sources in areas of rapidly varying noise (such as found in coadded images 
        made from semi-overlapping exposures). The two options are equivalent 
        when noise is constant. Default is 'matched'.
    deblend_nthresh : int, optional
        Number of thresholds used for object deblending, by default 32.
    deblend_cont : float, optional
        Minimum contrast ratio used for object deblending. Default is 0.005. 
        To entirely disable deblending, set to 1.0.    
    clean : bool, optional
        If True (default), perform cleaning.
    clean_param : float, optional
        Cleaning parameter (see SExtractor manual), by default 1.0.
    bw : int, optional
        Size of background box width in pixels, by default 64.
    bh : int, optional
        Size of background box height in pixels. If None, will use value of `bw`.
    fw : int, optional
        Filter width in pixels, by default 3.
    fh : int, optional
        Filter height in pixels.  If None, will use value of `fw`.
    mask : np.ndarray, optional
        Mask array, by default None.
    subtract_sky : bool, optional
        If True (default), perform sky subtraction. 
    flux_aper : list of float, optional
        Radii of aperture fluxes, by default [2.5, 5, 10].
    flux_ann : list of tuple, optional
        Inner and outer radii for flux annuli, by default [(3, 6), (5, 8)].
    zpt : float, optional
        Photometric zero point. If not None, magnitudes will be calculated.
    **kwargs
        Arguments for sep.Background. 

    Returns
    -------
    source : Sources
        Source object with `cat` and `segmap` as attributes. 
    """
    import sep

    # Use square boxes if heights not given.
    bh = bw if bh is None else bh
    fh = fw if fh is None else fh
    
    # Build background map using sep.
    mask = _byteswap(mask)
    data = _byteswap(image)
    bkg = sep.Background(image, bw=bw, bh=bh, fw=fw, fh=fh, mask=mask, **kwargs)

    # If desired, subtract background. 
    if subtract_sky:
        data = data - bkg

    # Extract sources using sep.
    cat, segmap = sep.extract(
        data, 
        thresh,  
        err=bkg.rms(),
        mask=mask, 
        minarea=minarea, 
        filter_kernel=filter_kernel, 
        filter_type=filter_type,
        deblend_nthresh=deblend_nthresh, 
        deblend_cont=deblend_cont, 
        clean=clean, 
        clean_param=clean_param, 
        segmentation_map=True    
    )
    return Table(cat), segmap


def assign_value_by_filters(values, filters=['G', 'R']):
    """ Return a dictionary by filters. """
    if len(values)==1:
        result = {filt:values for filt in filters}
    else:
        result = {filt:val for (filt, val) in zip(filters, values)}
    return result

def resample_image(image, mask=None, shape_new=None,
                   method='cubic', fill_value=0):
    
    """ Resample image and mask """
    
    NY, NX = image.shape
    
    # data grid
    yy, xx = np.mgrid[0:NY, 0:NX] # IMAGE coordinates
    
    # new NAXIS
    if shape_new is None:
        return image, mask 
    else:
        NY_, NX_ = shape_new
    
    # new grid
    xxp, yyp = np.meshgrid(np.linspace(0, NX-1, NX_), np.linspace(0, NY-1, NY_))
    
    # interpolation with griddata
    ma = np.isnan(image)
    image_new = interpolate.griddata((yy[~ma], xx[~ma]), image[~ma], (yyp, xxp), method=method, fill_value=fill_value)
    if mask is not None:
        mask_new = interpolate.griddata((yy.ravel(), xx.ravel()), mask.ravel(), (yyp, xxp), method='nearest')
    else:
        mask_new = None
    
    # set nan back
    ma_new = interpolate.griddata((yy.ravel(), xx.ravel()), ma.ravel(), (yyp, xxp), method='nearest') > 0.5
    image_new[ma_new] = np.nan
    
    return image_new, mask_new
    
    
def match_gaussian_beam(PSF, pixel_scale, fwhm_target=5*u.arcmin, plot=False):
    """ 
    Create kernel for matching the to Gaussian beam with taget FWHM.
    
    PSF: 2d array of the PSF
    pixel_scale: pixel scale of image for matching, arcsec/pix 
    fwhm: FWHM of the target Gaussian, astropy.Angle
    
    """
    
    from astropy.stats import gaussian_fwhm_to_sigma
    from astropy.modeling.models import Gaussian2D
    from photutils.psf import create_matching_kernel
    from photutils.psf import (HanningWindow, TukeyWindow, TopHatWindow)
    
    sigma_gbeam = (gaussian_fwhm_to_sigma *  fwhm_target / (pixel_scale * u.arcsec)).decompose().value
    print("Sigma Beam = %.3f pix"%sigma_gbeam)
    
    # Grid
    size = PSF.shape[0]
    cen = ((size-1)/2., (size-1)/2.)
    y, x = np.mgrid[0:size, 0:size]

    # Build Gassuain model
    gm = Gaussian2D(1, cen[0], cen[1], sigma_gbeam, sigma_gbeam)
    gbeam = gm(x, y)
    gbeam /= gbeam.sum()

    kernel = create_matching_kernel(PSF, gbeam, window=HanningWindow())

    if plot:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
        ax1.imshow(PSF, vmax=1e-5, vmin=1e-9)
        ax1.set_title('Input PSF')
        ax2.imshow(gbeam)
        ax2.set_title('Target PSF')
        im = ax3.imshow(kernel, cmap='Greys_r')
        ax3.set_title('Matched Kernel')
        plt.show()
    
    return kernel
    
def create_matching_kernel(source_psf, target_psf, window=None):
    """
    Create kernel transform resolution from given source PSF to target PSF.

    Parameters
    ----------
    source_psf : np.ndarray
        Point-spread function of source image.
    target_psf : np.ndarray
        Point-spread function of target image.
    window : callable, optional
        Window function, by default None.

    Returns
    -------
    kernel : np.ndarray
        Matching kernel.
    """
    if source_psf.shape != target_psf.shape:
        raise ValueError('source_psf and target_psf must have the same shape '
                         '(i.e. registered with the same pixel scale).')

    source_otf = np.fft.fftshift(np.fft.fft2(source_psf))
    target_otf = np.fft.fftshift(np.fft.fft2(target_psf))
    ratio = target_otf / source_otf

    # Apply a window function in frequency space.
    if window is not None:
        ratio *= window(target_psf.shape)

    kernel = np.real(np.fft.fftshift((np.fft.ifft2(np.fft.ifftshift(ratio)))))

    return kernel

def remove_compact_emission(image, mask=None, 
                            kernel_stddev=(20,3), 
                            kernel_type='Gaussian',
                            quantile=0.995,
                            n_theta=None, 
                            rht_radius=36,
                            background_percentile=50,
                            fill_mask=True,
                            use_peak=True,
                            use_output='residual',
                            n_threshold=None,
                            kernel_replace_masked=9,
                            background_size=128,
                            source_extractor='photutils',
                            plot=False, figsize=(18, 6)):
    
    """ 
    Remove compact emission using rolling Hough Transform. 
    
    n_theta: number of directional filters in [0, pi].
    quantile: quantile of residual maps for isolated overdensity detection.
    rht_radius: radii of the rolling Hough Transform filter size.
    background_percentile: percentile of local background in the transformed image. Must be 0 to 100.
    
    """

    image_ = image.copy()
    image = np.ma.array(image, mask=mask)
    
    rht = RHT_worker(image, mask, radius=rht_radius)
    
    if kernel_type == 'Gaussian':
        logger.info(f'Removing compact emission using Gaussian kernel [{kernel_stddev[0]}x{kernel_stddev[1]}]:')
        if n_theta is None:
            n_theta = np.round(np.arctan2(max(kernel_stddev), min(kernel_stddev)) * 180/np.pi).astype(int) // 5
            
        # do spatial filtering on the input image in different directions
        imgs_conv = np.empty([n_theta, image.shape[0], image.shape[1]])
        bkg = np.nanmean(image)
        for i in range(n_theta):
            kernel = Gaussian2DKernel(x_stddev=kernel_stddev[0], y_stddev=kernel_stddev[1], theta=i*np.pi/n_theta)
            img_conv = convolve_fft(image, kernel, boundary='fill', fill_value=bkg, mask=mask)
            imgs_conv[i] = img_conv

        image_conv_bkg = np.nanmean(imgs_conv, axis=0)

        # smooth the input image
        image_smooth = convolve_fft(image, Gaussian2DKernel(kernel_stddev[1]), boundary='fill', fill_value=bkg, mask=mask)
        image_out = image_smooth - image_conv_bkg
        
        rht.image_out = image_out
        rht.image_smooth=image_smooth
        rht.image_conv_bkg=image_conv_bkg
    
    elif kernel_type == 'linear':
        logger.info('Removing compact emission using RHT R = {:}:'.format(rht_radius))
        rht.work(n_theta, background_percentile, kernel_replace_masked, use_peak=use_peak)
        
        if use_output=='residual':
            image_out = rht.image_residual
        elif use_output=='ratio':
            image_out = rht.image_ratio
        
    # detect compact emission
    if n_threshold is None:
        logger.info('    - Detecting blobs using quantiles...')
        q_proc = np.nanquantile(abs(image_out), quantile)
        isolated = image_out>=q_proc
    else:
        logger.info('    - Detecting blobs using source detection S/N={:.1f}...'.format(n_threshold))
        if source_extractor == 'sep':
            cat, segmap = sep_extract_sources(data, thresh=n_threshold, min_area=10, bw=background_size, subtract_sky=False)
            segm = segmap.copy()
            for label in np.argwhere(cat['b']/cat['a'] < 0.5):
                segm[segmap==label] = 0
            isolated = segm>0
        elif source_extractor == 'photutils':
            data_ma, segm_sm = photutils_source_detection(image_out, mask=None, n_threshold=n_threshold, b_size=background_size)
            segm_deb = deblend_sources(image_out, segm_sm, npixels=10,
                                       nlevels=64, contrast=0.001)

            cat_segm = SourceCatalog(image, segm_deb)
            segm_deb.keep_labels(segm_deb.labels[cat_segm.elongation<2])
            isolated = segm_deb.data>0
        else:
            logger.error('Tools of Source Extraction not found!')
            raise NameError
        
    isolated = ndimage.binary_dilation(isolated, iterations=5)
    rht.isolated = isolated

    # remove unconnected emission from image and interploate
    image_[isolated|mask] = np.nan
    
    if fill_mask:
        image_proc, image_proc_unsmoothed = maskfill(image_, np.isnan(image_), size=kernel_replace_masked)
        
        # image_proc = fill_nan_iterative(image_, k0=1)
#        image_proc = interpolate_replace_nans(image_, Gaussian2DKernel(kernel_replace_masked), 
#                                              convolve=convolve_fft)
    else:
        image_proc = image_

    if plot:
        plt.figure()
        plt.hist(image_conv_sum.ravel())
        plt.show()
    
        fig, (ax1,ax2) = plt.subplots(1, 2, figsize=figsize)
        im=ax1.imshow(image, vmin=-2, vmax=3, cmap='viridis')
        im=ax2.imshow(image_conv_sum, vmin=-2, vmax=3, cmap='viridis')
        
        fig, (ax1,ax2) = plt.subplots(1, 2, figsize=figsize)
        im=ax1.imshow(image_smooth, vmin=-2, vmax=3, cmap='viridis')
        im=ax2.imshow(image_out, vmin=0, vmax=2, cmap='viridis')
        plt.savefig('tmp/image_out.png',dpi=75)
        plt.show()
    
    return image_proc, rht


class RHT_worker:
    """
    A class running the rolling Hough Transform.
    """

    def __init__(self, image, mask, radius=36):
        
        self.image = image
        self.mask = mask
        self.radius = radius
        
        self.image_shape = image.shape
        
        self.bkg = np.nanmedian(image[~mask])

    def work(self, n_theta=None,
             background_percentile=50,
             kernel_replace_masked=9, use_peak=True,
             normed_by_background=False, bkg=1e5):
        """ Run the RHT and compute the max response. """
        from fil_finder.rollinghough import circular_region
        from photutils.aperture import RectangularAperture
        
        image, mask = self.image.copy(), self.mask.copy()
        radius = self.radius
        bkg = self.bkg
        
        if n_theta is None:
            n_theta = int(np.pi/np.sqrt(2)*(self.radius-1))
        
        self.n_theta = n_theta
        
        thetas = np.linspace(0, np.pi, n_theta)
        self.thetas = thetas

        circle, mesh = circular_region(radius)
        cen_circle = (circle.shape[1]-1)/2., (circle.shape[0]-1)/2.

        line_cube = np.empty((n_theta, circle.shape[0], circle.shape[1]))
        for k, theta in enumerate(thetas):
            PA = theta - np.pi/2.
            rect_mask = RectangularAperture(cen_circle, 2*circle.shape[0], 2, PA).to_mask()
            line_cube[k] = rect_mask.to_image(circle.shape) * circle
        self.line_cube = line_cube

        imgs_conv = np.empty([n_theta, self.image_shape[0], self.image_shape[1]])
        
        for i, line in enumerate(line_cube):
            img_conv = convolve_fft(image, line, boundary='fill', fill_value=bkg, mask=mask)
            imgs_conv[i] = img_conv
            
        self.images_conv = imgs_conv
        imgs_conv = np.ma.array(imgs_conv, mask=np.isnan(imgs_conv))
        
        size_mf = np.max([np.min([radius//4, 4]), 1])
        
        logger.info('    - Smoothing the image by median filtering and caculating the residual...')
        if np.isnan(image).any():
            logger.info('    - Filling nan values for smoothing...')
            # image = fill_nan_iterative(image, k0=1)
            image = interpolate_replace_nans(image, Gaussian2DKernel(kernel_replace_masked), convolve=convolve_fft)
            
        image_smooth = convolve_fft(image, Gaussian2DKernel(1), boundary='fill', fill_value=bkg, mask=mask)
        image_smooth = ndimage.median_filter(image, size=size_mf, mode='reflect', cval=bkg)
    
        self.image_smooth = image_smooth

        logger.info('    - Computing the local RHT responses...')
        if use_peak:
            image_response = np.nanmax(imgs_conv, axis=0)
        else:
            image_response = np.nanquantile(imgs_conv, background_percentile/100, axis=0)
            
        image_response = ndimage.median_filter(image_response, size=size_mf, mode='reflect', cval=bkg)
        
        residual_conv = image_smooth - image_response
        ratio_conv = (image_smooth+bkg) / (image_response+bkg) - 1 # add bkg to avoid 0/0
#        image_conv_bkg = np.quantile(imgs_conv, background_percentile/100, axis=0)
#        ratio_conv = image * (1-(image_response-image_conv_bkg)/image_response) # add bkg to avoid 0/0

        if use_peak & normed_by_background:
            ratio_map = (image_conv_bkg+bkg) / (image_response+bkg) - 1
            residual_conv = residual_conv * ratio_map
                        
        # residual_conv[residual_conv<0] = 0
        residual_conv = ndimage.median_filter(residual_conv, size=size_mf, mode='reflect', cval=0)
        #ratio_conv = ndimage.median_filter(ratio_conv, size=size_mf, mode='reflect', cval=0)
        
        self.image_response = image_response
        self.image_residual = residual_conv
        self.image_ratio = ratio_conv
        #self.image_conv_bkg = image_conv_bkg
        
        
def fill_nan(image, image_fill, max_distance=1):
    """ 
    Fill nan pixels in image using the image_fill.
    """
    
    is_nan_map = np.isnan(image)
    
    # Euclidean distance to background
    distance = ndimage.distance_transform_edt(is_nan_map)
    
    # fill small holes
    to_fill = (distance < max_distance) & (distance>0) 
    image[to_fill] = image_fill[to_fill]
    
    return image, to_fill


def fill_nan_iterative(image, k0=0, kmax=None):
    from skimage.measure import regionprops
    
    i = 0
    k = k0
    max_d = 2**k # initial max distance
    kernel = Gaussian2DKernel(max_d)
    
    # Premliminary mask
    image_fill = interpolate_replace_nans(image, kernel, convolve=convolve_fft)
    image, to_fill = fill_nan(image, image_fill, max_distance=max_d)
    is_nan_map = np.isnan(image)
    
    while is_nan_map.any():
        msg = "    - Filling nan values using Gaussian kernel interpolation... Iteration {:d}: {:d} pixels left"
        logger.info(msg.format(i+1, is_nan_map.sum()))
        
        i += 1
        k += 1
        max_d *= 2
        kernel = Gaussian2DKernel(max_d)
        
        # distance to background
        label_im, n_labels = ndimage.label(is_nan_map)
        distance = ndimage.distance_transform_edt(is_nan_map)
        
        # nan hole size
        nan_size_map = np.zeros_like(label_im)
        properties = np.array(regionprops(label_im))
        for prop in properties:
            nan_size_map[label_im==prop.label] = prop.equivalent_diameter_area
            
        # fill small holes
        image_fill = interpolate_replace_nans(image, kernel, convolve=convolve_fft)
        to_fill = ((distance>0) & (distance < 2*max_d)) | (nan_size_map < 2*max_d) 
        image[to_fill] = image_fill[to_fill]
        
        is_nan_map = np.isnan(image)
        if (kmax is not None) & (k==kmax):
            break
        
    return image
      
    
    

def gaussian_process_regression(image, noise=1e-2, 
                                scale_length=5,
                                n_splits=5,
                                fix_scale_length=True,
                                random_state=123456):
    
    """ 
    Gaussian regression modeling of sky using landmarks. 

    Parameters
    ----------
    X: Nx2 np.ndarray
        Variables for training (positions of landmarks).

    y: MxN np.ndarray
        Target values for training (normalized local background value)

    noise: MxN array
        Noise of y for training (normalized local background rms)

    scale_length: float
        Scale length in pixel of the kernel.
        If fix_scale_length=False, this is the initial guess.

    n_splits: int, optional, default 5
        K-fold number for cross-validation

    max_n_samples: int, optional, default 10000
        Maximum number of sample in each fold

    fix_scale_length: bool, optional, default True
        Whether to fix scale length in the training.

    wcs: astropy.wcs.WCS, optional
        WCS of landmarks to transform the unit in plotting.

    Ind: MxN np.ndarray, default None
        Indice stores the index of the frame in the list in the arrays.
        This is used to connect points from the same frame.

    ra_quantile: float, opional, default 0.5
        Quantile at which the Dec slice is displayed for viz.

    dec_quantile float, opional, default 0.5
        Quantile at which the RA slice is displayed for viz.

    dX_range: float, opional, default 200
        X range in pixel at which the RA slice is displayed for viz.

    dY_range: float, opional, default 200
        Y range in pixel at which the Dec slice is displayed for viz.

    plot: bool, optional, default True
        If True, plot modeling result in X/Y slice for viz.

    Returns
    -------
    gp_list: list of sklearn.gaussian_process.GaussianProcessRegressor
        List of Gaussian Process Regressor

    """
    from sklearn.gaussian_process import GaussianProcessRegressor, kernels
    from sklearn.model_selection import KFold

    gp_list = []
    
    nY, nX = image.shape
    
    X_min, X_max = 0, nX-1
    Y_min, Y_max = 0, nY-1
    X_grid = np.linspace(X_min, X_max, nX)
    Y_grid = np.linspace(Y_min, Y_max, nY)

    X = np.array([[x, y] for x in X_grid for y in Y_grid])
    y = image.reshape(-1)
    
    X = X[~np.isnan(y)]
    y = y[~np.isnan(y)]
    
    if np.ndim(noise)==0:
        noise = np.ones_like(y) * noise
    elif np.ndim(noise)==0:
        noise = noise.reshape(-1)

    kf = KFold(n_splits, shuffle=True, random_state=random_state) # w/ shuffle

    print('Training with %d-fold GP...'%n_splits)

    if fix_scale_length==True:
        length_scale_bounds = 'fixed'
        print('Scale length is fixed.')
    else:
        length_scale_bounds = (1, 50)
    kernel = kernels.RBF(scale_length, length_scale_bounds)

    for k, (remain_ind, fold_ind) in enumerate(kf.split(X, y)):
        X_train, y_train, noise_train = X[fold_ind], y[fold_ind], noise[fold_ind]
        gp = GaussianProcessRegressor(kernel=kernel, alpha=noise_train**2, n_restarts_optimizer=4, random_state=random_state)
        gp.fit(X_train, y_train)
        gp_list.append(gp)
        print('Model #%d is trained.'%(k+1))
        if fix_scale_length==False:
            print("GP Kernel Scale Lenght in pix: ", gp.kernel.length_scale)

    return gp_list
        


from skimage.morphology import (erosion, dilation, closing, opening,
                                area_closing, area_opening)
from skimage.measure import label, regionprops, regionprops_table

def multi_dil(im, num, element=np.ones((3,3))):
    for i in range(num):
        im = dilation(im, element)
    return im
def multi_ero(im, num, element=np.ones((3,3))):
    for i in range(num):
        im = erosion(im, element)
    return im
def multi_open(im, num, element=np.ones((3,3))):
    for i in range(num):
        im = opening(im, element)
    return im


def gini(data):
    r"""
    Calculate the `Gini coefficient
    <https://en.wikipedia.org/wiki/Gini_coefficient>`_ of a 2D array.

    The Gini coefficient is calculated using the prescription from `Lotz
    et al. 2004
    <https://ui.adsabs.harvard.edu/abs/2004AJ....128..163L/abstract>`_
    as:

    .. math::
        G = \frac{1}{\left | \bar{x} \right | n (n - 1)}
        \sum^{n}_{i} (2i - n - 1) \left | x_i \right |

    where :math:`\bar{x}` is the mean over all pixel values
    :math:`x_i`.

    The Gini coefficient is a way of measuring the inequality in a given
    set of values. In the context of galaxy morphology, it measures how
    the light of a galaxy image is distributed among its pixels.  A Gini
    coefficient value of 0 corresponds to a galaxy image with the light
    evenly distributed over all pixels while a Gini coefficient value of
    1 represents a galaxy image with all its light concentrated in just
    one pixel.

    Usually Gini's measurement needs some sort of preprocessing for
    defining the galaxy region in the image based on the quality of the
    input data. As there is not a general standard for doing this, this
    is left for the user.

    Parameters
    ----------
    data : array-like
        The 2D data array or object that can be converted to an array.

    Returns
    -------
    gini : `float`
        The Gini coefficient of the input 2D array.
    """
    flattened = np.sort(np.ravel(data))
    npix = np.size(flattened)
    normalization = np.abs(np.nanmean(flattened)) * npix * (npix - 1)
    kernel = (2. * np.arange(1, npix + 1) - npix - 1) * np.abs(flattened)

    return np.sum(kernel) / normalization
    
# Helper function for n-d data

def calculate_roots_coefs(coefs, x0=0):
    """ Calculate the roots of the inverse of a polynomial curve numericly.
    If several roots exist, returns the one closest to x0. """
    roots = np.roots(coefs)
    return roots[np.argmin(abs(roots-x0))]
    
def calculate_roots(coefs, range=None):
    """
    Calculate the roots of the inverse of a polynomial curve numericly.
    If several roots exist, returns the smallest root within the range (if given).
    """
    roots = np.roots(coefs)
    return get_min_root(roots, range)

def get_min_root(values, range=None):
    if range is not None:
        cond = (values>=range[0]) & (values<=range[1])
        if len(values[cond]) > 0:
            return min(values[cond])
        else:
            return range[0]
    else:
        return np.min(values)
        
def solve_equation_poly(coefs, values, initial_guess=0, range=None):
    """  Return solution of y given polynomial coefficients (in numpy order). """
    
    from numpy.polynomial import Polynomial
    from scipy.optimize import fsolve

    # Define the system of equations
    def equations(x, y=0):
        return Polynomial(coefs)(x) - y

    # Solve the system of equations
    solution = np.array([get_min_root(fsolve(equations, initial_guess, args=(val)), range=range)
                        for val in values])
    
    return solution

#def solve_equation_sym(coefs):
#    import sympy as sp
#
#    # Define the variables
#    x = sp.symbols('x')
#    y = sp.symbols('y')
#
#    # Define the polynomials
#    p1 = 0.001*x**2 + 0.01131133*x + 0.00540889
#    p2 = 0.001*y**2 + 0.00778693*y + 0.00706165
#
#    # Set up the equation p1 = p2
#    equation = sp.Eq(p1, p2)
#
#    # Solve the equation
#    solutions = sp.solve(equation, y)

def calculate_arc_length_points(xpoints, ypoints, spline=None, verbose=False):
    """ Numerically Integrate the Arc Length of a curve given discrete grid points.
        Use cubic spline interpolation to provide a continuous curve. """
        
    from scipy.interpolate import UnivariateSpline
    from scipy.integrate import quad
    
    if spline is not None:
        # Define the derivative of the spline function
        def spline_derivative(x):
            return spline.derivative()(x)

        # Define the integrand function for arc length
        def integrand(x):
            return np.sqrt(1 + spline_derivative(x)**2)

        # Set the limits of integration
        a, b = xpoints[0], xpoints[-1]

        # Perform the numerical integration
        arc_length, error = quad(integrand, a, b)
    else:
        if len(xpoints) > 5:
            # Perform spline interpolation
            spline = UnivariateSpline(xpoints, ypoints, k=3, s=0)  # Cubic spline

        elif len(xpoints) > 1:
            distances = np.sqrt(np.diff(xpoints)**2 + np.diff(ypoints)**2)

            # Sum up the distances to get the total arc length
            arc_length, error = np.sum(distances), None
            
        else:
            arc_length, error = 0, None

    if verbose:
        print(f"Arc length: {arc_length}")
        print(f"Integration error estimate: {error}")

    return arc_length

def distance_spline(t, x0, y0, spline):
    # Define the distance function between a point and the spline
    x_spline = spline(t)
    return np.sqrt((t - x0)**2 + (x_spline - y0)**2)

def calculate_distance_spline(xpoints, ypoints, spline, init_guess, bounds):
    """ Project the coordinates of a point onto a given spline curve. """
    from scipy.optimize import minimize
    # Use optimization to find the parameter t that minimizes the distance
    
    tpoints = np.empty_like(xpoints)
    spoints = np.empty_like(xpoints)
    for i, (xp, yp, x0) in enumerate(zip(xpoints, ypoints, init_guess)):
        if np.isnan(xp) | np.isnan(yp):
            tpoints[i] = spoints[i] = np.nan
        else:
            result = minimize(lambda t: distance_spline(t, xp, yp, spline), x0=x0, bounds=bounds)

            # Find the closest point on the spline
            t_optimal = result.x[0]
            s_optimal = spline(t_optimal)
            
            tpoints[i] = t_optimal
            spoints[i] = s_optimal
    
    return tpoints, spoints


def project_distance_spline(xpoints, ypoints, spline, x0, t_p0=None):
    """ Return the projected arc length along a curve for given points. """
    bounds = [(np.nanmin(xpoints), np.nanmax(xpoints))]
    
    # projected cooridnates
    t_p, s_p = calculate_distance_spline(xpoints, ypoints, spline, init_guess=xpoints, bounds=bounds)
    
    if t_p0 is None:
        # start point
        t_p0, s_p0 = calculate_distance_spline([x0[0]], [x0[1]], spline, init_guess=[x0[0]], bounds=bounds)
    
    # projected arc length
    x_p = np.array([calculate_arc_length_points(t_p0 + [t_p[k]], [0, s_p[k]], spline) for k in range(len(t_p))])
    
    return x_p
