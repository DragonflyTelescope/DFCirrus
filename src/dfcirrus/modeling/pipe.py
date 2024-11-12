import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.special import expit
from scipy import stats, ndimage
from functools import partial

import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.stats import mad_std, SigmaClip, sigma_clip
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel

from photutils.background import MADStdBackgroundRMS

from ..io import logger
from .worker import Worker
from .utils import mode, match_gaussian_beam
from .image import ImageButler, PlanckImage, IRISImage

try:
    from elderflower.utils import downsample_wcs
    from elderflower.utils import make_psf_2D
except:
    logger.error('elderflower not installed. Missing some utilities.')
    

p0_RonP = [15., -8, 0.7, 0.001, 3, 0.99]
p0_GonP = [8., -5, 0.5, 0.001, 3, 0.99]
p0_RonG = [1.6, 1., 2., 1.9, 0.6, 0, 3., 0.95]
p0_GonR = [0.65, 1., 0.5, 0.45, 0.2, 0, 3, 0.95]
    
def run_cirrus_modeling(fn_g, fn_r, 
                        fn_residual_g, fn_residual_r, 
                        starmods_g, starmods_r, 
                        scale=0.25,
                        coords_cutout=None,
                        shape_cutout=(4000, 3000),
                        p0={'RonP':p0_RonP, 'GonP':p0_GonP, 'RonG':p0_RonG, 'GonR':p0_GonR},
                        p_range=(0,2),
                        g_range=(-4,12),
                        r_range=(-6,18),
                        std_g=None,
                        std_r=None,
                        mask_std=2.5,
                        planck_model='radiance',
                        fit_Planck=True,
                        fill_mask=True,
                        plot=True,
                        target=''):
    
    foo = ImageButler(hdu_path_G=fn_residual_g, hdu_path_R=fn_residual_r, obj_name=target)
    
    if coords_cutout is None:
        coords_cutout = foo.Image_G.get_center_coord()
    foo.make_cutout(shape=shape_cutout, coord=coords_cutout)
    
    ZP_g, ZP_r = foo.Image_G.ZP, foo.Image_R.ZP

    image_g_cutout = foo.Image_G.image_cutout.copy()
    image_r_cutout = foo.Image_R.image_cutout.copy()

    wcs = foo.wcs_cutout
    
    starmods_g_cutout = foo.apply_cutout(starmods_g)
    starmods_r_cutout = foo.apply_cutout(starmods_r)
    
    bkgrms = MADStdBackgroundRMS(sigma_clip=SigmaClip(cenfunc=np.nanmedian, stdfunc=mad_std))
    mask_nan = np.isnan(image_g_cutout) | np.isnan(image_r_cutout)

    for k in range(2):
        if k==0:
            mask = mask_nan.copy()
        if std_g is None:
            std_g = bkgrms(image_g_cutout[~mask])
        if std_r is None:
            std_r = bkgrms(image_r_cutout[~mask])

        mask_g = starmods_g_cutout > mask_std * std_g
        mask_r = starmods_r_cutout > mask_std * std_r

        mask = mask_g | mask_r | mask_nan

        bkg_g = mode(image_g_cutout[~mask])
        bkg_r = mode(image_r_cutout[~mask])

        print("Iteration #{:}:  BKG g/r = {:.4f} / {:.4f},  RMS g/r = {:.4f} / {:.4f}".format(k+1, bkg_g, bkg_r, std_g, std_r))
        
    image_g_input, image_r_input = image_g_cutout.copy() - bkg_g, image_r_cutout.copy() - bkg_r
    
    worker = Worker([image_g_input, image_r_input], mask, wcs, filters=['G', 'R'], 
                    bkg_vals=[0,0], stds=[std_g,std_r], ZPs=[ZP_g, ZP_r])
    if plot:
        worker.CMD(bins=100, xyrange=[[25, 30], [-1.3, 2.7]])
        plt.show()

    if fill_mask:
        worker.fill_mask(stddev=2)

    worker.rescale(scale=scale, method='binning')
    worker.set_cond(sigma_lower=-15, sigma_upper=25)
    
    if plot:
        worker.CMD(bins=100, xyrange=[[25, 30], [-1.3, 2.7]])
        plt.show()
    
    pla = PlanckImage('/Users/qliu/Data/PLA/HFI_CompMap_ThermalDustModel_2048_R1.20.fits')
    
    wcs_rp = downsample_wcs(wcs, scale=scale)
    shape_output = int(worker._image_shape[0] * scale), int(worker._image_shape[1] * scale)
    
    dust_map = pla.reproject(wcs_rp, shape_output, model=planck_model)
    
    if planck_model == 'tau':
        planck_scale_factor = 1e5
    elif planck_model == 'radiance':
        planck_scale_factor = 1e7
    print(f'Using planck {planck_model} dust models.')
    
    val_Planck_ = dust_map[worker.fit_range].ravel() * planck_scale_factor
    val_Planck = val_Planck_.copy()
    worker.x_P = val_Planck
    
    if plot:
        fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(26,6), dpi=120)
        im = ax1.imshow(worker.data_G, vmin=-3*std_g, vmax=+8*std_g, cmap=plt.cm.inferno)
        plt.colorbar(im, ax=ax1)
        im = ax2.imshow(worker.data_R, vmin=-3*std_r, vmax=+8*std_r, cmap=plt.cm.inferno)
        plt.colorbar(im, ax=ax2)
        im = ax3.imshow(dust_map, cmap=plt.cm.inferno)
        plt.colorbar(im, ax=ax3)
        plt.tight_layout()
        plt.show()
    
    if fit_Planck:
        # Convolve to Planck beam
        params = {"fwhm":6.1, "beta":6.7, "frac":0.3,
                  "n_s":np.array([3.5, 2.1]), "theta_s":np.array([5, 10**2.13])}
        PSF_DF, psf = make_psf_2D(params['n_s'], params['theta_s'], 
                                  frac=0.3, beta=6.6, fwhm=6, 
                                  psf_range=900, pixel_scale=2.5 * (1/scale))

        kernel = match_gaussian_beam(PSF_DF, pixel_scale=2.5 * (1/scale), fwhm_target=5*u.arcmin)

        worker.convolve(kernel)
        
        # Prepare for Planck correlation
        worker.preprocessing(use_conv=True, normed=False, physical_units=True)
        val_DF_G = worker.x_G
        val_DF_R = worker.x_R
        
        # p0
        p0_RonP, p0_GonP = p0['RonP'], p0['GonP']
        
        # Run Planck correlation
        worker.fit(val_Planck, val_DF_R, p0=p0_RonP,
                   n_model=1, poly_deg=1, sigmoid=False, fit_bkg2d=False, include_noise=True,
                   method='Nelder-Mead', name='RonP')
        if plot:
            worker.fitter_RonP.plot(nbins=50, xrange=p_range, yrange=r_range, color='r', xlabel='$I_{P}$', ylabel='$I_{R}$')

        worker.fit(val_Planck, val_DF_G, p0=p0_GonP,
               n_model=1, poly_deg=1, sigmoid=False, fit_bkg2d=False, include_noise=True,
               method='Nelder-Mead', name='GonP')
        if plot:
            worker.fitter_GonP.plot(nbins=50, xrange=p_range, yrange=g_range, color='g', xlabel='$I_{P}$', ylabel='$I_{G}$')
        
        slope_rg = worker.fitter_RonP.params_fit[0]/worker.fitter_GonP.params_fit[0]
        print('Slope from Planck = {:.4f}'.format(slope_rg))
        print('Color from Planck = {:.4f}'.format(2.5*np.log10(slope_rg)))
        
        Ig_0 = worker.fitter_GonP.params_fit[1]
        Ir_0 = worker.fitter_RonP.params_fit[1]
        
    else:
        Ig_0 = Ir_0 = 0
    
    # Prepare for G and R correlation
    worker.preprocessing(use_conv=False, normed=False, physical_units=True)
    val_DF_G = worker.x_G
    val_DF_R = worker.x_R

    worker.val_DF_G_new = val_DF_G - Ig_0
    worker.val_DF_R_new = val_DF_R - Ir_0
    
    # Run G and R fit
    g_range = [g_range[0]-1, g_range[1]+2]
    r_range = [r_range[0]-1, r_range[1]+2]
    
    # p0
    p0_RonG, p0_GonR = p0['RonG'], p0['GonR']
    
    worker.fit(val_DF_G, val_DF_R, p0=p0_RonG,
           poly_deg=1, sigmoid=False, include_noise=True, piecewise=True,
           method='Nelder-Mead', name='RonG')
    if plot:
        worker.fitter_RonG.plot(nbins=100, xrange=g_range, yrange=r_range, color='r', xlabel='$I_{G}$', ylabel='$I_{R}$')
    
    worker.fit(val_DF_R, val_DF_G, p0=p0_GonR,
           poly_deg=1, sigmoid=False, include_noise=True, piecewise=True,
           method='Nelder-Mead', name='GonR')
    if plot:
        worker.fitter_GonR.plot(nbins=100, xrange=r_range, yrange=g_range, color='g', xlabel='$I_{R}$', ylabel='$I_{G}$')
    
    return worker

def run_cirrus_removal(worker, rht_radius=18, scale_factor=0.5, vlim=[-3, 15], plot=True, target=''):
    print('Running cirrus decomposition...')
    # Run color modeling
    f_G2R = build_model(worker.fitter_RonG.params_fit, poly_deg=1, sigmoid=False, piecewise=True)
    f_R2G = build_model(worker.fitter_GonR.params_fit, poly_deg=1, sigmoid=False, piecewise=True)

    worker.processing(f_R2G, f_G2R, 
                      rht_radius=rht_radius, 
                      kernel_type='linear', 
                      fill_mask=True, 
                      use_peak=True,
                      remove_compact=True, 
                      remove_compact_qantile=0.995, 
                      n_threshold=3, 
                      background_percentile=25,
                      scale_factor=scale_factor, 
                      median_filter_size=3, 
                      vlim=vlim, 
                      name=target, 
                      plot=plot,
                      figsize=(22, 7))
    return worker

def build_model(params, poly_deg=2, piecewise=False, sigmoid=True, a=0.2):
    
    if piecewise is False:
        ind = poly_deg+1
        # 1d polynomials
        coefs = params[:ind]
        poly = np.polynomial.Polynomial(coefs[::-1])
        y = lambda x: poly(x)
    else:
        ind = 4
        # piecewise linear
        A1 = params[0]
        x0, y0 = params[1:3]
        A2 = params[3]

        B1 = y0 - A1 * x0 
        B2 = y0 - A2 * x0

        f1 = lambda x: A1 * x + B1
        f2 = lambda x: A2 * x + B2

        y = lambda x: np.piecewise(x, [x<x0, x>=x0], [f1, f2])

    if sigmoid:
        # smoothing with sigmoid function
        x_min = params[ind+1]
        h = lambda x: expit((x-x_min)/a)
    else:
        x_min = 0
        h = lambda x: 1

    func = lambda x: y(x) * h(x) + (1-h(x)) * y(x_min-a)
    return func
