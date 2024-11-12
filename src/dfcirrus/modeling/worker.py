import os
import time
import numpy as np
from tqdm.auto import tqdm
from functools import partial

import matplotlib.pyplot as plt
from scipy import ndimage

from astropy.convolution import convolve_fft, Gaussian2DKernel, interpolate_replace_nans
from astropy.stats import bootstrap
from astropy.utils import NumpyRNGContext
from reproject import reproject_interp

from ..io import logger

try:
    from elderflower.utils import downsample_wcs
except:
    logger.error('elderflower not installed. Missing some utilities.')
    
from .utils import assign_value_by_filters, fill_nan


class Worker:
    """

    A class running the extended emission component separation.

    Parameters
    ----------

    image_G : numpy.array
    image_R : numpy.array
    mask_G : numpy.array, bool
    mask_R : numpy.array, bool
    wcs : astropt.wcs.WCS

    """

    def __init__(self, images, mask, wcs, 
                 filters=['G', 'R'], 
                 pixel_scale=2.5,
                 bkg_vals=[0,0], 
                 stds=[5,5],
                 ZPs=[27.5, 27.5], name=''):
        
        self.images = {filt:image for (filt, image) in zip(filters, images)}
        self.filters = filters
        
        self.image_G = image_G = self.images.get('G')
        self.image_R = image_R = self.images.get('R')
        
        if np.ndim(mask)==2:
            self.mask = mask
        else:
            self.masks = {filt:mask for (filt, ma) in zip(filters, mask)}
            self.mask_G = self.masks.get('G')
            self.mask_R = self.masks.get('R')
            self.mask = self.mask_G | self.mask_R            
        
        self.wcs = wcs
        
        # Copy of original input
        self._image_G = image_G.copy()
        self._image_R = image_R.copy()
        self._mask = self.mask.copy()
        self._image_shape = self._image_G.shape
        
        self.image_G[self.mask] = np.nan
        self.image_R[self.mask] = np.nan
        
        self.bkg_vals = assign_value_by_filters(bkg_vals, filters)    
        self.bkg_val_G = self.bkg_vals.get('G')
        self.bkg_val_R = self.bkg_vals.get('R')
        
        self.stds = assign_value_by_filters(stds, filters)    
        self.std_G = self.stds.get('G')
        self.std_R = self.stds.get('R')
        
        self.ZPs = assign_value_by_filters(ZPs, filters)    
        self.ZP_G = self._ZP_G = self.ZPs.get('G')
        self.ZP_R = self._ZP_R = self.ZPs.get('R')
        
        # Data after manipulation
        self.data_G = image_G.copy()
        self.data_R = image_R.copy()

        self.filled_mask = False
        self.scale_factor = 1
        self._pixel_scale = pixel_scale
        self.convolved = False

    def __str__(self):
        return "A Worker class for cirrus modeling"
    
    @property
    def pixel_scale(self):
        """ Pixel size in arcsec """
        return self._pixel_scale / self.scale_factor 
    
    @property
    def mag_G(self):
        """ R magnitude Map """
        vals = (self.image_G-self.bkg_val_G)
        vals[vals<=0] = np.nan
        return -2.5*np.log10(vals) + self.ZP_G  + 2.5 * np.log10(self._pixel_scale**2)
    
    @property
    def mag_R(self):
        """ G magnitude Map """
        vals = (self.image_R-self.bkg_val_R)
        vals[vals<=0] = np.nan
        return -2.5*np.log10(vals) + self.ZP_R  + 2.5 * np.log10(self._pixel_scale**2)
    
    @property
    def ratio_GR(self):
        """ G/R Map """
        mask = (self.image_R<=self.bkg_val_R) | (self.image_G<=self.bkg_val_G)
        ratio_GR = ((self.image_G-self.bkg_val_G)[mask]/(self.image_R-self.bkg_val_R)[mask])
        return ratio_GR
    
    @property
    def image_shape(self):
        """ Image shape """
        return self.image_G.shape
    
    def fill_mask(self, stddev=2):
        """ Fill masked value in data """
        
        image_G = self._image_G.copy()
        image_R = self._image_R.copy()
        mask = self._mask.copy()
        
        image_G[mask] = np.nan
        image_R[mask] = np.nan
        
        print(f"Filling masked pixels with kernel size = {stddev}...")
        
        kernel = Gaussian2DKernel(stddev)
        image_fill_G = interpolate_replace_nans(image_G, kernel, convolve=convolve_fft)
        image_fill_R = interpolate_replace_nans(image_R, kernel, convolve=convolve_fft)
        
        image_G, to_fill_G = fill_nan(image_G, image_fill_G, max_distance=stddev)
        image_R, to_fill_R = fill_nan(image_R, image_fill_R, max_distance=stddev)
        mask = np.isnan(image_G) | np.isnan(image_R)
        
        self.image_G = image_G
        self.image_R = image_R
        
        self._image_G[to_fill_G] = image_G[to_fill_G]
        self._image_R[to_fill_R] = image_R[to_fill_R]
        
        self.mask = self._mask = mask
        
        print(f"Finished!")
        self.filled_mask = True
        
    
    def rescale(self, scale=0.25, method='binning',order=1):
        """ Rebin the image. Surface brightness is retained. """
        
        from astropy.nddata import block_reduce
        from .utils import resample_image
        
        self.scale_factor = scale
        print('Rescale the image by a factor of {:}'.format(scale))
        
        shape = self._image_shape
        shape_new = (int(shape[0]*scale), int(shape[1]*scale))
        
        ## resampled image and mask
        image_G = np.ma.array(self._image_G, mask=self._mask)
        image_R = np.ma.array(self._image_R, mask=self._mask)
        
        if method == 'reproject':
            wcs_ds = downsample_wcs(self.wcs, scale=scale)
            shape_out = (int(shape[0]*scale), int(shape[1]*scale))
            self.image_G, _ = reproject_interp((image_G, self.wcs), wcs_ds, shape_out=shape_out, order=order)
            self.image_R, _ = reproject_interp((image_R, self.wcs), wcs_ds, shape_out=shape_out, order=order)
            
            mask, _ = reproject_interp((self._mask, self.wcs), wcs_ds, shape_out=shape_out, order=order)
            self.mask = mask.astype(bool)
            
        elif method == 'binning':
            block_size = (int(1/scale), int(1/scale))
            self.image_G = block_reduce(image_G, block_size, func=np.ma.median)
            self.image_R = block_reduce(image_R, block_size, func=np.ma.median)
            self.mask = block_reduce(self._mask, block_size, func=np.ma.mean) > 1/(1/scale+1) 
        
        self.image_G[self.mask] = np.nan
        self.image_R[self.mask] = np.nan
        
        self.data_G = self.image_G.copy()
        self.data_R = self.image_R.copy()
        
        # resampled data
        # if not self.filled_mask:
        #     self.data_G = self.image_G.copy()
        #     self.data_R = self.image_R.copy()
        # else:
            # self.data_G = block_reduce(np.ma.array(self._image_G, mask=self._mask), block_size, func=np.ma.median)
            # self.data_R = block_reduce(np.ma.array(self._image_R, mask=self._mask), block_size, func=np.ma.median)
    
    def convolve(self, kernel, pad=None):
        """ Convolve the image with the matched kernel """
        from astropy.convolution import convolve_fft
        
        # convolve data
        data_G_conv = convolve_fft(self.data_G.copy(), kernel=kernel, boundary='fill', fill_value=self.bkg_val_G)
        data_R_conv = convolve_fft(self.data_R.copy(), kernel=kernel, boundary='fill', fill_value=self.bkg_val_R)
        
        # padding
        if pad is None:
            pad = kernel.shape[0]//10
        data_G_conv[-pad:] = data_G_conv[:, -pad:] = data_G_conv[:pad,:] =  data_G_conv[:, :pad] = np.nan
        data_R_conv[-pad:] = data_R_conv[:, -pad:] = data_R_conv[:pad,:] =  data_R_conv[:, :pad] = np.nan
        
        self.data_G_conv = data_G_conv
        self.data_R_conv = data_R_conv
        self.convolved = True
    
    def hist1d(self, sigma_lower=-5, sigma_upper=20, show_mag=False):
        """ Display 1d histogram of g and r data """
        
        mask = self.mask
        
        bkg_val_G = self.bkg_val_G
        bkg_val_R = self.bkg_val_R
        
        val_G_range = bkg_val_G + np.array([sigma_lower, sigma_upper]) * self.std_G
        val_R_range = bkg_val_R + np.array([sigma_lower, sigma_upper]) * self.std_R
        
        if show_mag:
            vals_G = self.mag_G[~mask].ravel()
            vals_R = self.mag_R[~mask].ravel()
            
            val_G_range = [-2.5*np.log10(val_G_range[1]-bkg_val_G) + self.ZP_G, 35]
            val_R_range = [-2.5*np.log10(val_R_range[1]-bkg_val_R) + self.ZP_R, 35]
            
            plt.gca().invert_xaxis()
            plt.xlabel('mag / arcsec^2')
        else:
            vals_G = self.image_G[~mask].ravel()
            vals_R = self.image_R[~mask].ravel()

            plt.xlabel('Intensity')
        
        plot_kws = dict(histtype='step', bins=50, lw=5, alpha=0.5)
        
        plt.hist(vals_G, range=val_G_range, color='seagreen', **plot_kws)
        plt.hist(vals_R, range=val_R_range, color='firebrick', **plot_kws)
        plt.ylabel('Number')

    def CMD(self, bins=100, xyrange=[[20, 28], [-1.2,2.5]], xlabel='g'):
        """ Display Color-Magnitude Diagram. """
        bins = int(bins*np.sqrt(self.scale_factor))
        
        y = self.mag_G.ravel()-self.mag_R.ravel()
        
        if xlabel.lower()=='g':
            x = self.mag_G.ravel()
            plt.xlabel('$\mu_g$')
        elif xlabel.lower()=='r':
            x = self.mag_R.ravel()
            plt.xlabel('$\mu_r$')
        plt.hist2d(x, y, bins=bins, range=xyrange)
        plt.ylabel('$\mu_g$ - $\mu_r$')
        

    def set_cond(self, sigma_lower=-5, sigma_upper=20, pad=0):
        """ Set ranges and priors for fitting """
        
        # from .models import prior_all_tf, make_legendre2d_grids
        
        bkg_val_G = self.bkg_val_G
        bkg_val_R = self.bkg_val_R
        std_G = self.std_G
        std_R = self.std_R
        
        vmin_G, vmax_G = bkg_val_G + np.array([sigma_lower, sigma_upper]) * std_G
        vmin_R, vmax_R = bkg_val_R + np.array([sigma_lower, sigma_upper]) * std_R
        
        self.fit_range = (self.image_G>vmin_G) & (self.image_G<vmax_G) \
                        & (self.image_R>vmin_R) & (self.image_R<vmax_R) & (~self.mask)
        
        if np.array(pad).ndim==0:
            pad = [pad, pad, pad, pad]
            
        yy, xx = np.indices(self.image_shape)
        pad_region = (xx>=pad[0]) & (xx<self.image_shape[1]-pad[1]) & (yy>=pad[2]) & (yy<self.image_shape[0]-pad[3])
        self.fit_range &= pad_region

#         self.prior = partial(prior_all_tf, bkg_val_G, bkg_val_R, std_G, std_R)
    
    def preprocessing(self, use_conv=False, normed=True, physical_units=False):
        """ Preprocess values for fitting """
        
        self.use_conv = use_conv
        self.normed = normed
        
        if use_conv:
            data_G = self.data_G_conv.copy()
            data_R = self.data_R_conv.copy()
            
            data_G[self.mask] = np.nan
            data_R[self.mask] = np.nan
            
        else:
            data_G = self.data_G.copy()
            data_R = self.data_R.copy()
            
        val_G = data_G[self.fit_range]
        val_R = data_R[self.fit_range]
        
        if normed:
            self.val_G_mean = np.nanmean(self.image_G[self.fit_range])
            self.val_R_mean = np.nanmean(self.image_R[self.fit_range])
            
            x_G = (val_G - self.val_G_mean)/self.std_G
            x_R = (val_R - self.val_R_mean)/self.std_R
            
            # transform
            self.tf_G = lambda x: (x-self.val_G_mean)/self.std_G
            self.tf_R = lambda x: (x-self.val_R_mean)/self.std_R
            
            # inverse transform
            self.itf_G = lambda x: x*self.std_G + self.val_G_mean
            self.itf_R = lambda x: x*self.std_R + self.val_R_mean
            
        else:
            
            if physical_units: # convert to kJy/arcsec^2
                self.ZP_G_new = self.ZP_G + 2.5 * np.log10(self.scale_factor**2)
                self.ZP_R_new = self.ZP_R + 2.5 * np.log10(self.scale_factor**2)
                pixel_scale = self.pixel_scale
                
                # transform
                self.tf_G = partial(ADU_to_kJy, ZP=self.ZP_G_new, pixel_scale=pixel_scale)
                self.tf_R = partial(ADU_to_kJy, ZP=self.ZP_R_new, pixel_scale=pixel_scale)
                self.itf_G = partial(kJy_to_ADU, ZP=self.ZP_G_new, pixel_scale=pixel_scale)
                self.itf_R = partial(kJy_to_ADU, ZP=self.ZP_R_new, pixel_scale=pixel_scale)
                
                x_G = self.tf_G(val_G)
                x_R = self.tf_R(val_R)
                
            else:
                x_G = val_G
                x_R = val_R  
                
                self.tf_G = self.itf_G = lambda x: x.copy()
                self.tf_R = self.itf_R = lambda x: x.copy()
        
        self.x_G = x_G
        self.x_R = x_R
        
    
    def fit(self, x, y, p0, 
            clip=None, name='1',
            xmin=None, ymin=None,
            xconstr=None, 
            method='Nelder-Mead',
            weights_filter=None,
            **kwargs):
        
        from .fit import Fitter
        start = time.time()
        
        # Set up fitter
        fitter = Fitter(x, y, xconstr, xmin, ymin,
                        clip=clip, weights_filter=weights_filter, name=name)
        fitter.fit_range = self.fit_range
        fitter.normed = self.normed
       
        fitter.setup(**kwargs)
        
        # Run fitting
        fitter.run(p0, method=method)
        
        end = time.time()
        
        if fitter.runned:
            self.__dict__['fitter'+'_'+fitter.name] = fitter
            print('Fitting successfully runned. Time used: {:.1f}s'.format(end-start))
            
    def bootstrap(self, n_sample, bootnum=100, seed=6946):
        """ Return bootstapped indice of n_sample. """
        index = np.arange(n_sample, dtype=int)
        
        # Bootstrap
        with NumpyRNGContext(seed):
            bootresult = bootstrap(index, bootnum).astype(int)
            
        self.bootresult = bootresult
        return bootresult
        
    def fit_with_uncertainty(self, x, y, p0, 
                             clip=None, name='1',
                             xmin=None, ymin=None,
                             xconstr=None, 
                             method='Nelder-Mead',
                             bootnum=100,
                             new_bootstrap=True,
                             seed=6946,
                             **kwargs):
        """ Fitting with uncertainties. """
        from .fit import Fitter
        print('Run fitting with uncertainties from bootstrap (%d times)...'%bootnum)
        
        if (~new_bootstrap) & hasattr(self, 'bootresult'):
            bootresult = self.bootresult
        else:
            bootresult = self.bootstrap(len(x), bootnum, seed)
            
        params_fit = np.zeros((bootnum,len(p0)))
        
        start = time.time()
        
        with tqdm(total=bootnum) as pbar:
            for k, ind in tqdm(enumerate(bootresult)):
                x_, y_ = x[ind], y[ind]

                # Set up fitter
                fitter = Fitter(x_, y_, xconstr, xmin, ymin, clip, name)
                fitter.fit_range = self.fit_range
                fitter.normed = self.normed

                fitter.setup(verbose=False, **kwargs)

                # Run fitting
                fitter.run(p0, method=method, verbose=False)

                params_fit[k] = fitter.params_fit

                pbar.update(1)
        
        end = time.time()
        
        if fitter.runned:
            print('Fitting with uncertainty successfully runned. Time used: {:.1f}s'.format(end-start))
            return params_fit
        else:
            return None

    def processing(self, 
                   f_R2G, f_G2R, 
                   kernel_stddev=(20,3), 
                   median_filter_size=1, 
                   scale_factor=0.5,
                   remove_compact=True, 
                   remove_compact_qantile=0.995,
                   n_threshold=None,
                   kernel_replace_masked=9,
                   background_percentile=50,
                   use_peak=False,
                   use_output='residual',
                   kernel_type='linear',
                   rht_radius=36,
                   name='', 
                   plot=True, 
                   initial_fill=True,
                   fill_mask=False,
                   vlim=[-2,6],
                   figsize=(18, 6)):      
        
        from .utils import remove_compact_emission
        # transform input image
        image_R = self.tf_R(self._image_R)
        image_G = self.tf_G(self._image_G)
        
        if plot:
            fig, (ax1,ax2) = plt.subplots(1, 2, figsize=figsize, dpi=120)
            im=ax1.imshow(image_G, vmin=vlim[0], vmax=vlim[1], cmap='YlGn')
            ax1.set_title('G')
            plt.colorbar(im, ax=ax1)
            im=ax2.imshow(image_R, vmin=vlim[0], vmax=vlim[1], cmap='OrRd')
            ax2.set_title('R')
            plt.colorbar(im, ax=ax2)
            plt.tight_layout()
            plt.savefig(f'tmp/{name}_image.png',dpi=120)
            plt.show()
        
        # Reference input image

        img_ref_R = np.ma.array(self.tf_R(self._image_R), mask=self._mask)
        img_ref_G = np.ma.array(self.tf_G(self._image_G), mask=self._mask)
        
        if (scale_factor != None) and (scale_factor<1):
            # downsampling
            print(f'Downsampling by {scale_factor}...')
            wcs_ds = downsample_wcs(self.wcs, scale=scale_factor)
            shape_out = (int(self._image_shape[0]*scale_factor), int(self._image_shape[1]*scale_factor))
            img_ref_R_ds, _ = reproject_interp((img_ref_R, self.wcs), wcs_ds, shape_out=shape_out, order=1)
            img_ref_G_ds, _ = reproject_interp((img_ref_G, self.wcs), wcs_ds, shape_out=shape_out, order=1)
            mask, _ = reproject_interp((self._mask, self.wcs), wcs_ds, shape_out=shape_out, order=1)
            mask = mask.astype(bool)
            
        else:
            img_ref_R_ds = img_ref_R
            img_ref_G_ds = img_ref_G
            mask = self._mask
            
        self.mask_ds = mask

        # median filtering
        print(f'Meidan filtering by [{median_filter_size}x{median_filter_size}]...')
        img_ref_R_ds_mf = ndimage.median_filter(img_ref_R_ds.copy(), size=median_filter_size, mode='nearest')
        img_ref_G_ds_mf = ndimage.median_filter(img_ref_G_ds.copy(), size=median_filter_size, mode='nearest')
        
        img_ref_R = img_ref_R_ds_mf.copy()
        img_ref_G = img_ref_G_ds_mf.copy()
        
        if remove_compact:
            if plot:
                fig, (ax1,ax2) = plt.subplots(1, 2, figsize=figsize, dpi=150)
                im=ax1.imshow(img_ref_G, vmin=vlim[0], vmax=vlim[1], cmap='YlGn')
                ax1.set_title('G input (raw)')
                plt.colorbar(im, ax=ax1)
                im=ax2.imshow(img_ref_R, vmin=vlim[0], vmax=vlim[1], cmap='OrRd')
                ax2.set_title('R input (raw)')
                plt.colorbar(im, ax=ax2)
                plt.tight_layout()
                plt.savefig(f'tmp/{name}_input.png',dpi=150)
                plt.show()
            
            # remove non-extended emssion
            kws_remove_compact = dict(mask=mask, 
                                      kernel_stddev=kernel_stddev, 
                                      kernel_type=kernel_type,
                                      quantile=remove_compact_qantile, 
                                      rht_radius=rht_radius, 
                                      n_threshold=n_threshold,
                                      kernel_replace_masked=kernel_replace_masked,
                                      use_peak=use_peak,
                                      use_output=use_output,
                                      background_percentile=background_percentile,
                                      fill_mask=fill_mask,
                                      figsize=figsize)
            
            self.img_ref_R_0 = img_ref_R
            self.img_ref_G_0 = img_ref_G
            
            img_ref_R, rht_R = remove_compact_emission(img_ref_R, **kws_remove_compact)
            img_ref_G, rht_G = remove_compact_emission(img_ref_G, **kws_remove_compact)
            
            self.rht = {'R':rht_R, 'G':rht_G}
        
        # upsampling
        if (scale_factor != None) and (scale_factor<1):
            print(f'Upsampling to original grid...')
            img_ref_R, _ = reproject_interp((img_ref_R, wcs_ds), self.wcs, shape_out=self._image_shape, order=1)
            img_ref_G, _ = reproject_interp((img_ref_G, wcs_ds), self.wcs, shape_out=self._image_shape, order=1)
            
        self.img_ref_R = img_ref_R
        self.img_ref_G = img_ref_G
        
        # prediction
        img_pred_R = f_G2R(img_ref_G)
        img_pred_G = f_R2G(img_ref_R)
        
        self.img_pred_R = img_pred_R
        self.img_pred_G = img_pred_G
        
        # leftover
        img_out_R = np.ma.array(image_R - img_pred_R, mask=self._mask)
        img_out_G = np.ma.array(image_G - img_pred_G, mask=self._mask)
        
        self.img_out_R = img_out_R
        self.img_out_G = img_out_G

        if plot:
            fig, (ax1,ax2) = plt.subplots(1, 2, figsize=figsize, dpi=120)
            im=ax1.imshow(img_ref_G, vmin=vlim[0], vmax=vlim[1], cmap='YlGn')
            ax1.set_title('G input')
            plt.colorbar(im, ax=ax1)
            im=ax2.imshow(img_ref_R, vmin=vlim[0], vmax=vlim[1], cmap='OrRd')
            ax2.set_title('R input')
            plt.colorbar(im, ax=ax2)
            plt.tight_layout()
            plt.savefig(f'tmp/{name}_remove_compact.png',dpi=120)

            fig, (ax1,ax2) = plt.subplots(1, 2, figsize=figsize, dpi=120)
            im=ax1.imshow(img_pred_G, vmin=vlim[0], vmax=vlim[1], cmap='YlGn')
            ax1.set_title('G pred')
            plt.colorbar(im, ax=ax1)
            im=ax2.imshow(img_pred_R, vmin=vlim[0], vmax=vlim[1], cmap='OrRd')
            ax2.set_title('R pred')
            plt.colorbar(im, ax=ax2)
            plt.tight_layout()
            plt.savefig(f'tmp/{name}_predict.png',dpi=120)

            fig, (ax1,ax2) = plt.subplots(1, 2, figsize=figsize, dpi=120)
            im=ax1.imshow(img_out_G, vmin=vlim[0], vmax=vlim[1], cmap='YlGn')
            ax1.set_title('G result')
            plt.colorbar(im, ax=ax1)
            im=ax2.imshow(img_out_R, vmin=vlim[0], vmax=vlim[1], cmap='OrRd')
            ax2.set_title('R result')
            plt.colorbar(im, ax=ax2)
            plt.tight_layout()
            plt.savefig(f'tmp/{name}_result.png',dpi=120)
            
            plt.show()
        
        
        
def ADU_to_kJy(x, ZP, pixel_scale):
    return 3.631 * x / (10**(ZP/2.5)) * ((180/np.pi*3600)**2) / pixel_scale**2

def kJy_to_ADU(x, ZP, pixel_scale):
    return x * (10**(ZP/2.5)) / ((180/np.pi*3600)**2) * pixel_scale**2 /3.631 
