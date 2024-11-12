import os
import yaml
import shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clip, mad_std
from astropy.convolution import interpolate_replace_nans, Gaussian2DKernel, convolve_fft
from astropy.utils.decorators import lazyproperty

from photutils.background import MADStdBackgroundRMS
bkgrms = MADStdBackgroundRMS()

def run_mrf(lores_mri_fn, hires_mri_fn, 
            lores_colcorr_fn, hires_colcorr_fn,
            band, band_corr,
            zpt_lores=27.5, zpt_hires=22.5,
            mrf_config_fn='mrf_config.yml', 
            mrfout_dir='', suffix=''):
    
    try:
        import mrf.pipelines as pipe
        from mrf import MRImage, load_config
        from mrf.image import proc
        
    except ModuleNotFoundError:
        print("MRF not installed! Functions running MRF are not available. ")
    
    # Full config.
    config = load_config(mrf_config_fn)
    
    # Primary `MRImage` object we want to perform MRF on.
    mri = MRImage(lores_mri_fn, hires_mri_fn, band=band, zpt_lores=zpt_lores, zpt_hires=zpt_hires)

    # Update filenames in config
    mri_image_kws = dict(band=band, lores_file_name=lores_mri_fn, hires_file_name=hires_mri_fn)
    
    # colcorr_image_kws = dict(band=band_corr, lores_file_name=lores_colcorr_fn, hires_file_name=hires_colcorr_fn)
    colcorr_image_kws = dict(band=band_corr, lores_file_name=None, hires_file_name=None)
    
    config['multi_resolution_image'].update(mri_image_kws)
    config['color_correction_image'].update(colcorr_image_kws)
        
    fn_mrf_config_run = os.path.join(mrfout_dir, f'mrf_config_run_{band}.yml')
    with open(fn_mrf_config_run, 'w') as yaml_file:
        
        config['output_path'] = mrfout_dir
        yaml_file.write(yaml.dump(config))

    # The pipeline functions take the same config as the command-line mrf script.
    mri = pipe.prepare_mri(mri, fn_mrf_config_run, mri_color_correction=None)
    
    # This pipeline takes the MRImage from the previous pipeline as input.
    mri = pipe.model_low_resolution_image(mri, config)
    
    # Save upsampled lores model and residual
    mri.lores_model.write(os.path.join(mrfout_dir, f'_lowres_model_{band}_3{suffix}.fits'))
    mri.residual.write(os.path.join(mrfout_dir, f'residual_{band}{suffix}.fits'))

    # Save lores model at original resolution
    lores_model_1, header_1 = proc.magnify(mri.lores_model.pixels, -3, mri.lores_model.header)
    fits.writeto(os.path.join(mrfout_dir, f'_lowres_model_{band}_1{suffix}.fits'), data=lores_model_1, header=header_1, overwrite=True)

    # Move brightstar.xy
    fn_brightstar_src = os.path.join(mrfout_dir, f'brightstar.xy')
    fn_brightstar_dst = os.path.join(mrfout_dir, f'brightstar_{band}.xy')
    if os.path.exists(fn_brightstar_src):
        shutil.move(fn_brightstar_src, fn_brightstar_dst)
        
    print('Done!')
    
    return mri
    
    
def run_wide_subbright(fn_mrf_residual, 
                       bounds, 
                       obj_name, 
                       band,
                       fn_mrf_model=None,
                       mag_threshold=[15,12.5],
                       mag_limit=None,
                       ZP=27.5, 
                       r_montage=10,
                       r_middle=6,
                       r_halo=20,
                       psf_middle_size=121,
                       threshold=5, 
                       mask_mrf_rms=10, 
                       mask_filled_rms=10,
                       mask_data=None,
                       config_file='elderflower_config.yml',
                       draw=True, work_dir='wide_psf/',
                       subbright_dir='subbright/'):
    try:
        from mrf.wide.io import load_pickle
        from mrf.wide.stack import montage_psf_image
        from mrf.wide.plotting import AsinhNorm, LogNorm
        from mrf.wide.task import berry
        from mrf.subbright.run import allsubbright as run_subbright
        
    except ModuleNotFoundError:
        print("MRF not installed! Functions running MRF are not available. ")
        
    # Get image (mrf output)
    image_mrf = fits.getdata(fn_mrf_residual)
    data_clip = sigma_clip(image_mrf[~np.isnan(image_mrf)], sigma_lower=3, sigma_upper=3, stdfunc=mad_std, maxiters=10)
    bkgval = np.around(np.ma.median(data_clip), 5)
    
    image_mrf_copy = image_mrf.copy()
    header_mrf = fits.getheader(fn_mrf_residual)
    
    # rms
    bkgrms = MADStdBackgroundRMS()
    rms = bkgrms(image_mrf)
    print("RMS = %.4g"%rms)
    
    # Mask map
    if mask_data is None:
        mask_data = np.zeros_like(fn_mrf_residual, dtype=bool)
    
    if fn_mrf_model is not None:
        # mask bright cores and holes of mrf models
        mrf_model = fits.getdata(fn_mrf_model)
        mask_mrf = (mrf_model > mask_mrf_rms * rms) |  (image_mrf < bkgval - mask_mrf_rms * rms)
    else:
        print(f'MRF model is not given, mask {mask_mrf_rms} sigma above the mean.')
        mask_mrf = image_mrf > bkgval + mask_mrf_rms * rms
    
    # dilation
    mask_mrf = ndimage.binary_dilation(mask_mrf)
    
    # set mask as nan avlue
    image_mrf_copy[mask_mrf] = np.nan

    # Fill nan values
    kernel = Gaussian2DKernel(3)
    image_mrf_copy = interpolate_replace_nans(image_mrf_copy, kernel=kernel, convolve=convolve_fft)

    # Set input masked area back to nan
    image_mrf_copy[mask_data] = np.nan

    # Save masked image
    fn_input = os.path.join(work_dir, f'{obj_name}_residual-{band.lower()}_intp.fits')  
    fits.writeto(fn_input, data=image_mrf_copy, header=header_mrf, overwrite=True)

    # Reset BACKVAL
    print("Set new BACKVAL = %.3f"%bkgval)
    fits.setval(fn_input, 'BACKVAL', value=bkgval)

    # Wide PSF modeling class
    elder = berry(fn_input, 
                  bounds, obj_name, band,
                  work_dir=work_dir, 
                  config_file=config_file)

    elder.config.update({'mag_threshold':mag_threshold})
    if mag_limit is not None:
        elder.config.update({'mag_limit':mag_limit})

    # Max / min mag for stacking saturated stars to get middle PSF
    mag_saturate = elder.config['mag_saturate']
    mag_min, mag_max = mag_saturate - 3.5, mag_saturate - 0.5

    # Positions of stars that are not subtracted in MRF
    bright_star_catalog = os.path.join(subbright_dir, f'brightstar_{band}.xy')
    
    
    ### ====== 1. Wide PSF Modeling ====== ###

    range_str = f'X[{bounds[0]}-{bounds[2]}]Y[{bounds[1]}-{bounds[3]}]'

    # run detection & fitting
    elder.detection(threshold=threshold)  # Source detection
    elder.run(bkg=bkgval, 
              draw=draw, 
              clean_measure=False)  # Run fitting

    # Retrieve image and bright star model
    sampler = elder.samplers[0]   # the 1st sampler corresponding to the 1st bounds.
    image, image_stars = sampler.image, sampler.image_stars

    # Read halo model
    psf_fit = sampler.psf_fit

    # Draw halo model
    image_psf = psf_fit.make_image_2D(psf_range=1200) # psf_range in arcsec

    # Montage
    fn_psf_stack = os.path.join(work_dir, f'Measure-PS1/{obj_name}-{band.lower()}-psf_stack_{range_str}.fits')
    psf_stack = fits.getdata(fn_psf_stack)
    psf_montage = montage_psf_image(psf_stack, image_psf, r=r_montage)

    # Save wide PSF model to fits
    fn_halo_image = os.path.join(work_dir, f'PSF_model_{band.lower()}_wide_mrf-{obj_name}.fits')
    psf_fit.write_psf_image(psf_montage, filename=fn_halo_image)
    
    
    ### ====== 2. Subbright ====== ###
    
    # pixel scale 
    pixel_scale = elder.config['pixel_scale']

    # norm radius in wide PSF modeling
    r_scale = elder.config['r_scale']

    # Magninitude limit for drawing with subbright.run
    mag_limit = elder.config['mag_limit']
    mag_str = str(mag_limit).replace('.', 'p')

    # kws for subbright
    subbbright_kws = dict(ZP=ZP, 
                          mag_limit=mag_limit,
                          mag_range=[mag_min, mag_max],
                          r_middle=r_middle, r_halo=r_halo, 
                          psf_middle_size=psf_middle_size,
                          sep_catalog=None, 
                          subbright_dir=subbright_dir,
                          draw_fit=False, verbose=True)

    # Thumbnail images
    res_thumb = load_pickle(os.path.join(work_dir, f'Measure-PS1/{obj_name}-thumbnail_{band.lower()}mag{mag_str}_{range_str}.pkl'))

    # Table of thumbnails for stacking bright saturated stars
    table_norm = Table.read(os.path.join(work_dir, f'Measure-PS1/{obj_name}-norm_{r_scale}pix_{band.lower()}mag{mag_str}_{range_str}.txt'), format='ascii')

    # PAN-STARRS catalog
    table_catalog = Table.read(os.path.join(work_dir, f'Measure-PS1/{obj_name}-catalog_PS_{band.lower()}_all.txt'), format='ascii')
    table_catalog = table_catalog[table_catalog['MAG_AUTO_corr'] < 20] # exclude very faint stars in PAN-STARRS catalog
    
    # Run bright star subtraction 
    mask_map = mask_mrf | mask_data
    image_stars, image_brightstars, image_subbright = run_subbright(fn_mrf_residual, 
                                                                    fn_halo_image, 
                                                                    bounds, 
                                                                    res_thumb, 
                                                                    table_norm, 
                                                                    table_catalog, 
                                                                    bright_star_catalog,
                                                                    mask_map=mask_map,
                                                                    method='PSF',
                                                                    **subbbright_kws)
     
    # Mask based on threshold of star models
    res_subbright = (image_mrf - image_stars).copy()
    res_subbright[mask_data] = np.nan
    mask_filled = image_stars >  mask_filled_rms*rms
    
    if fn_mrf_model is not None:
         mask_filled = mask_filled | mask_mrf
    
    mask_filled = ndimage.binary_dilation(mask_filled)
    
    res_subbright_ma = res_subbright.copy()
    res_subbright_ma[mask_filled] = np.nan
    
    kernel = Gaussian2DKernel(5)
    res_subbright_filled = interpolate_replace_nans(res_subbright_ma, kernel=kernel, convolve=convolve_fft)
        
    if draw:

        # Display
        fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4, figsize=(24,6), dpi=100)

        plot_kws = dict(norm=AsinhNorm(0.5, vmin=bkgval-2*rms, vmax=bkgval+4*rms), cmap='gray_r', origin='lower')
        plot_kws2 = dict(norm=AsinhNorm(0.5, vmin=0, vmax=6*rms), cmap='gray_r', origin='lower')

        ax1.imshow(image_mrf, **plot_kws)
        ax2.imshow(image_stars, **plot_kws2)
        ax3.imshow(res_subbright, **plot_kws)
        ax4.imshow(res_subbright_filled, **plot_kws)

        for ax in (ax1,ax2,ax3,ax4):
            ax.tick_params(size=1, labelsize=0)

        ax1.set_title(f'MRF Image {band.lower()}')
        ax2.set_title(f'Bright Models {band.lower()}')
        ax3.set_title(f'Residual {band.lower()}')
        ax4.set_title(f'Residual {band.lower()} filled')

        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.0, hspace=0.0)
        plt.show()
        
    # Set header BACKVAL
    header_mrf['BACKVAL'] = 0
    
    # Save star models
    fn_output = os.path.join(subbright_dir, f'{obj_name}_starmods-{band.lower()}.fits') 
    fits.writeto(fn_output, data=image_stars, header=header_mrf, overwrite=True)
    
    # Save outputs
    fn_output = os.path.join(subbright_dir, f'{obj_name}_subbright_residual-{band.lower()}.fits') 
    fits.writeto(fn_output, data=res_subbright, header=header_mrf, overwrite=True)
    
    fn_output = os.path.join(subbright_dir, f'{obj_name}_subbright_residual-{band.lower()}_filled.fits') 
    fits.writeto(fn_output, data=res_subbright_filled, header=header_mrf, overwrite=True)

    return image_stars, image_brightstars, image_subbright


class MRF_Result:
    def __init__(self, hdu_path,
                 obj_name, band,
                 mrfout_dir='mrfout/', 
                 subbright_dir='subbright/', suffix=''):
        
        """

        A class storing MRF outputs.

        Parameters
        ----------

        hdu_path : str
            path of hdu data
        obj_name : str
            object name
        band : str
            filter name

        """
        
        self.obj_name = obj_name
        self.band = band.lower()
        self.mrfout_dir = mrfout_dir
        self.subbright_dir = subbright_dir
        self.suffix = suffix
        
        with fits.open(hdu_path) as hdul:
            self.image = hdul[0].data
            self.header = hdul[0].header
    
    @lazyproperty
    def mrf_residual(self):
        mrf_residual = fits.getdata(os.path.join(self.mrfout_dir, f'residual_{self.band}{self.suffix}.fits'))
        return mrf_residual
        
    @lazyproperty
    def mrf_models(self):
        mrf_models = fits.getdata(os.path.join(self.mrfout_dir, f'_lowres_model_{self.band}_1{self.suffix}.fits'))
        return mrf_models
    
    @property
    def brightstar_models(self):
        brightstar_models = fits.getdata(os.path.join(self.subbright_dir, f'{self.obj_name}_starmods-{self.band}.fits'))
        
        # shift the bright stars model by 0.25
        from scipy.ndimage import shift
        is_nan = np.isnan(brightstar_models)
        brightstar_models[is_nan] = 0
        brightstar_models_shifted = shift(np.ma.array(brightstar_models, mask=is_nan), [0.25, 0.25], cval=0.0, order=3, mode='nearest')
        brightstar_models_shifted[is_nan] = np.nan
        
        return brightstar_models_shifted
    
    @property
    def allstar_models(self):
        return self.mrf_models + self.brightstar_models
    
    @lazyproperty
    def residual(self):
        fn_residual = os.path.join(self.subbright_dir, f'{self.obj_name}_subbright_residual-{self.band}.fits')
        image_residual = fits.getdata(fn_residual)
        self.header_residual = fits.getheader(fn_residual)
        return image_residual
    
    @lazyproperty
    def rms(self):
        rms = bkgrms(self.residual)
        return rms
    
    @property
    def residual_ma(self, *args, **kwargs):
        return self.get_residual_ma(*args, **kwargs)
    
    def get_residual_ma(self, mask_rms=5, mask_val=0): 
        self.mask_rms = mask_rms
        self.mask = (self.allstar_models > mask_rms*self.rms) | (self.image==mask_val)
        vals = self.residual[~self.mask]
        # self.bkgval = 2.5*np.nanmedian(vals) - 1.5*np.nanmean(vals)
        self.bkgval = np.nanmedian(vals)
        
        self._residual_ma = np.ma.array(self.residual, mask=self.mask)
        return self._residual_ma

    
    def display(self, mask_rms=5, mask_val=0, lower=2, upper=4, asinh=0.5):
        
        residual_ma = self.get_residual_ma(mask_rms=mask_rms, mask_val=0)
        rms, bkgval = self.rms, self.bkgval
        
        fig, ((ax1,ax2,ax3)) = plt.subplots(1,3, figsize=(11,5), dpi=100)

        plot_kws = dict(vmin=bkgval-lower*rms, vmax=bkgval+upper*rms, cmap='gray_r', origin='lower')
        plot_kws2 = dict(vmin=0, vmax=(lower+upper)*rms, cmap='gray_r', origin='lower')

        ax1.imshow(self.image, **plot_kws)
        ax2.imshow(self.allstar_models, **plot_kws2)
        ax3.imshow(residual_ma, **plot_kws)

        for ax in (ax1,ax2,ax3):
            ax.tick_params(size=1, labelsize=0)

        ax1.set_title(f'Image {self.band}')
        ax2.set_title(f'All Star Models {self.band}')
        ax3.set_title(f'Residual {self.band}')

        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.0, hspace=0.0)
        plt.show()