import os
import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
import astropy.units as u

from astropy.coordinates import Galactic
from reproject import reproject_from_healpix

from .plotting import display

# Pixel scale (arcsec/pixel) for reduced and raw Dragonfly data
DF_pixel_scale = 2.5
DF_raw_pixel_scale = 2.85


class PlanckImage:
    
    def __init__(self, hdu_path):
    
        with fits.open(hdu_path) as hdul: 
            hdu = hdul[1]
            self.header = hdul[1].header

            self.radiance = hdu.data['RADIANCE']
            self.tau = hdu.data['TAU353']
            self.ebv = hdu.data['EBV']
            self.temp = hdu.data['TEMP']
            self.beta = hdu.data['BETA']

            hdul.close()
            
        self.data = dict(radiance=self.radiance, tau=self.tau, EBV=self.ebv, temp=self.temp, beta=self.beta)
    
    def reproject(self, wcs, shape, model='radiance'):
        """ Make a map from Healpix data """
        
        dust_map = self.data.get(model)
        
        if dust_map is not None:
            dust_map_rp, _ = reproject_from_healpix((dust_map, Galactic()), output_projection=wcs, shape_out=shape, nested=True)
            return dust_map_rp
        
class IRISImage:
    
    def __init__(self, hdu_path=None):
        
        self.hdu_path = hdu_path
    
        with fits.open(hdu_path) as hdul: 
            hdu = hdul[1]
            self.header = hdul[1].header
            self.data = hdu.data

            hdul.close()
                
    def reproject(self, wcs, shape):
        """ Make a map from Healpix data """
        
        if self.hdu_path is not None:
            dust_map_rp, _ = reproject_from_healpix(self.hdu_path, output_projection=wcs, shape_out=shape)
            return dust_map_rp
        
    
class ImageButler:
    """
    
    A butler storing Image class for two bands.
    
    Parameters
    ----------
    
    hdu_path_G : str
        path of hdu g-band data
    hdu_path_R : str
        path of hdu r-band data
    obj_name : str
        object name
    pixel_scale : float
        pixel scale in arcsec/pixel
    
    """
    
    def __init__(self, 
                 hdu_path_G, 
                 hdu_path_R, 
                 obj_name='',
                 pixel_scale=DF_pixel_scale):
        
        self.hdu_path_G = hdu_path_G
        self.hdu_path_R = hdu_path_R
        self.Image_G = Image(hdu_path_G, 'G', obj_name, pixel_scale)
        self.Image_R = Image(hdu_path_R, 'R', obj_name, pixel_scale)
        self.wcs = self.Image_G.wcs
            
    def __str__(self):
        return "An ImageButler class"

    def __repr__(self):
        return f"{self.__class__.__name__}"
    
    def make_cutout(self, coord=None, shape=None):
        """ 
        Make image cutouts.
        Image stored as self.Image_G.data_cutout and self.Image_R.data_cutout
        
        coord: SkyCoord
        shape: array or tuple
        
        """
        if coord is None:
            coord = self.Image_G.get_center_coord()
        if shape is None:
            shape = self.Image_G.image.shape
            
        self.Image_G.make_cutout(coord, shape)
        self.Image_R.make_cutout(coord, shape)
        
        self.coord_cutout = coord
        self.shape_cutout = self.Image_G.image_cutout.shape
        
        if self.shape_cutout != shape[::-1]:
            print('Cutout shape exceeds FoV!')
        
        self.wcs_cutout = self.Image_G.cutout.wcs
        self.bounds_cutout = self.Image_G.bounds_cutout
    
    def apply_cutout(self, data):
        if hasattr(self, 'wcs_cutout'): 
            cutout = Cutout2D(data, self.coord_cutout, self.shape_cutout, wcs=self.wcs)
            return cutout.data
    
    def make_RGB(self, image_G, image_R,
                 mask=None, plot_kws={},
                 method='interpolate',**kws):
        from astropy.visualization import make_lupton_rgb
        img_R = np.ma.array(image_R, mask=mask).filled(np.nan)
        img_G = np.ma.array(image_G, mask=mask).filled(np.nan)
        if method=='interpolate':
            img_rgb = make_lupton_rgb(img_R, (img_R+img_G)/2., img_G, **kws)
        elif method=='extrapolate':
#            img_rgb = make_lupton_rgb(img_R, img_G, (1.5*img_G - img_R), **kws)
            img_rgb = make_lupton_rgb(img_R, img_G, (2*img_G - 0.5*img_R), **kws)
        self.img_rgb = img_rgb
        
        display(img_rgb, **plot_kws)
        
        

        
class Image:
    """
    
    A class storing Image info and outputs.
    
    Parameters
    ----------
    
    hdu_path : str
        path of hdu data
    band : str
        filter name
    obj_name : str
        object name
    pixel_scale : float
        pixel scale in arcsec/pixel
    
    """
    
    def __init__(self, hdu_path, band, obj_name='',
                 pixel_scale=DF_pixel_scale):
        
        self.obj_name = obj_name
        self.band = band
        self.pixel_scale = pixel_scale
        
        self.norm = self.norm0 = None # for plotting
        
        assert os.path.isfile(hdu_path), "File do not exist. Check hdu paths."
            
        with fits.open(hdu_path) as hdul:
            self.hdu_path = hdu_path
            self.header = header = hdul[0].header
            self.wcs = WCS(header)
            self.image = hdul[0].data
            hdul.close()
            
        self.shape = self.image.shape
        self.ZP = header.get('REFZP')
            
    def __str__(self):
        return "An Image class"

    def __repr__(self):
        return f"{self.__class__.__name__} for {self.hdu_path}"
    
    def display(self, **kwargs):
        """ Display the image """
        display(self.image, **kwargs)
    
    def get_center_coord(self, unit=(u.deg, u.deg)):
        """ Get center coordinates of the field """
        from astropy.coordinates import SkyCoord
        # self.center_coord = SkyCoord(self.header['CRVAL1'], self.header['CRVAL2'], unit=unit)
        coord_ref = self.wcs.all_pix2world((self.shape[1]-1)/2., (self.shape[0]-1)/2., 1)
        self.center_coord = SkyCoord(coord_ref[0], coord_ref[1], unit=(u.deg, u.deg))

        return self.center_coord
        
    def make_cutout(self, coord, shape):
        """ 
        Make image cutouts.
        Image stored as self.data_cutout
        
        coord: SkyCoord
        shape: array or tuple
        
        """

        LX, LY = shape
        cutout = Cutout2D(self.image.copy(), coord, (LY, LX), wcs=self.wcs)
        
        self.cutout = cutout
        self.image_cutout = cutout.data.copy()
        self.wcs_cutout = cutout.wcs
        bbox = cutout.bbox_original
        self.bounds_cutout = np.array([bbox[1][0], bbox[0][0], bbox[1][1]+1, bbox[0][1]+1])
        
    def display_cutout(self, **kwargs):
        """ Display the cutout image """
        if hasattr(self, 'image_cutout'):
            display(self.image_cutout, **kwargs)
    
