import os
import time
import numpy as np
from numpy.polynomial import Polynomial

from scipy import stats
from scipy.special import expit

import matplotlib.pyplot as plt
from skimage.transform import rescale

from astropy.io import fits
import astropy.units as u
from astropy.table import Table
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.modeling import models

# from cap_mpfit import mpfit

def linear(x, A, B):
    return A * x + B

def make_legendre2d_grids(shape):
    from numpy.polynomial.legendre import leggrid2d
    Ximage_size, Yimage_size = shape
    Xgrid = np.linspace(-(1-1/Ximage_size)/2., (1-1/Ximage_size)/2., Ximage_size)
    Ygrid = np.linspace(-(1-1/Yimage_size)/2., (1-1/Yimage_size)/2., Yimage_size)
    P10 = leggrid2d(Xgrid, Ygrid, c=[[0,1],[0,0]])
    P01 = leggrid2d(Xgrid, Ygrid, c=[[0,0],[1,0]])
    P11 = leggrid2d(Xgrid, Ygrid, c=[[0,0],[0,1]])
    P20 = leggrid2d(Xgrid, Ygrid, c=[[0,0,1],[0,0,0], [0,0,0]])
    P02 = leggrid2d(Xgrid, Ygrid, c=[[0,0,0],[0,0,0], [1,0,0]])
    
    return P10, P01, P11, P20, P02

#### Prior ####

def prior_cirrus_tf(u,
                    bkg_val_G=0, bkg_val_R=0, 
                    std_G=3, std_R=4, 
                    fix_bkg=False, fit_bkg2d=False):
    v = u.copy()
    v[0] = stats.uniform(loc=0.4, scale=0.6).ppf(u[0])       # m
    v[1] = stats.uniform(loc=0.01, scale=0.19).ppf(u[1])     # s

    if not fix_bkg:
        v[2] = stats.uniform(loc=bkg_val_G-std_G, scale=2*std_G).ppf(u[2])       # mu_g
        v[3] = stats.uniform(loc=bkg_val_R-std_R, scale=2*std_R).ppf(u[3])      # mu_r

    if fit_bkg2d:
        for i in range(5):
            v[4+i] = stats.uniform(loc=-2, scale=4).ppf(u[4+i])        # A
    
    return v

def prior_all_tf(u, 
                 bkg_val_G=0, bkg_val_R=0, 
                 std_G=3, std_R=4, 
                 fix_bkg=False, fit_bkg2d=False):
    v = u.copy()
    v[0] = stats.uniform(loc=0.4, scale=0.8).ppf(u[0])        # f_mean
    v[1] = stats.uniform(loc=0.01, scale=0.99).ppf(u[1])      # f_std
    
    v[2] = stats.uniform(loc=0.1, scale=0.9).ppf(u[2])        # b_mean
    v[3] = stats.uniform(loc=0.01, scale=0.99).ppf(u[3])      # b_std
    v[4] = stats.uniform(loc=0, scale=1).ppf(u[4])            # b_frac
    
    if not fix_bkg:
        v[5] = stats.uniform(loc=bkg_val_G-std_G, scale=2*std_G).ppf(u[5])      # mu_g
        v[6] = stats.uniform(loc=bkg_val_R-std_R, scale=2*std_R).ppf(u[6])      # mu_r (can be constrained conditionally from f1/f2 ?)
    
    if fit_bkg2d:
        for i in range(5):
            v[7+i] = stats.uniform(loc=-2, scale=4).ppf(u[7+i])        # A_ij

    return v



#### Loglikelihood ####

def eval_likelihood_poly(params, xdata, ydata, poly_deg,
                         sigmoid=True, a=0.2, 
                         legendre_terms=None):
    """ Likelihood function for polynoimal modeling """
    # read parameters
    coefs = params[:poly_deg+1]
    eps = params[poly_deg+1]
    
    # 1d polynomials
    poly = Polynomial(coefs[::-1])
    
    if sigmoid:
        # smoothing with sigmoid function
        x0 = params[poly_deg+2]
        h = lambda x: expit((x-x0)/a)
        ypred = poly(xdata) * h(xdata) + (1-h(xdata)) * poly(x0-a)
    else:
        ypred = poly(xdata)
    
    if legendre_terms is not None:
        # background terms
        coefs_bkg = params[-5:]
        bkg2d = np.sum([A*P for (A, P) in zip(coefs_bkg, legendre_terms)], axis=0)
        ypred += bkg2d
    
    prob = np.exp( -0.5 * (ydata-ypred)**2 / eps**2) / eps
    
    return -np.sum(np.log(prob))
    
def eval_likelihood_poly_mix_constr(params, xdata, ydata, poly_deg,
                                    xconstr=None, penalty=1e-2, 
                                    sigmoid=False, a=0.2,
                                    weights_filter=[1,1],
                                    legendre_terms=None,
                                    give_prob=False):
    
    """ Likelihood function for mixture populations of polynoimal models with linear constraint """
    
    # number of variable dimensions
    ndim = np.ndim(xdata)
    if ndim > 1:
        xdata = xdata.T
        
    # read parameters
    coefs = params[:poly_deg+1]
    eps = params[poly_deg+1]
    
    if ndim > 1:
        coefs = np.delete(params[:(poly_deg+1) * ndim + 1], poly_deg+1)
        coefs = np.array(coefs).reshape(ndim, poly_deg+1)
    
    # j is index of the outlier population
    j = (poly_deg+1) * ndim + 1
    if sigmoid: j += 1
    
    mu, sigma, frac = params[j:j+3]
    
    if (frac<=0.5) or (frac>1) or (eps<=0) or (sigma<=eps):
        return np.inf
    
    # 1d polynomials
    if ndim == 1:
        poly = Polynomial(coefs[::-1])
    else:
        polys = [Polynomial(coef_n[::-1]) for coef_n in coefs]
    
    if sigmoid:
        # smoothing with sigmoid function
        x_min = params[j-1]
        h = lambda x: expit((x-x_min)/a)
    else:
        x_min = 0
        h = lambda x: 1
    
    if ndim == 1:
        ypred = poly(xdata) * h(xdata) + (1-h(xdata)) * poly(x_min-a)
    else:
        ypred = np.array([polys[k](xdata[:,k]) * h(xdata[:,k]) + (1-h(xdata[:,k])) * polys[k](x_min-a) for k in range(ndim)])
#        ypred = np.average(ypred, weights=weights_filter, axis=0)
                
    if legendre_terms is not None:
        # background terms
        coefs_bkg = params[-5:]
        bkg2d = np.sum([A*P for (A, P) in zip(coefs_bkg, legendre_terms)], axis=0)
        ydata -= bkg2d
        
    # Note: prob may be lower than float limit, use np.logaddexp instead
    if ndim == 1:
        logprob1 = -0.5 * ((ydata-ypred)/eps)**2 + np.log(frac/ eps)
    else:
        logprob1 = [-0.5 * ((ydata-ypred[k])/eps)**2 + np.log(frac/ eps) for k in range(ndim)]
        logprob1 = np.average(logprob1, weights=weights_filter, axis=0)
    logprob2 = -0.5 * ((ydata-mu)/sigma)**2 + np.log((1-frac)/sigma)
    
    logprob = np.logaddexp(logprob1, logprob2) # length = ydata

    prob1 = np.exp(logprob1)
    prob = np.exp(logprob)
        
    if give_prob:
        return  prob1 / (prob + 1e-9)
    else:
        llh = -np.sum(logprob)  # scalar
        
        if xconstr is not None:
            b1, b0 = params[j+3:j+5]
            # linear constraint
            yconstr  = b1 * xconstr + b0
            
            member  = (prob1/prob) > 0.5
            penalization = np.sqrt((ydata[member]-yconstr[member])**2) * frac * penalty
            
            llh += np.sum(penalization)
            
        return llh
    

def eval_likelihood_piecewise_lin_mix_constr(params, xdata, ydata,
                                             xconstr=None, penalty=1e-2,
                                             sigmoid=False, a=0.2,
                                             weights_filter=[1,1],
                                             legendre_terms=None,
                                             give_prob=False):
    
    """ Likelihood function for mixture populations of piecewise linear models with linear constraint """
    
    # number of variable dimensions
    ndim = np.ndim(xdata)
    if ndim > 1:
        xdata = xdata.T
    
    # ending index of linear models
    ind = 3 * ndim + 2
    
    # read parameters
    A1, x0, y0, A2, eps = params[0:5]
    
    # convert to array
    if ndim > 1:
        A1 = np.atleast_1d([A1] + params[5:ind:3])
        x0 = np.atleast_1d([x0] + params[6:ind:3])
        A2 = np.atleast_1d([A2] + params[7:ind:3])
                
    
    # j is index of the outlier population
    j = ind
    if sigmoid: j += 1

    mu, sigma, frac = params[j:j+3]
    frac0 = 1 - np.sum(frac)
    
    if sigmoid:
        # smoothing with sigmoid function
        x_min = params[ind]
        h = lambda x: expit((x-x_min)/a)
    else:
        x_min =  np.quantile(xdata, 0.0001) if ndim == 1 else np.quantile(xdata[:,0], 0.0001)
        h = lambda x: 1
        
    if frac<=0.5 or frac>1 or eps<=0 or sigma<=eps:
        return np.inf
    
    if (sigmoid) & (np.atleast_1d(x0)<x_min).any(): # set transition point
        return np.inf
    
    ### NEW lines started ###
    # use constraint: A1 * x + B1 = y0
    B1 = y0 - A1 * x0
    B2 = y0 - A2 * x0
    
    f1 = lambda x: A1 * x + B1
    f2 = lambda x: A2 * x + B2
        
    if ndim == 1:
        cond_list = [xdata < x0, xdata >= x0]
        
    else:
        cond_list = [ydata < y0, ydata >= y0]
    
    func_list = [lambda x: f1(x) * h(x) + (1-h(x)) * f1(x_min-a),
                 lambda x: f2(x) * h(x) + (1-h(x)) * f2(x_min-a)]
                 
    ypred = np.piecewise(xdata, cond_list, func_list)
    
    if ndim > 1:
        ypred = np.average(ypred, weights=weights_filter, axis=-1)
    
    if legendre_terms is not None:
        # background terms
        coefs_bkg = params[-5:]
        bkg2d = np.sum([A*P for (A, P) in zip(coefs_bkg, legendre_terms)], axis=0)
        ydata -= bkg2d
        
    # Note: prob may be lower than float limit, use np.logaddexp instead
    logprob1 = -0.5 * ((ydata-ypred)/eps)**2 + np.log(frac/ eps)
    logprob2 = -0.5 * ((ydata-mu)/sigma)**2 + np.log((1-frac)/sigma)
    
    logprob = np.logaddexp(logprob1, logprob2) # length = ydata
    
    ### NEW lines ended ###

    prob1 = np.exp(logprob1)
    prob = np.exp(logprob)
        
    if give_prob:
        return  prob1 / (prob + 1e-9)
    else:
        llh = -np.sum(logprob)  # scalar
        
        if xconstr is not None:
            b1, b0 = params[j+3:j+5]
            # linear constraint
            yconstr  = b1 * xconstr + b0
            
            member  = (prob1/prob) > 0.5
            penalization = np.sqrt((ydata[member]-yconstr[member])**2) * frac * penalty
            
            llh += np.sum(penalization)
            
        return llh
    
def eval_likelihood_multi_lin_mix_constr(params, xdata, ydata, n_model=2,
                                        xconstr=None, penalty=1e-2, 
                                        sigmoid=True, a=0.2, include_noise=True,
                                        legendre_terms=None, give_prob=False):
    
    """ Likelihood function for multiple-mixture populations of linear models with linear constraint """
    
    # ending index of linear models
    ind = 3*n_model
    
    # read parameters
    As = params[:ind:3]
    Bs = params[1:ind:3]
    fracs = params[2:ind:3]
    eps = params[ind]
    
    if include_noise:
        if sigmoid:
            # index of the outlier population
            j = ind+2
        else:
            j = ind+1

        mu, sigma = params[j:j+2]
        frac0 = 1 - np.sum(fracs)
    else:
        fracs[-1] = 1-np.sum(fracs[:-1])
        frac0 = 1e-5
        sigma = 1e-5
    
    if np.any(fracs<=0) or np.any(fracs>=1) or frac0<=0 or eps<=0 or sigma<=eps:
        return np.inf
    
    if sigmoid:
        # smoothing with sigmoid function
        x0 = params[ind+1]
        h = lambda x: expit((x-x0)/a)
    else:
        x0 = 0
        h = lambda x: 1
    
    if legendre_terms is not None:
        # background terms
        coefs_bkg = params[-5:]
        bkg2d = np.sum([A*P for (A, P) in zip(coefs_bkg, legendre_terms)], axis=0)
        ydata -= bkg2d
    
    # probs = np.zeros([n_model+1, len(xdata)])
    logprobs = np.zeros([n_model+1, len(xdata)])
    
    if include_noise:
        # probs[0] = np.exp( -0.5 * (ydata-mu)**2 / sigma**2) / sigma * frac0
        logprobs[0] = -0.5 * ((ydata-mu)/sigma)**2 + np.log(frac0/sigma) 
    else:
        # probs[0] = 0.
        logprobs[0] = 0.
    
    for k, params_k in enumerate(zip(As, Bs, fracs)):
        A_k, B_k, frac_k = params_k
        yfunc = lambda x: A_k * x + B_k
        ypred_k = yfunc(xdata) * h(xdata) + (1-h(xdata)) * yfunc(x0-a)
        # prob_k = np.exp( -0.5 * (ydata-ypred_k)**2 / eps**2) / eps * frac_k
        # probs[k+1] = prob_k
        logprob_k = -0.5 * ((ydata-ypred_k)/eps)**2 + np.log(frac_k/eps)
        logprobs[k+1] = logprob_k   
    
    # prob = np.sum(probs, axis=0)
    logprobs_sum = np.logaddexp.reduce(logprobs, axis=0, dtype=np.float64)   # length = ydata
    
    probs = np.exp(logprobs)  # (n_model, ydata)
    probs_sum = np.exp(logprobs_sum) + 1e-9
        
    if give_prob:
        return probs / probs_sum
    else:
        # llh = -np.sum(np.log(prob))
        llh = -np.sum(logprobs_sum)  # scalar
        
        if include_noise & (xconstr is not None):
            b1, b0 = params[j+3:j+5]
            # linear constraint
            yconstr  = b1 * xconstr + b0
        
            prob1 = probs[np.argmax(fracs)] # main population
            member = prob1 > (probs_sum - prob1)
            
            penalization = np.sqrt((ydata[member]-yconstr[member])**2) * frac * penalty
            llh += np.sum(penalization)
            
        return llh

#def eval_likelihood_piecewise(params, xdata, ydata):
#    a2, a1, x0, y0, eps = params
#    
#    if frac<=0.5 or frac>=1 or sigma<=0 or eps<=0:
#        return np.inf
#
#    b1 = y0 - a1*x0 
#    b2 = y0 - a2*x0
#    
#    ypred =  np.piecewise(xdata, [xdata < x0, xdata >= x0], [lambda x: a1*x + b1, lambda x: a2*x + b2])
#    
#    prob = np.exp( -0.5 * (ydata-ypred)**2 / eps**2) / eps
#    
#    return -np.sum(np.log(prob))


def loglike_hist(v, 
                 data_G, data_R, cond, 
                 bkg_val_G=1e-5, bkg_val_R=1e-5,
                 q_l=0.03, q_u=0.95, 
                 nbin=51, fmin=-1.5, fmax=2.5, 
                 cirrus_only=False, fix_bkg=False, 
                 fit_bkg2d=False, legendre_grids=None):
    
    f_m, f_s = v[0], v[1]   # mean/std of flux ratio for cirrus
    
    if not cirrus_only:        
        b_m, b_s = v[2], v[3]  # mean/std of flux ratio for bkg
        b_frac = v[4]
    
    if fix_bkg:
        mu_g, mu_r = bkg_val_G, bkg_val_R
    else:
        # include background distribution
        j = 0 if cirrus_only else 3
        mu_g, mu_r = v[j+2], v[j+3]

    # fit 2D background with Legendre polynomials
    if fit_bkg2d:
        coeffs = v[j+4:]
        bkg2d_g = np.sum([A*P for (A, P) in zip(coeffs, legendre_grids)], axis=0)
    else:
        bkg2d_g = 0
    
    # final model background
    bkg_g = mu_g + bkg2d_g
    bkg_r = mu_r * (1 + bkg2d_g/mu_g)
    
    # flux ratio 
    f_ratio = ((data_G-bkg_g)/(data_R-bkg_r))[cond].ravel()
    
    # fit range
    f1, f2 = np.quantile(f_ratio, [q_l, q_u])
    f1, f2 = max([f1, fmin]), min([f2, fmax])
    
    if (f2-f1) < 0.1:
        loglike = -1e100
        return loglike
        
    f_range = (f_ratio>f1) & (f_ratio<f2)
    
    # binning
    bins = np.linspace(f1, f2, nbin)
    bin_width = bins[1] - bins[0]
    
    # node for evaluate histogram
    x = 0.5*(bins[1:]+bins[:-1])
    
    # get data histogram
    H = np.histogram(f_ratio, bins=bins, density=True)[0]
    
    # assume poisson noise 
    N = len(f_ratio[f_range])
    H_sigma = np.sqrt(H/(N*bin_width))
    
    # deal with zero bin
    H_sigma[(H_sigma)==0] = np.inf
    not_empty = (H_sigma>0)  # only fit not empty bin
    
    # get cirrus model histogram
    dist = stats.norm(loc=f_m, scale=f_s)
    
    if not cirrus_only:
        # background model histogram
        dist_b = stats.cauchy(loc=b_m, scale=b_s)      # background model histogram
        H_model = (dist.pdf(x) * (1-b_frac) + dist_b.pdf(x) * b_frac) / q_u
    else:
        H_model = dist.pdf(x)/q_u
    
    # reidual and likelihood
    residsq = (((H - H_model)[not_empty])/H_sigma[not_empty])**2
    loglike = -0.5 * np.sum(residsq + np.log(2 * np.pi * H_sigma[not_empty]**2))

    if not np.isfinite(loglike):
        loglike = -1e100
        
    return loglike


#### Modeling ####

def build_color_model(params, data_G, data_R, cond, 
                      nbin=51, fmin=-1., fmax=2, 
                      q_l=0.03, q_u=0.95, bkg_fit_G=0, bkg_fit_R=0,
                      fit_bkg2d=False, fix_bkg=False, legendre_grids=None):

    # read parameters
    f_m, f_s,  = params[0:2]
    b_m, b_s, b_frac = params[2:5]
    
    if not fix_bkg:
        bkg_fit_G, bkg_fit_R = params[5:7]

    if fit_bkg2d:
        coefs = params[-5:]
        bkg2d = np.sum([A*P for (A, P) in zip(coefs, legendre_grids)], axis=0)
        bkg2d_g = bkg_fit_G + bkg2d
        bkg2d_r = bkg2d_g * bkg_fit_R/bkg_fit_G
    else:
        bkg2d_g = bkg_fit_G 
        bkg2d_r = bkg_fit_R

    f_fit = ((data_G-bkg2d_g)/(data_R-bkg2d_r))[cond].ravel()
    
    # fitting range
    f1, f2 = np.quantile(f_fit, [q_l, q_u])
    f1, f2 = max([f1, fmin]), min([f2, fmax])
    f_range = (f_fit>f1) & (f_fit<f2)
    
    # histogram nodes
    bins = np.linspace(f1, f2, nbin)
    x = 0.5*(bins[1:]+bins[:-1])

    H, bins = np.histogram(f_fit, bins=bins, density=True)
    
    # make hitograms
    dist = stats.norm(loc=f_m, scale=f_s)
    dist_b = stats.cauchy(loc=b_m, scale=b_s)
    H_pred = (dist.pdf(x) * (1-b_frac) + dist_b.pdf(x) * b_frac) / q_u
    
    return H, H_pred, bins, dist, dist_b
    
    
def subtract_extended_model(image_target, image_ref, mask, factor, 
                            downsampling=0.25, sigma_conv=3, vmin=0, vmax=100, 
                            smooth_output=False, sigma_smooth=1, plot=True):
    
    ### factor = target/ref
    
    # Smoothing the reference image
    image_ref_ = image_ref.copy()
    image_ref_[mask] = np.nan
    
    Image.image_extend_conv = convolve(image_ref_, Gaussian2DKernel(x_stddev=sigma_conv))
    
    # Downsampling
    Image.image_extend_ds = rescale(Image.image_extend_conv, downsampling)
    
    # Upsamping
    Image.image_extend_us = rescale(Image.image_extend_ds, 1/downsampling)
    
    # Subtract extended component in both images
    Image.image_ref_res = Image.image_ref - Image.image_extend_us
    Image.image_output = image_target - Image.image_extend_us * factor
    
    if smooth_output:
        Image.image_output = convolve(img_out.copy(), Gaussian2DKernel(x_stddev=sigma_smooth),
                                      nan_treatment='fill', preserve_nan=True)
    
    if plot:
        from .plotting import draw_reference_residual
        draw_reference_residual(Image)
