import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from astropy import stats

class Fitter:
    
    """ 
    
    A fitter helper storing fitting inputs and outputs.
    
    """
    
    def __init__(self, x, y,
                 xconstr=None,
                 xmin=None, ymin=None,
                 weights_filter=None,
                 clip=None, name='1'):
        
        self.name = name
        
        # set x in 2d for compatibility
        x = np.atleast_2d(x)
        
        assert x.shape[-1] == y.shape[-1], print("Length of x do not match y!")
        if xconstr is not None:
            assert x.shape[-1] == len(xconstr), print("Length of x do not match xconstr!")
        
        # set domain of x and y
        cond = (~np.isnan(x)) & ~np.isnan(y)
        if xmin is not None:
            cond &= np.greater_equal(x.T, xmin).T
        if ymin is not None:
            cond &= (y>ymin)
        if clip is not None:
            x_lo, x_hi = np.nanquantile(x, [clip, 1-clip], axis=-1)
            clip_lo = np.greater_equal(x.T, x_lo).T
            clip_hi = np.less_equal(x.T, x_hi).T

            cond &= clip_lo & clip_hi
            
        cond = cond.all(axis=0)
        
        self.cond = cond
        self.x, self.y = np.squeeze(x), y
        self.xdata, self.ydata = np.squeeze(x[:,cond]), y[cond]
        self.ndim = np.ndim(self.xdata)
        self.xmin, self.ymin = xmin, ymin
        
        if weights_filter is None:
            self.weights_filter = np.ones(self.ndim)
    
        if xconstr is not None:
            xconstr = xconstr[cond]
        self.xconstr = xconstr 
        
        self.runned = False
        
        # Initialize labels for populations
        labels = -1 * np.ones_like(self.y)
        self.labels = labels
     
    def check_ndim(self, nd=1):
        """ Check dimension of xdata """
        if self.ndim > nd:
            raise Exception("Dimension > {:d} is not supported for this setup!".format(nd))
        else:
            return
                
    def setup(self, 
              n_model=1, 
              poly_deg=2,
              include_noise=True,
              piecewise=False,
              penalty=1e-2,
              sigmoid=False,
              sigmoid_a=0.2,
              fit_bkg2d=False,
              verbose=True):
        
        """ Setup fitting """
        
        from .models import (eval_likelihood_poly,
                             eval_likelihood_poly_mix_constr,
                             eval_likelihood_piecewise_lin_mix_constr,
                             eval_likelihood_multi_lin_mix_constr)
        
        xdata, ydata = self.xdata, self.ydata
        ndim = self.ndim
        
        self.mixture_model = False
        self.piecewise = piecewise
        if piecewise:
            n_model = 2
            poly_deg = 1
            
        # printout message
        if verbose: print("Dimension of variables = {:d}".format(ndim))
        if verbose: print("Fitting Parameters: ")
        if n_model==1:
            msg = ''.join(["a%d={:.4g}, "%(poly_deg-i) for i in range(poly_deg+1)]) + "eps={:.4g}"
            if ndim>1:
                for nd in range(ndim-1):
                    msg += ''.join([", a%d_%d={:.4g}"%(poly_deg-i, nd+1) for i in range(poly_deg+1)])
        elif piecewise:
            msg = ''.join(["A1={:.4g}, x0={:.4g}, y0={:.4g}, A2={:.4g}, "]) + "eps={:.4g}"
            if ndim>1:
                for nd in range(ndim-1):
                    msg += ''.join([", A1_%d={:.4g}, x0_%d={:.4g}, A2_%d={:.4g}"%(nd+1,nd+1,nd+1)])
        else:
            self.check_ndim(1)
            msg = ''.join(["A%d={:.4g}, B%d={:.4g}, frac%d={:.4g}, "%tuple([i+1]*3) for i in range(n_model)]) + "eps={:.4g}"
            self.mixture_model = True
        
        if sigmoid:
            self.sigmoid_a = sigmoid_a
            msg += ", x_min={:.4g}"
        self.sigmoid = sigmoid
            
        if include_noise:
            msg += ", mu={:.4g}, sigma={:.4g}"
            if not self.mixture_model:
                msg += ", frac={:.4g}"
        self.include_noise = include_noise
                
        xconstr = self.xconstr
        if xconstr is not None:
            self.check_ndim()
            msg += ", b1={:.4g}, b0={:.4g}"
        if fit_bkg2d:
            self.check_ndim()
            msg += ", P10={:.4g}, P01={:.4g}, P11={:.4g}, P20={:.4g}, P02={:.4g}"
        
        self.msg = msg
        if verbose: print(f"  [{msg.replace('={:.4g}', '')}]")
        
        if n_model==1:
            n_param = (poly_deg+1) * ndim + 1
            if verbose: print("    - polynomial degree = {}".format(poly_deg))
        elif piecewise:
            n_param = 3 * ndim + 2
        else:
            n_param = 3 * n_model + 1
            if verbose: 
                print("    - # of models : {}".format(n_model))
                print("    - Only poly_deg = 1 available for mixture models.")
            poly_deg = 1
         
        self.n_model = n_model
        self.poly_deg = poly_deg
    
        if fit_bkg2d:
            from .models import make_legendre2d_grids
            # make legendre grids
            self.legendre_grids = make_legendre2d_grids(self.image_shape)
            legendre_terms = [grid[self.fit_range].ravel()[cond] for grid in self.legendre_grids]
            n_param += 5  # add A10, A01, A11, A20, A02
            if verbose:  print("    - Fit Legendre 2D background.")
        else:
            legendre_terms = None
            
        self.legendre_terms = legendre_terms
        
        fit_kwargs = dict(xconstr=xconstr, 
                          sigmoid=sigmoid, 
                          a=sigmoid_a,
                          penalty=penalty, 
                          legendre_terms=legendre_terms,
                          weights_filter=self.weights_filter,
                          give_prob=False)
        
        self.penalty = penalty
        self.fit_kwargs = fit_kwargs
        
        if sigmoid: n_param += 1  # add x_min
        
        if include_noise:
            # mixtue model including outliers
            n_param +=2   # add mu, sigma
            
            if xconstr is not None:
                if verbose: print("    - Penelized fitting with linear constraint.")
                n_param += 2 # add c, d
            
            if n_model==1:
                n_param +=1   # add frac
                p_eval_lld = partial(eval_likelihood_poly_mix_constr, 
                                     xdata=xdata, ydata=ydata, 
                                     poly_deg=poly_deg,
                                     **fit_kwargs)
                
            elif piecewise:
                n_param +=1   # add frac
                p_eval_lld = partial(eval_likelihood_piecewise_lin_mix_constr,
                                     xdata=xdata, ydata=ydata,
                                     **fit_kwargs)
            else:
                p_eval_lld = partial(eval_likelihood_multi_lin_mix_constr, 
                                     xdata=xdata, ydata=ydata, 
                                     n_model=n_model, 
                                     **fit_kwargs)
            
        else:
            if n_model==1:
                p_eval_lld = partial(eval_likelihood_poly, 
                                     xdata=xdata, ydata=ydata,
                                     poly_deg=poly_deg, 
                                     sigmoid=sigmoid, a=a, 
                                     legendre_terms=legendre_terms)
            else:
                p_eval_lld = partial(eval_likelihood_multi_lin_mix_constr, 
                                     xdata=xdata, ydata=ydata, 
                                     n_model=n_model, 
                                     include_noise=False,
                                     **fit_kwargs)
                
        self.n_param = n_param  # number of parameters
        self.lld = p_eval_lld
        
        
    def run(self, p0, method='Nelder-Mead', verbose=True):
        """ Run fitting """
        from scipy.optimize import minimize
        
        # Check if # of parameters match
        n_param = self.n_param
        assert n_param == len(p0), print("# of p0 does no match!")
        
        self.p0 = p0
        self.method = method
        
        lld = self.lld
        
        # run negative likelihood minimization
        res = minimize(lld, p0, method=method)
        params_fit = res.x
        
        self.res = res
        self.params_fit = params_fit
        self.runned = True
        
        # evaluate population probabilities
        if not self.mixture_model:
            if self.include_noise:
                prob = lld(params_fit, give_prob=True)
                self.prob = prob
                self.labels[self.cond] = (prob>0.5).astype(int)
        else:
            probs = lld(params_fit, give_prob=True)
            self.probs = probs
            
            prob = np.sum(probs, axis=0)
            self.labels[self.cond] = np.argmax(probs / prob, axis=0)
        
        # print out
        if verbose:
            print(self.msg.format(*self.params_fit))
    
    @property
    def BIC(self):
        lld = - self.lld(self.params_fit)
        n_params = len(self.params_fit)
        n_samples = len(self.ydata)
        
        bic = stats.bayesian_info_criterion(lld, n_params, n_samples)
        return bic
    
    def plot(self, 
             xlabel='', ylabel='',
             xrange=None, yrange=None,
             shrink=1, plot_quantiles=0.000001,
             nbins=None, color='k',
             return_fig=False):
        
        """ Plot fitting result. """
        
        from numpy.polynomial import Polynomial
        from scipy.special import expit
        from scipy.interpolate import UnivariateSpline
        from .utils import (solve_equation_poly, calculate_roots_coefs,
                            calculate_distance_spline, calculate_arc_length_points,
                            project_distance_spline)
        
        x_, y_ = self.x, self.y
        cond, labels = self.cond, self.labels
        ndim = self.ndim
        
        if shrink < 1:
            n_sample = int(len(y_) * shrink)
            idx = np.random.choice(np.arange(len(y_)), n_sample, replace=True)
            y_ = y_[idx]
            if ndim==1:
                x_ = x_[idx]
            else:
                x_ = x_.T[idx].T
            cond, labels = cond[idx], labels[idx]
        
        n_model = self.n_model
        poly_deg = self.poly_deg
        params_fit = self.params_fit
        piecewise = self.piecewise
        mixture_model = self.mixture_model
        
        if xrange is None:
            if ndim == 1:
                xrange = np.nanquantile(x_, [plot_quantiles,1-plot_quantiles])
            else:
                xrange = np.nanquantile(x_[0], [plot_quantiles,1-plot_quantiles])
        if yrange is None:
            yrange = np.nanquantile(y_, [plot_quantiles,1-plot_quantiles])
        
        xx = np.linspace(xrange[0], xrange[1], 100)

        if n_model==1:
            # 1d polynomial
            if ndim == 1:
                coefs = params_fit[:poly_deg+1]
                poly = Polynomial(coefs[::-1])
                yfunc = lambda x: poly(x)
            else:
                coefs = np.delete(params_fit[:(poly_deg+1) * ndim + 1], poly_deg+1)
                coefs = np.array(coefs).reshape(ndim, poly_deg+1)
                polys = [Polynomial(coef_n[::-1]) for coef_n in coefs]
                
                yy = polys[0](xx) # target value
                
                # xx_nd is grid points of the projected curve on the y=0 plane
                xx_nd = np.zeros((ndim, len(xx)))
                xx_nd[0] = xx
                for k in range(ndim-1):
                    xx_n_range = [np.nanmin(x_[k+1]), np.nanmax(x_[k+1])]
                    initial_guess = np.nanmedian(x_[k+1])
                    
                    xx_nd[k+1] = solve_equation_poly(coefs[k+1][::-1], yy, initial_guess, range=xx_n_range)
                    
                # Perform spline interpolation
                spline = UnivariateSpline(xx_nd[0], xx_nd[1], k=3, s=0)  # Cubic spline
            
                # projected arc length along the curve
                xx_p = [calculate_arc_length_points(xx_nd[0][:k+1], xx_nd[1][:k+1], spline) for k in range(len(xx))]
                
                # projected data points onto the curve
                x_0 = np.array([calculate_roots_coefs(coefs[k]) for k in range(ndim)])
                x_p = project_distance_spline(x_[0], x_[1], spline, x_0, t_p0=[0])
                
                # regression function for projected points, dummy function on xx
                yfunc = lambda x: np.average([polys[k](xx_nd[k]) for k in range(ndim)], weights=self.weights_filter, axis=0)
                
                self.x_0 = x_0
                self.spline = spline
                self.x_p = x_p
                
                # Now set xx and x_ to the new projected points for plot
                xx, x_ = xx_p, x_p
                
            self.coefs = coefs
            
        elif piecewise:
            ind = 3 * ndim + 2
            A1, x0, y0, A2, eps = params_fit[0:5]
            if ndim >1:
                A1 = np.atleast_1d([A1] + params_fit[5:ind:3])
                x0 = np.atleast_1d([x0] + params_fit[6:ind:3])
                A2 = np.atleast_1d([A2] + params_fit[7:ind:3])
            
            B1 = y0 - A1 * x0 
            B2 = y0 - A2 * x0
            
            f1 = lambda x: A1 * x + B1
            f2 = lambda x: A2 * x + B2

            if ndim == 1:
                yfunc = lambda x: np.piecewise(x, [x<x0, x>=x0], [f1, f2])
            else:
                # new variables on the projected plane
                f1_proj = lambda x: np.sqrt(np.sum((x.T+B1/A1)**2, axis=1))
                f2_proj = lambda x: np.sqrt(np.sum((x.T+B2/A2)**2, axis=1))
                x_ = x_p = np.piecewise(x_, [y_<y0, y_>=y0], [f1_proj, f2_proj])
                self.x_p = x_p
                x0_p = np.sqrt(np.sum((x0+B1/A1)**2))
                yfunc = lambda x: np.piecewise(x, [x<x0_p, x>=x0_p], [f1, f2])
            
        else:
            As = params_fit[:3*n_model:3]
            Bs = params_fit[1:3*n_model:3]
            fracs = params_fit[2:3*n_model:3]
            yfunc = lambda x: np.sum(np.r_['0,2,1', [A * x * frac + B * frac for (A, B, frac) in zip(As, Bs, fracs)]], axis=0)
            
        self.yfunc = yfunc
        
        # smoothing with sigmoid function
        if self.sigmoid:
            a = self.sigmoid_a
            if n_model == 1:
                x_min = params_fit[poly_deg+2]
                y_min = poly(x_min-a)
            elif piecewise:
                x_min = params_fit[ind]
                y_min = A1 * (x_min-a) + B1
            else:
                x_min = params_fit[3*n_model+1]
                y_min = yfunc(xx)[np.argmin(abs(xx-(x_min-a)))]

            h = lambda x: expit((x-x_min)/a)
            yy = yfunc(xx) * h(xx) + y_min * (1-h(xx))
        else:
            yy = yfunc(xx)
            
        self.xx = xx
        self.yy = yy
                
        # remove 2D background
        if self.legendre_terms is not None:
            coefs_bkg = params_fit[-5:]  # last five parameters
            bkg = np.sum([A*P for (A, P) in zip(coefs_bkg, self.legendre_terms)], axis=0)
            y -= bkg

        # set canvas
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12, 5))

        # Hess plot
        xyrange = [[*xrange], [*yrange]]
        if nbins is None:
            nbins = int(100*self.rescale_factor)
        H,xbins,ybins,image = ax1.hist2d(x_[cond], y_[cond], bins=nbins, range=xyrange)
        levels = np.linspace(0, np.log10(0.8*H.max()), 7)[1:]
        ax1.contour(np.log10(H.T+1e-2), levels, extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()], linewidths=1, cmap='gray')

        # Draw points outside contours
        indices = np.column_stack((np.digitize(x_, xbins[1:-1]), np.digitize(y_, ybins[1:-1])))
        inside = np.array([np.log10(H[ind[0],ind[1]]+0.01) for ind in indices]) >= levels[0]
        ax1.scatter(x_[(~inside)&cond], y_[(~inside)&cond], s=3, alpha=0.2, color='w')

        # scatter plot
        if mixture_model:
            # plot labeled data
            for k in range(n_model):
                if fracs[k] > 1e-4:
                    ax2.scatter(x_[labels==k+1], y_[labels==k+1], s=0.1, alpha=0.02)
        else:
            ax2.scatter(x_[cond], y_[cond], s=1, alpha=0.05, color=color)
                    
        # plot data not in use
        ax2.scatter(x_[~cond], y_[~cond], s=0.1, alpha=0.02, color='gray')
        if self.include_noise:
            # plot outliers
            ax2.scatter(x_[labels==0], y_[labels==0], s=0.1, alpha=0.05, color='gray')
        
        xlim, ylim = xrange, yrange

        ax2.set_xlim(xlim[0],xlim[1])
        ax2.set_ylim(ylim[0],ylim[1])

        for ax in [ax1,ax2]:
            if mixture_model:
                for (A, B, frac) in zip(As, Bs, fracs):
                    if frac > 1e-4:
                        ax.plot(xx, A * xx + B, ls='--')
            ax.plot(xx, yy, color='lime')
            if self.xmin is not None:
                ax.axvline(self.xmin, color='gold', ls='--')
            if self.ymin is not None:
                ax.axhline(self.ymin, color='gold', ls='--')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        
        plt.tight_layout()
        
        if return_fig:
            return fig, (ax1,ax2)
        else:
            plt.show()
            return None
