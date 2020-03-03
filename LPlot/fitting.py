from pylab import plot,loglog,semilogx,semilogy, ginput,close
from numpy import isscalar, where
def close_to(d_xy,xy):
    num = min(abs(d_xy - xy)) 
    ind = where(num==abs(d_xy-xy))
    return ind


def set_fitrange(x, y, _plot = plot ):
    
    try:            
        x[isnan(x)] = 0.
    except:
        pass
    #fig = figure()
    _plot(x, y, 'o')

    fit_range_ok = True
    while fit_range_ok:
        print "Select the left and right border of the fitrange!"
        print "\n\t* Left mouse click in the graph for left an right border "
        print "\t* Right clicking cancels last input."
        a, b = ginput(2,timeout = -1) 
        right_border = a[0]
        left_border = b[0]
        fit_range_ok = False
    start = close_to(x, right_border)
    end = close_to(x, left_border)
    if start > end:
        _start = start
        start = end
        end = _start
    close()
    return start, end, right_border, left_border

class data_fitting():
    def __init__(self, xdata, ydata, p0, const, mimas, func, *args, **kwargs):
        self.xdata = xdata
        self.ydata = ydata
        self.p0 = p0
	self.const = const
	self.mima = mimas
        self.func = func
        self.args = args

        if "weights" in kwargs:
            self.weights0 = kwargs['weights']
            kwargs.pop('weights')
        else:
            self.weights0 = 1.

        if 'plot' in kwargs:
            self.plot = kwargs['plot']
            kwargs.pop('plot')
        else:
            self.plot = plot

        if 'fit_range' in kwargs:
            if isscalar(kwargs['fit_range']):
                start, end, right_border, left_border = set_fitrange(self.xdata, self.ydata,  _plot = self.plot)
                self.x = xdata[start[0]:end[0] + 1]
                self.y = ydata[start[0]:end[0] + 1]
                self.start = start[0]
                self.end = end[0]
            elif isinstance(kwargs['fit_range'], list):
                start, end = kwargs['fit_range']
                self.x = xdata[start:end + 1]
                self.y = ydata[start:end + 1]
                self.start = start
                self.end = end
            kwargs.pop('fit_range')
            try:
                self.weights = self.weights0[self.start:self.end + 1]
            except:
                self.weights = 1.
        else:
            self.x = xdata
            self.y = ydata
            self.weights = self.weights0
        
        self.kwargs = kwargs

    def leastsq(self, *args, **kwargs):
        from scipy.optimize import leastsq

        if 'plotting' in kwargs:
            plotting = True
            kwargs.pop('plotting')
        else:
            plotting = False

        f = lambda p, x, y, weights: (self.func(p, x,self.const,self.mima) - y) / weights
        self.p, self.cov = leastsq(f, self.p0, reargs = (self.x, self.y, self.weights), **kwargs)
        
        if plotting:
            self.plot(self.xdata, self.ydata, 'o', label = 'original data')
            self.plot(self.x, self.y, '+')
            self.plot(self.xdata, self.func(self.p, self.xdata,self.const,self.mima), label = 'fit with scipy.optimize.leastsq')
        


    def curve_fit(self, **kwargs):
        from scipy.optimize import curve_fit

        if 'plotting' in kwargs:
            plotting = True
            kwargs.pop('plotting')
        else:
            plotting = False

        def f(x, *p):
            return self.func(p[0], x, self.const, self.mima)

        self.p, self.cov = curve_fit(f, self.x, self.y, p0 = self.p0, sigma = self.weights, **kwargs)

        if plotting:
            self.plot(self.xdata, self.ydata, 'o', label = 'original data')
            self.plot(self.x, self.y, '+')
            self.plot(self.xdata, self.func(self.p, self.xdata,self.const,self.mima), label = 'fit with scipy.optimize.curve_fit')




    def odr(self, *args, **kwargs):
        from scipy import odr

        if 'plotting' in kwargs:
            plotting = True
            kwargs.pop('plotting')
        else:
            plotting = False


        f = lambda p, x: self.func(p, x,self.const,self.mima)
        self.mod = odr.Model(f)
        self.dat = odr.Data(self.x, self.y, we = 1. / self.weights)
        self.my_odr = odr.ODR(self.dat, self.mod, self.p0, *args, **kwargs)
        self.out = self.my_odr.run()
        self.p = self.out.beta
        self.cov = self.out.cov_beta
        #self.sd = out.sd_beta
        
        if plotting:
            self.plot(self.xdata, self.ydata, 'o', label = 'original data')
            self.plot(self.x, self.y, '+')
            self.plot(self.xdata, self.func(self.p, self.xdata,self.const,self.mima), label = 'fit with scipy.odr')

        

    def openopt(self, *args, **kwargs):
        from openopt import NLP
        
        if 'plotting' in kwargs:
            plotting = True
            kwargs.pop('plotting')
        else:
            plotting = False
        try:
            solver = args[0]
        except:
            solver = 'ralg'
        self.args += args[1:]
        self.kwargs.update(kwargs)

        
        f = lambda p: sum(((self.func(p, self.x, self.const,self.mima) - self.y) / self.weights)**2)
        self.prob = NLP(f, x0 = self.p0, *self.args, **self.kwargs)
        self.res = self.prob.solve(solver)
        self.p = self.res.xf

        if plotting:
            self.plot(self.xdata, self.ydata, 'o', label = 'original data')
            self.plot(self.x, self.y, '+')
            self.plot(self.xdata, self.func(self.p, self.xdata,self.const,self.mima), label = 'fit with openopt:\n solver: '+ solver)


