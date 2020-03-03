#!/usr/bin/python
# -*- coding: iso-8859-1 -*-
#  Januar 2009 (JL)

import numpy, os, string, time
from LP_object import m_Object

def _conv(obj, dtype=None):
    """ Convert an object to the preferred form for input to the odr routine.
    """
    if obj is None:
        return obj
    else:
        if dtype is None:
            obj = numpy.asarray(obj)
        else:
            obj = numpy.asarray(obj, dtype)
        if obj.shape == ():
            # Scalar.
            return obj.dtype.type(obj)
        else:
            return obj

def _asarray(obj):
    if numpy.array(obj).ndim == 0:
        return numpy.array([obj])
    else:
        return numpy.array(obj)

def _ndim(obj):
    """
    Get dimension of object
    """
    try:
        if isinstance(obj,dict):
            return len(obj)
        return numpy.array(obj).ndim
    except:
        return len(obj)

# Secure checking if a object is an instance of testobj
def isinstanceof(obj, testobj):
    try:
        if isinstance(obj, testobj):
            return True
    except:
        pass
    for i in (1,2):
        basesstr = []
        b = obj.__class__
        bases = list(b.__mro__)
        for b in bases:
            if str(b).count('<class') > 0:
                basesstr.append(str(b).lstrip("<class '").rstrip("'> "))
            elif str(b).count('<type') > 0:
                basesstr.append(str(b).lstrip("<type '").rstrip("'> "))
            for part in basesstr[-1].split('.'):
                if part not in basesstr:
                    basesstr.append(part)

        try:
            if str(testobj).lstrip("<class''> ").rstrip("'> ") in basesstr:
                return True
            if testobj in bases:
                return True
        except:
            pass
        obj, testobj = testobj, obj
    return False

def getBases(obj):
    basesstr = []
    b = obj.__class__
    while 1:
        basesstr.append(str(b).lstrip("<class''> ")[:-2])
        b = b.__base__
        if b is None:
            break
    return basesstr


def savitzky_golay(y, window_size, order, deriv=0):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techhniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = numpy.linspace(-4, 4, 500)
    y = numpy.exp( -t**2 ) + numpy.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, numpy.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    try:
        window_size = numpy.abs(numpy.int(window_size))
        order = numpy.abs(numpy.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = numpy.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = numpy.linalg.pinv(b).A[deriv]
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - numpy.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + numpy.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = numpy.concatenate((firstvals, y, lastvals))
    return numpy.convolve( m, y, mode='valid')


def smooth(data,window_len=10,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    """

    if data.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if data.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return data


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=numpy.r_[2*data[0]-data[window_len:1:-1],data,2*data[-1]-data[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='same')
    return y[window_len-1:-window_len+1]


def smooth_data2D(data,s1,s2):
        tmp = []
        for i in xrange(data.shape[0]):
                i1 = numpy.where(i-s1/2<0,0,i-s1/2)
                i2 = numpy.where(i+s1/2>data.shape[0]-1,data.shape[0]-1,i+s1/2)
                if i1 == i2: i2+=1
                if s2 < 2:
                    tmp.append(numpy.array([numpy.sum(numpy.nan_to_num(data[i1:i2,j]))/len(data[i1:i2,j]) for j in xrange(0,data.shape[1])]))
                else:
                    tmp.append(numpy.array([numpy.sum(numpy.nan_to_num(data[i1:i2,numpy.where(j-s2/2<0,0,j-s2/2):numpy.where(j+s1/2>data.shape[1]-1,data.shape[1]-1,j+s2/2)]))/len(data[i1:i2,numpy.where(j-s2/2<0,0,j-s2/2):numpy.where(j+s1/2>data.shape[1]-1,data.shape[1]-1,j+s2/2)]) for j in xrange(0,data.shape[1])]))
        return(numpy.array(tmp))

def diffusion(b,t,x):
    dd = []
    #loop over timeslots
    for i in xrange(0,b.shape[1]-1,1):
        a = 0
        a2 = 0
        c = 0
        a = numpy.mean(numpy.nan_to_num(x*b[:,i]))*1e-9
        for j in xrange(0,b.shape[0]-1,1):
            if not numpy.isnan(b[j,i]) and b[j,i] > 0:
                a2 += (a-(x[j]*b[j,i]*1e-9))**2
                c += 1

        d = a2/(c*t[i]*2)
        dd.append(d)
    return numpy.array(dd)

def meanValue(b,x):
    dd = []
    #loop over timeslots
    for i in xrange(0,b.shape[1],1):
        c = numpy.size(numpy.where(numpy.nan_to_num(b[:,i])>0))
        a = numpy.sum(numpy.nan_to_num(x*b[:,i]))
        a /= (1.*b.shape[1])
        dd.append(a)
    return numpy.array(dd)

def interpol(x, y, start=None, end=None, step=None, num = None, kind='lin'):
    from scipy import interpolate, exp, log10, linspace, logspace
    from pylab import stineman_interp
    from numpy import where
    # check if it is a linear vector or a log one
    if (x[1]-x[0])*(1-0.01) <= x[2]-x[1] <=(x[1]-x[0])*(1+0.01):
        scale = 'lin'
    else:
        scale = 'log'
    # set starting point of interpolation
    # if it is not given
    if start is None:
        start = x[0]
    # set ending point of interpolation
    # if it is not given
    if end is None:
        end = x[-1]

    # get stepsize / number of values of interpolation
    # if the x vector should be logarithmic
    if kind == 'log':
        if start <= 0.0: start = x[where(x>0.0)[0][0]]
        if num is None:
            if scale == 'log':
                if step is None:
                    step = log10(x[2]) - log10(x[1])
                num = 1 + (log10(end) - log10(start)) / step
            else:
                if step is None:
                    step = x[2] - x[1]
                num = 1 + (end - start) / step
        x1 = 10**( linspace( log10(start), log10(end), num ) )
    # if the x vector should be linear
    else: # linear
        if num is None:
            if scale == 'log':
                if start <= 0.0: start = x[where(x>0.0)[0][0]]
                if step is None:
                    step = log10(x[2]) - log10(x[1])
                num = 1 + (log10(end) - log10(start)) / step
            else:
                if step is None:
                    step = x[2] - x[1]
                num = 1 + (end - start) / step
        x1 =linspace( start, end, num )

    # build a spline out of the x, y bvalues
    #~ spline_coeffs=interpolate.splrep(x,y,s=0)
    #~ # interpolate for new x vector
    #~ y1=interpolate.splev(x1,spline_coeffs,der=0)
    y2 = stineman_interp(x1, x, y)


    #~ y2 = interpolate.splev(x1, interpolate.splrep(x, y, k=3))
    #~ y2 = interpolate.interp1d(x,y)(x1)

    return x1, y2#~, y2



def derive((x), (y), n=3):
    """Berechnet die Ableitung von y(x), indem ein quadratischer Fit abgeleitet wird.

    rein: 1d-array x , unabh�ngige Variable nach der gefittet wird
          1d-array y, abhangige Variable die angefittet wird
          integer n, Anzahl der einzubeziehenden Punkte rechts und links (Standard: 3)
    raus: 1d-array dy, Ableitung der Funktion y(x)"""

    from numpy import shape, zeros, polyfit, nan, array, transpose, ndim

    if shape(x)[0] != shape(y)[0]:
        if ndim(y) == 2:
            if shape(x)[0] != transpose(y).shape[0]:
                print "Fehler in sammlung.derive: len(x) != len(y)"
                return zeros(shape(y), float) * nan
            else:
                y = transpose(y)
        else:
            print "Fehler in sammlung.derive: len(x) != len(y)"
            return zeros(shape(y), float) * nan

    dy = zeros(shape(y), float) * nan
    for i in xrange(array(x).shape[0]):

        # Grenzen des Fitbereichs festlegen (j Unter-, k Obergrenze):
        if (i>=n) & (i<array(x).shape[0]-n): # wenn i n Punkte vom Rand von x entfernt
            j = i-n/2
#            k = i+n+1
            k = i+n/2+1
        elif i<n:                 # wenn i in den n Punkten am unteren Ende von x
            j = 0
#            k = 2*n+1
            k = n+1
        elif i>=array(x).shape[0]-n:         # wenn i in den n Punkten am oberen Ende von x
#            j = len(x)-(2*n+1)
            j = array(x).shape[0]-n
            k = array(x).shape[0]

        if ndim(y) == 2:
            for l in xrange(y.shape[1]):
                coeffs = polyfit(x[j:k], y[j:k,l], 2)     # Koeffizienten des Polynomfits
                dy[i,l] = coeffs[0]*2*x[i] + coeffs[1]    # Ableitung des Fits
        else:
            coeffs = polyfit(x[j:k], y[j:k], 2)     # Koeffizienten des Polynomfits
            dy[i] = coeffs[0]*2*x[i] + coeffs[1]    # Ableitung des Fits

    return dy

def derive_spline(x, y, n=3, der=1, s=0, k=2):
    """Berechnet die Ableitung von y(x), indem ein quadratischer Fit abgeleitet wird.

    rein: 1d-array x , unabh�ngige Variable nach der gefittet wird
          1d-array y, abhangige Variable die angefittet wird
          integer n, Anzahl der einzubeziehenden Punkte rechts und links (Standard: 3)
    raus: 1d-array dy, Ableitung der Funktion y(x)"""

    from numpy import shape, zeros, nan, array, transpose, ndim
    from scipy.interpolate import splrep, splev

    if shape(x)[0] != shape(y)[0]:
        if ndim(y) == 2:
            if shape(x)[0] != transpose(y).shape[0]:
                print "Fehler in sammlung.derive: len(x) != len(y)"
                return zeros(shape(y), float) * nan
            else:
                y = transpose(y)
        else:
            print "Fehler in sammlung.derive: len(x) != len(y)"
            return zeros(shape(y), float) * nan

    dy = zeros(shape(y), float) * nan
    for i in xrange(array(y).shape[0]):

        # Grenzen des Fitbereichs festlegen (j Unter-, k Obergrenze):
        if (i>=n) & (i<array(x).shape[0]-n): # wenn i n Punkte vom Rand von x entfernt
            j = i-n/2
            m = i+n/2+1
        elif i<n:                 # wenn i in den n Punkten am unteren Ende von x
            j = 0
            m = n+1
        elif i>=array(x).shape[0]-n:         # wenn i in den n Punkten am oberen Ende von x
#            j = len(x)-(2*n+1)
            j = array(x).shape[0]-n
            m = array(x).shape[0]

        if ndim(y) == 2:
            for l in xrange(y.shape[1]):
                tck = splrep(x[j:m], y[j:m,l], k=k, s=s)
                dy[i,l] = splev(x[i], tck, der=der)    # 1. Ableitung des Fits
        else:
            tck = splrep(x[j:m], y[j:m], k=k, s=s)
            dy[i] = splev(x[i], tck, der=der)    # 1. Ableitung des Fits

    return dy
# Ende derive


# -------------------------------------------------------------------------------


def derive_lin(x, y):
    """Berechnet die Ableitung von y(x), indem eine Ausgleichsgerade abgeleitet wird.

    rein: 1d-array x , unabhaengige Variable nach der gefittet wird
          1d-array y, abhangige Variable die angefittet wird
    raus: 1d-array dy, Ableitung der Funktion y(x)"""

    from scipy import diff
    from numpy import array, append, ndim
    x=_asarray(x)
    y=_asarray(y)
    if ndim(y) == 2:
        diffy=array([append(1.*diff(y)[i],diff(y)[i,-1]) for i in xrange(y.shape[0])])
    else:
        diffy=append(1.*diff(y),diff(y)[-1])
    diffx=append(1.*diff(x),diff(x)[-1])

    dy = diffy/diffx
    #~ dy = numpy.append(dy,dy[-1])

    return dy


def reversed(x):
    if hasattr(x, 'keys'):
        raise ValueError("mappings do not support reverse iteration")
    i = len(x)
    while i > 0:
        i -= 1
        yield x[i]

def bigger(a,b):
    return a>b

def smaller(a,b):
    return a<b


def sec_ode(mu = 1e-10, A = 1e4, d = 1e-7, b = 0.1, eps = 3, n =1e23 ):
    import phc
    from numpy import linspace
    tau = eps*phc.eps0/(phc.q*n*mu*b)
    c1 = mu*A/d
    c2 = 1/(d*b*tau)
    t = linspace(0,1e-4,500)
    print tau, c1, c2
    def f(l,t):
        dl1dt = l[1]
        dl2dt = -c2*l[1]*l[0]/(1+t/tau) + c1
        return (dl1dt, dl2dt)
    from scipy.integrate import odeint
    l = odeint(f,[0,0], t)
    dldt = l[:,1]
    l = l[:,0]
    j = phc.eps0*eps/d*A + phc.q*n*(1-l/d)/(1+t/tau)*dldt
    return l,dldt,j, t


def test_mu(A = 5e4, n=5e22, d = 1e-7, eps = 3.3, Vmax = 20000000):
    import CELIV_evaluation
    import LP_data;       reload(LP_data)
    from LP_data import m_dataObject, m_dataHandler
    from numpy import array, linspace, logspace, sqrt,log10
    from phc import eps0
    mu_start = -16; mu_end = -5
    b = CELIV_evaluation.CELIV_evaluation()
    dH=m_dataHandler()
    real_mu = []; calc_mu = []; tmaxs = []; fOs = []; errs = []
    for mu in logspace(mu_start, mu_end, 1 + (abs(mu_end-mu_start) * 3)):
        mu_approx=(sqrt(2/mu/A)*d)
        t = logspace(log10(mu_approx)-1, log10(mu_approx)+2,50000)
        fO = m_dataObject()
        b.eq1.set_values(A = A, n = n, d = d, eps = eps, Vmax = Vmax, mu = mu, t = t)
        fO.setData('j', b.eq1.offset() + b.eq1.fj['exact1']())
        fO.setData('l', b.eq1.fl['exact1']())
        fO.setData('t', t)
        fO.setHeader('A', A)
        fO.setHeader('T', 300)
        fO.setHeader('eps', eps)
        fO.setHeader('Vmax', Vmax)
        fO.setHeader('F', 0)
        fO.setHeader('d', d)
        fO.setHeader('n', n)
        fO.setHeader('j0', eps0*eps*A/d)
        b.calc_tmax( fO )
        b.calc_dj( fO ,fO.getHeader('j0'))
        fO.setHeader('mu/0',mu)
        b.calc_mobility( fO )
        real_mu.append( mu )
        calc_mu.append( [fO.getHeader('mu/simple'), fO.getHeader('mu/juska'), fO.getHeader('mu/juskacor18'),
                         fO.getHeader('mu/deibel'), fO.getHeader('mu/bange'),
                         fO.getHeader('mu/bange_cor'), fO.getHeader('mu/lorrmann1'), fO.getHeader('mu/lorrmann2')])
        fO.setHeader('mu/simple/error',abs(1-fO.getHeader('mu/simple')/mu))
        fO.setHeader('mu/juska/error',abs(1-fO.getHeader('mu/juska')/mu))
        fO.setHeader('mu/juskacor18/error',abs(1-fO.getHeader('mu/juskacor18')/mu))
        fO.setHeader('mu/deibel/error',abs(1-fO.getHeader('mu/deibel')/mu))
        fO.setHeader('mu/bange/error',abs(1-fO.getHeader('mu/bange')/mu))
        fO.setHeader('mu/bange_cor/error',abs(1-fO.getHeader('mu/bange_cor')/mu))
        fO.setHeader('mu/lorrmann1/error',abs(1-fO.getHeader('mu/lorrmann1')/mu))
        fO.setHeader('mu/lorrmann2/error',abs(1-fO.getHeader('mu/lorrmann2')/mu))
        #~ fO.setHeader('mu/new11/error',abs(1-fO.getHeader('mu/new11')/mu))
        #~ fO.setHeader('mu/new21/error',abs(1-fO.getHeader('mu/new21')/mu))
        #~ fO.setHeader('mu/new12/error',abs(1-fO.getHeader('mu/new12')/mu))
        #~ fO.setHeader('mu/new22/error',abs(1-fO.getHeader('mu/new22')/mu))
        dH.add(fO)
        #~ errs.append([abs(1-array(calc_mu)[-1,0]/real_mu[-1]),(abs(1-array(calc_mu)[-1,1]/real_mu[-1])),
        #~ (abs(1-array(calc_mu)[-1,2]/real_mu[-1])),(abs(1-array(calc_mu)[-1,3]/real_mu[-1])),
        #~ (abs(1-array(calc_mu)[-1,4]/real_mu[-1])),(abs(1-array(calc_mu)[-1,5]/real_mu[-1])),
        #~ (abs(1-array(calc_mu)[-1,6]/real_mu[-1])),(abs(1-array(calc_mu)[-1,7]/real_mu[-1]))])
        errs.append([fO.getHeader('mu/simple/error'),fO.getHeader('mu/juska/error'),
        fO.getHeader('mu/juskacor18/error'),fO.getHeader('mu/deibel/error'),
        fO.getHeader('mu/bange/error'),fO.getHeader('mu/bange_cor/error'),
        fO.getHeader('mu/lorrmann1/error'),fO.getHeader('mu/lorrmann2/error')])
    return array(calc_mu), array(real_mu), array(errs),dH

def test_mu1(mu = 1e-8, n=5e22, d = 1e-7, eps = 3.3, Vmax = 20000000):
    import CELIV_evaluation
    import LP_data;       reload(LP_data)
    from LP_data import m_dataObject, m_dataHandler
    from numpy import array, linspace, logspace, sqrt,log10
    from phc import eps0
    A_start = 3; A_end = 7
    b = CELIV_evaluation.CELIV_evaluation()
    dH=m_dataHandler()
    real_mu = [];real1_mu = []; calc_mu = []; tmaxs = []; fOs = []; errs = []
    for A in logspace(A_start, A_end,20):
        mu_approx=(sqrt(2/mu/A)*d)
        t = logspace(log10(mu_approx)-1, log10(mu_approx)+2,50000)
        fO = m_dataObject()
        b.eq1.set_values(A = A, n = n, d = d, eps = eps, Vmax = Vmax, mu = mu, t = t)
        fO.setData('j', b.eq1.offset() + b.eq1.fj['exact1']())
        fO.setData('l', b.eq1.fl['exact1']())
        fO.setData('t', t)
        fO.setHeader('A', A)
        fO.setHeader('A/scaled', b.eq1.s_A*A)
        fO.setHeader('T', 300)
        fO.setHeader('eps', eps)
        fO.setHeader('Vmax', Vmax)
        fO.setHeader('F', 0)
        fO.setHeader('d', d)
        fO.setHeader('n', n)
        fO.setHeader('j0', eps0*eps*A/d)
        b.calc_tmax( fO )
        b.calc_dj( fO ,fO.getHeader('j0'))
        fO.setHeader('mu/0',mu)
        b.calc_mobility( fO )
        real_mu.append( A )
        real1_mu.append( b.eq1.s_A*A )
        calc_mu.append( [fO.getHeader('mu/simple'), fO.getHeader('mu/juska'), fO.getHeader('mu/juskacor18'),
                         fO.getHeader('mu/deibel'), fO.getHeader('mu/bange'),
                         fO.getHeader('mu/bange_cor'), fO.getHeader('mu/lorrmann1'), fO.getHeader('mu/lorrmann2')])
        fO.setHeader('mu/simple/error',abs(1-fO.getHeader('mu/simple')/mu))
        fO.setHeader('mu/juska/error',abs(1-fO.getHeader('mu/juska')/mu))
        fO.setHeader('mu/juskacor18/error',abs(1-fO.getHeader('mu/juskacor18')/mu))
        fO.setHeader('mu/deibel/error',abs(1-fO.getHeader('mu/deibel')/mu))
        fO.setHeader('mu/bange/error',abs(1-fO.getHeader('mu/bange')/mu))
        fO.setHeader('mu/bange_cor/error',abs(1-fO.getHeader('mu/bange_cor')/mu))
        fO.setHeader('mu/lorrmann1/error',abs(1-fO.getHeader('mu/lorrmann1')/mu))
        fO.setHeader('mu/lorrmann2/error',abs(1-fO.getHeader('mu/lorrmann2')/mu))
        #~ fO.setHeader('mu/new11/error',abs(1-fO.getHeader('mu/new11')/mu))
        #~ fO.setHeader('mu/new21/error',abs(1-fO.getHeader('mu/new21')/mu))
        #~ fO.setHeader('mu/new12/error',abs(1-fO.getHeader('mu/new12')/mu))
        #~ fO.setHeader('mu/new22/error',abs(1-fO.getHeader('mu/new22')/mu))
        dH.add(fO)
        errs.append([fO.getHeader('mu/simple/error'),fO.getHeader('mu/juska/error'),
        fO.getHeader('mu/juskacor18/error'),fO.getHeader('mu/deibel/error'),
        fO.getHeader('mu/bange/error'),fO.getHeader('mu/bange_cor/error'),
        fO.getHeader('mu/lorrmann1/error'),fO.getHeader('mu/lorrmann2/error')])
    return array(calc_mu), array(real_mu), array(real1_mu), array(errs),dH

def strSort_cmp(x,y):
   return natcmp(x,y)

def try_int(s):
    "Convert to integer if possible."
    try: return int(s)
    except: return s

def natsort_key(s):
    "Used internally to get a tuple by which s is sorted."
    import re
    return map(try_int, re.findall(r'(\d+|\D+)', s))

def natcmp(a, b):
    "Natural string comparison, case sensitive."
    return cmp(natsort_key(a), natsort_key(b))

def natcasecmp(a, b):
    "Natural string comparison, ignores case."
    return natcmp(a.lower(), b.lower())



class m_Progress:
    def __init__(self, finalcount, header=None, progresschar=None):
        import sys
        self.finalcount=finalcount
        self.blockcount=0
        self.count = 0
        self.last_update = time.time()
        self.start = self.last_update
        self.elapsed = {}
        self.total = {}
        self.remaining = {}
        self.rate = 0
        if header is None:
            self.header='% Progress'
        else:
            self.header=header
        self.topline = ('\n|' + '-'*(23 - len(self.header)/2) + ' ' + self.header +
                        ' ' +'-'*(22 - len(self.header)/2) + '|\n')
        while 1:
            if len(self.topline) < 52:
                self.topline=self.topline[:-2]+'-|\n'
            else:
                break
        #
        # See if caller passed me a character to use on the
        # progress bar (like "*").  If not use the block
        # character that makes it look like a real progress
        # bar.
        #
        if not progresschar: self.block='x'
        else:                self.block=progresschar
        self.front='>>'
        #
        # Get pointer to sys.stdout so I can use the write/flush
        # methods to display the progress bar.
        #
        self.f=sys.stdout
        #
        # If the final count is zero, don't start the progress gauge
        #
        if not self.finalcount : return
        #~ self.f.write('\n------------------ % Progress -------------------|\n')
        self.f.write(self.topline)
        self.f.write('   10   20   30   40   50   60   70   80   90   100\n')
        self.f.write('|------------------------------------------------|\n')
        return

    def progress(self, add=1,name=None):
        #
        # Make sure I don't try to go off the end (e.g. >100%)
        #
        self.count=min(self.count+add, self.finalcount)
        #
        # If finalcount is zero, I'm done
        #
        if self.finalcount:
            percentcomplete=int(round(100*self.count/self.finalcount))
            if percentcomplete < 1: percentcomplete=1
        else:
            percentcomplete=100

        blockcount=max(3,int(percentcomplete/2))

        now = time.time()
        # Caclulate times
        self.rate = (now - float(self.last_update))
        self.elapsed['v'] = now - float(self.start)
        self.total['v'] = (self.elapsed['v']*(1.*self.finalcount)/self.count)
        self.remaining['v'] = self.total['v'] - self.elapsed['v']
        self.last_update = now

        for part in (self.elapsed, self.total, self.remaining):
          part['s'] = '%s'%('%.1fs'%((part['v'] % 60)))
          part['m'] = '%s'%('%dm:'%(int(part['v'] % 3600)/60) if int(part['v'] % 3600)/60>0 else '')
          part['h'] = '%s'%('%dh:'%(int(part['v'] / 3600)) if int(part['v'] / 3600)>0 else '')


        if percentcomplete < 100:
            #~ if blockcount > self.blockcount:
                self.f.write('[')
                for i in range(blockcount-3):
                    self.f.write(self.block)

                self.f.write(self.front)

                for i in range(52-blockcount-3):
                      self.f.write(' ')

                if name is not None:
                    self.f.write(']  %s e:[%s%s%s] r:[%s%s%s]                  '\
                          %(name,self.elapsed['h'], self.elapsed['m'], self.elapsed['s'],
                            self.remaining['h'], self.remaining['m'], self.remaining['s']))
                else:
                  self.f.write(']  e:[%s%s%s] r:[%s%s%s]                   '\
                          %(self.elapsed['h'], self.elapsed['m'], self.elapsed['s'],
                            self.remaining['h'], self.remaining['m'], self.remaining['s']))
                self.f.write('\r\r')
                self.f.flush()

        else:
            if blockcount > self.blockcount:
                fstr=' Total time: [%s%s%s] ' %(self.total['h'],
                                              self.total['m'], self.total['s'])
                self.f.write(fstr+'\r\r')
                self.f.write('[')
                if len(fstr)%2 ==0:
                  for i in range(blockcount/2-len(fstr)/2-1):
                      self.f.write(self.block)
                else:
                  for i in range(blockcount/2-len(fstr)/2-2):
                    self.f.write(self.block)

                self.f.write(fstr)
                self.f.flush()

                for i in range(blockcount/2-len(fstr)/2-1):
                    self.f.write(self.block)

                self.f.write("]")
                for i in range(55):
                    self.f.write(' ')

                #self.f.write('\n')

                self.f.flush()

        self.blockcount=blockcount


def getObjRange(obj, start=None, stop=-1, step=1):
    from numpy import iterable
    #
    # check if obj is iterable
    #
    def _getPos(n):
        if n < 0:
            return max(len(obj)+n+1, 0)
        else:
            return min(n,len(obj))

    if not iterable(obj):
        print('ERROR: Object "%s" is not iterable' %type(obj))
        return 1

    if start is not None:
        start=_getPos(start)
        stop=_getPos(stop)
        return [obj[i] for i in xrange(start, stop, step)]
    else:
        return obj

def getIndex(array, item):
    if len(array.shape) == 1:
        try:
            return [array.tolist().index(item)]
        except:
            pass
    else:
        for i, subarray in enumerate(array):
            try:
                return [i] + index(subarray, item)
            except:
                pass

def isNumeric(test_str):
    import re
    test_str=str(test_str).lower().replace(' ','')
    return (re.search('[a-zA-Z]',test_str.replace('e','')) is None
            and re.search(':',test_str.replace('e','')) is None
            and re.search('[0-9]',test_str.replace('e','')) is not None)

def isFloat(test_str):
    import re
    test_str=str(test_str).lower().replace(' ','')
    return ((re.search('[0-9][e].[0-9]',test_str) is not None
           and isNumeric(test_str))
           or (re.search('[0-9].[0-9]+',test_str) is not None
           and isNumeric(test_str)))

def isInt(test_str):
    import re
    test_str=str(test_str).replace(' ','')
    return (re.search('[a-zA-Z]',test_str) is None
            and re.search('[\.]',test_str) is None
            and isNumeric(test_str))

def isAscii(test_str):
    import re
    test_str=str(test_str).replace(' ','')
    return not isNumeric(test_str)

def makeTexPower(conv_str):
    import re
    try:
        conv_str=re.sub('\w+/\w+','$\\\\frac{\mathrm{'+re.search('\w+/',
                 conv_str).group()[:-1]+'}}{\mathrm{'+re.search('/\w+',
                 conv_str).group()[1:]+'}}$',conv_str)
    except: pass
    try:
        conv_str=re.sub('\w+\\\\\w+','$\\\\frac{\mathrm{'+re.search('\w+\\\\',
                 conv_str).group()[:-1]+'}}{\mathrm{'+re.search('\\\\\w+',
                 conv_str).group()[1:]+'}}$',conv_str)
    except: pass
    try:
        conv_str=re.sub('e[+-][0-9]+','$\cdot$ 10$^{\mathrm{'+re.search('e[+-][0-9]+',
                 conv_str).group()[1:]+'}}$',conv_str)
    except: pass
    return conv_str

def filterStrings(test_strs,filter):
    import re
    return [test_str for test_str in test_strs
            if re.search(filter,test_str) is not None]

def atof(number_str):
    return (float(str(number_str).lower().replace(',','.')) if isNumeric(str(number_str))
                                               else number_str)

def atoi(number_str):
    return (int(str(number_str).lower().replace(',','.')) if isInt(str(number_str))
                                               else atof(number_str))

def transpose(lists):
    if not lists: return []
    return map(lambda *row: list(row), *lists)



def calc_coeff(num_points, pol_degree, diff_order=0):

    """ calculates filter coefficients for symmetric savitzky-golay filter.
        see: http://www.nrbook.com/a/bookcpdf/c14-8.pdf

        num_points   means that 2*num_points+1 values contribute to the
                     smoother.

        pol_degree   is degree of fitting polynomial

        diff_order   is degree of implicit differentiation.
                     0 means that filter results in smoothing of function
                     1 means that filter results in smoothing the first
                                                 derivative of function.
                     and so on ...

    """

    # setup interpolation matrix
    # ... you might use other interpolation points
    # and maybe other functions than monomials ....

    x = numpy.arange(-num_points, num_points+1, dtype=int)
    monom = lambda x, deg : pow(x, deg)

    A = numpy.zeros((2*num_points+1, pol_degree+1), float)
    for i in range(2*num_points+1):
        for j in range(pol_degree+1):
            A[i,j] = monom(x[i], j)

    # calculate diff_order-th row of inv(A^T A)
    ATA = numpy.dot(A.transpose(), A)
    rhs = numpy.zeros((pol_degree+1,), float)
    rhs[diff_order] = (-1)**diff_order
    wvec = numpy.linalg.solve(ATA, rhs)

    # calculate filter-coefficients
    coeff = numpy.dot(A, wvec)

    return coeff

def smooth_sg(signal, wl=20, ord=2, diff=0):

    """ applies coefficients calculated by calc_coeff()
        to signal """


    coeff = calc_coeff(wl, ord, diff)
    N = numpy.size(coeff-1)/2
    res = numpy.convolve(signal, coeff)
    return res[N:-N]

def deconvolve1(num, den, n=None):
    """Deconvolves divisor out of signal, division of polynomials for n terms

    calculates den^{-1} * num

    Parameters
    ----------
    num : array_like
        signal or lag polynomial
    denom : array_like
        coefficients of lag polynomial (linear filter)
    n : None or int
        number of terms of quotient

    Returns
    -------
    quot : array
        quotient or filtered series
    rem : array
        remainder

    Notes
    -----
    If num is a time series, then this applies the linear filter den^{-1}.
    If both num and den are both lagpolynomials, then this calculates the
    quotient polynomial for n terms and also returns the remainder.

    This is copied from scipy.signal.signaltools and added n as optional
    parameter.

    """
    import numpy as np
    from scipy import signal
    num = np.atleast_1d(num)
    den = np.atleast_1d(den)
    N = len(num)
    D = len(den)
    if D > N and n is None:
        quot = [];
        rem = num;
    else:
        if n is None:
            n = N-D+1
        input = np.zeros(n, float)
        input[0] = 1
        quot = signal.lfilter(num, den, input)
        num_approx = signal.convolve(den, quot, mode='full')
        if len(num) < len(num_approx):  # 1d only ?
            num = np.concatenate((num, np.zeros(len(num_approx)-len(num))))
        rem = num - num_approx
    return quot, rem


def Convolve(image1, image2, pad=True, sign=1):
    """ Not so simple convolution """
    from numpy import array, fft, log
    #Just for comfort:
    FFt = fft.fft
    iFFt = fft.ifft

    #The size of the images:
    r1 = array(image1).shape[0]
    r2 = array(image2).shape[0]

    #if the Numerical Recipies says so:
    print r1,r2
    r = 2*max(r1,r2)

    #For nice FFT, we need the power of 2:
    if pad:
        pr2 = int(log(r)/log(2.0) + 1.0 )
        rOrig = r/2
        r = 2**pr2
    #end of if pad

    #numpy fft has the padding built in, which can save us some steps
    #here. The thing is the s(hape) parameter:
    if sign == 1:
        fftimage = FFt(image1, n=r)*FFt(image2,n=r)
    else:
        fftimage = FFt(image1, n=r)/FFt(image2,n=r)

    if pad:
        return (iFFt(fftimage)[:rOrig]).real
    else:
        return (iFFt(fftimage)).real

