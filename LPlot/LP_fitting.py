#!/usr/bin/python
# -*- coding: iso-8859-1 -*-
#  Januar 2009 (JL)

import LP_object;       reload(LP_object)
from pylab import *
from scipy import odr
from scipy.interpolate import splrep, splev
from numpy import array, asarray, exp, sum, sqrt, ones, mean, nan, inf
from LP_func import getObjRange, interpol, isAscii
from LP_pylabhelpers import ginput, selector
from itertools import izip
import sys
import copy
import rainbow


try:
    import sherpa.astro.ui as sherpa; reload(sherpa)
    SHERPA = True
except:
    SHERPA = False

def _conv(obj, dtype=None):
    """ Convert an object to the preferred form for input to the odr routine.
    """

    if obj is None:
        return obj
    else:
        if dtype is None:
            obj = asarray(obj)
        else:
            obj = asarray(obj, dtype)
        if obj.shape == ():
            # Scalar.
            return obj.dtype.type(obj)
        else:
            return obj


class m_Model(LP_object.m_Object):

    # SubClass for parameters
    class parameter(LP_object.m_Object):
        def __init__(self, param, name, derived_class=LP_object.m_Object):
            derived_class.__init__(self)

            self.name=name
            self.value = 0.0
            self.min = -1e120
            self.max = 1e120
            self.unit = ''
            self.guessMin = False
            self.guessMax = False
            self.froozen = False

            varList = ['value','froozen','min','max','unit','guessMin','guessMax']
            if array(param).ndim == 0:
                self.value = param
            else:
                for p,v in izip(param,varList):
                    self[v] = p

            #
            # Now we have to check if self.min is bigger then self.max
            #
            self.checkMinMax()

        def __call__(self):
            return self.__dict__


        def checkMinMax(self,fac=1e5):
            """
            If I don't have boarders (min, max) for the fitting parameters
            I try to guess them. This can be inhibited by setting
            parameter "guessMin"/"guessMax" to False (which it is by default).

            """
            # True means the minimum/maximum boarder should be estimated
            if self.guessMax == True:
                self.max = float(self.value)*fac
            if self.guessMin == True:
                self.min = float(self.value)/fac
            if self.froozen:
                self.min=self.value
                self.max=self.value
            else:
                maxValue = max(self.min, self.max)
                self.min = min(self.min, self.max)
                self.max = maxValue


        def get(self, name):
            try:
                return self.__getattribute__(str(name))
            except:
                raise AttributeError, str(name)

        def set(self, name, value):
            try:
                self.__setattr__(str(name),value)
            except:
                raise AttributeError, str(name)


    def __init__(self, fcn, x0, const = [], name = '',derived_class=LP_object.m_Object):
        derived_class.__init__(self)
        self.fcn = fcn
        self.x0 = []
        for p in x0:
            if isinstance(p,dict):
                self.x0.append(m_Model.parameter( p.values()[0], p.keys()[0]) )
            elif array(p).ndim == 0:
                self.x0.append(m_Model.parameter( p, 'p%d'%(len(self.x0)+1)) )
            else:
                if isAscii(p[0]):
                    self.x0.append(m_Model.parameter( p[1:], p[0] ) )
                else:
                    self.x0.append(m_Model.parameter( p, 'p%d'%(len(self.x0)+1)) )

        self.const = const
        self.name = name

        self.params = [x()['value'] for x in self.x0]
        self.froozen = [x()['froozen'] for x in self.x0]
        self.mins = [x()['min'] for x in self.x0]
        self.maxs = [x()['max'] for x in self.x0]
        self.names = [x()['name'] for x in self.x0]
        self.units = [x()['unit'] for x in self.x0]

    def __call__(self):
        self.__update()
        ret = 'Model: %s\n' %(self.name if self.name <> '' else 'User Model')
        for p in self.x0:
            ret += ' |-Parameter: %s\n' %(p.name)
            ret += '   |---Value: %.3f\n' %(p.value)
            ret += '     |---Min: %.3f\n' %(p.min)
            ret += '     |---Max: %.3f\n' %(p.max)
            ret += '   |---Unit: [%s]\n' %(p.unit)
            ret += '   |---Froozen: %s\n\n' %(p.froozen)

        print ret

    def plot(self, x, kind='linear',appTo=None):
        if appTo is None:
            _f=figure()
            _ax=_f.add_subplot(111)
        else:
            _ax=appTo

        if kind is None or kind == 'linear':
            _ax.plot(x, self.fcn(self.params, x, self.const),'--',linewidth=1.5)

        elif  kind == 'loglog':
            _ax.loglog(x, self.fcn(self.params, x, self.const),'--',linewidth=1.5)

        elif kind == 'semilogy':
            _ax.semilogy(x, self.fcn(self.params, x, self.const),'--',linewidth=1.5)

        elif kind == 'semilogx':
            _ax.semilogx(x, self.fcn(self.params, x, self.const),'--',linewidth=1.5)

        show()
        draw()
        draw()
        show()

    def set_params(self):
        self.__update()

    def reset_params(self):
        self.__reset()

    def get_paramCount(self):
        return _conv(self.x0).shape[0]

    def __update(self):
        for p,x0 in izip(array(self.params),self.x0):
            if abs(p) == nan or abs(p) == inf:
                p = x0['value']
            if p > x0.max:
                p = x0['max']
            elif p < x0.min:
                p = x0['min']
            x0.set('value',p)

    def __reset(self):
        for p,x0 in izip(array(self.params),self.x0):
            p = x0['value']

    def __str__(self):
        ret = '\nPARAMS:\n-------\n'
        for k,v,f in izip(self.names,self.params, self.froozen):
            if not f:
                ret += ' %s:\t%s\n' %(k, v)
        return ret



class m_Data(odr.Data):
    def __init__(self, x, y, we = None, fix=None, meta={}, kind=None,
                 derived_class=odr.Data):

        derived_class.__init__(self, _conv(x), _conv(y), _conv(we), None, fix, meta)
        self.x_org = _conv(x)
        self.y_org = _conv(y)
        self.we_org = _conv(we)
        self.kind=kind

class m_Result(LP_object.m_Dict):
    def __init__(self,  derived_class=LP_object.m_Dict):
        derived_class.__init__(self)
        self.pprint = lambda : None


    def export(self):
        header=LP_object.m_Dict()
        data=LP_object.m_Dict()
        for key,val in self.getItems():

            if LP_object._ndim(val) == 0:
                header.set(key, val)
            else:
                if LP_object.isinstanceof('dict', val):
                    header.set(key, val)
                elif LP_object.isinstanceof('numpy.ndarray', val):
                    data.set(key,val)
        try:
            del header['format']
            del header['pprint']
            del data['modelvals']
        except:
            pass
        return header, data

#
# TODO: Result Klasse fertigstellen
#
class m_Fit(LP_object.m_Object):

    def __init__(self, data, model, fitType = 0, verbose = True,
                 method = 'levmar', stat=None, maxfev = None, fast=False,
                 tRange=[None,-1,1], chooseRange=False, plotResult=False,
                 derived_class=LP_object.m_Object, **kwargs):
        derived_class.__init__(self)
        self.data = data
        self.model = model
        self.fitType = fitType
        self.verbose = verbose
        self.maxfev = maxfev
        self.method = method
        self.stat = stat
        self.rerunParams = None
        self.fast=fast
        self.range=tRange
        self.ax = None
        self.choose=chooseRange
        self.plotres=plotResult
        self.coords = []

        if self.maxfev == None:
            self.maxfev = int(self.model.get_paramCount()*3e3)

        self.setFitType(fitType)

        self.kwargs=kwargs

    def setFitType(self,fitType):
        fType=['leastsq','odr1','odr2','odr3','sherpa','de','openopt']

        self.fitType = fitType
        if not SHERPA:
            if self.fitType == 4:
                self.fitType == 0

        if self.fitType == 0:
            self._run = self._fit_leastsq
            self.storeResult = self._store_leastsq
        elif self.fitType in (1,2,3):
            if self.fitType > 2:
                self.fitType -= 1
            self._run = self._fit_odr
            self.storeResult = self._store_odr
        elif self.fitType == 4:
            self._run = self._fit_sherpa
            self.set = self._sherpaSet
            self.get = self._sherpaGet
            if self.stat is not None:
                self.set(set_stat=self.stat)
            else:
                self.stat='chi2constvar'
                self.set(set_stat=self.stat)
            if self.maxfev is not None:
                if self.method == 'levmar':
                    self.set(set_method_opt=("maxfev",self.maxfev*50))
                else:
                    self.set(set_method_opt=("maxfev",self.maxfev))

            self.solver = sherpa
            self.storeResult = self._store_odr
        elif self.fitType == 5:
            self._run = self._fit_deSolver
            self.storeResult = self._store_deSolver
            self.set = self._deSolverSet
            self.get = self._deSolverGet
        else:
            self._run = self._fit_openopt
            self.storeResult = self._store_openopt

    def rerun(self,fit_is_OK=False):
        self.model.set_params()
        return self.run(fit_is_OK)

    def run(self, fit_is_OK=False):
        if self.choose:
            self.choose = self.plotChoose()

        sys.stdout.write('\nFitting ...')
        sys.stdout.flush()
        self.result = self._run()

        if self.plotres:
            self.plotResult()

        while not fit_is_OK:
            Fit_action = True
            self.result.pprint()
            while Fit_action:
                sys.stdout.write('\nIs Fit ok (y,n), do you want to change the fitting method(f)\n,do you want to plot(p) the result or just cancle (c)? [y/n/f/p/C]: ')
                raw_res=raw_input()

                if raw_res == 'n':
                    fit_is_OK = False
                    Fit_action = False
                elif raw_res == 'y':
                    Fit_action = False
                    fit_is_OK ==True
                    return True
                elif raw_res == 'p':
                    Fit_action = True
                    self.plotResult()
                elif raw_res == 'f':
                    sys.stdout.write('\nChange fit type to:\n')
                    sys.stdout.write('\t*Least Squares [0]\n')
                    sys.stdout.write('\t*ODR 1 [1]\n')
                    sys.stdout.write('\t*ODR 2 [2]\n')
                    sys.stdout.write('\t*ODR 3 [3]\n')
                    sys.stdout.write('\t*Sherpa [4]\n')
                    sys.stdout.write('\t\tSherpa params should be set before!\n')
                    sys.stdout.write('\t*Differetial evolution [5]\n')
                    sys.stdout.write('Choose: [0,1,2,3,4,5]: ')
                    raw_res=raw_input()

                    if raw_res in ['0','1','2','3','4','5']:
                        self.setFitType(int(raw_res))
                    Fit_action = False
                    fit_is_OK = False
                    sys.stdout.write('\nDo you want to change fitting conditions?')
                    sys.stdout.write('\n\tNo(n) or any fitting condition like')
                    sys.stdout.write('\n\tmaxfev = 300 [N/...]: ')
                    raw_res=raw_input()
                    if raw_res != 'n' and raw_res.count('=') > 0:
                        param, value = raw_res.split('=')
                        try:
                            exec('self.%s = %s' %(param,value))
                        except:
                            print 'Can´t set fitting condition "%s"' %raw_res
                            pass
                else:
                    if self.ax is not None:
                        close(self.ax.get_figure())
                    fit_is_OK ==True
                    self.model.reset_params()
                    return False

            self.result=self.rerun()

        self.model.set_params()
        return True

    def _close_to(self,d_xy,xy):
        num = min(abs(d_xy - xy))
        ind = where(num==abs(d_xy-xy))
        return ind

    def plotChoose(self):
        fit_range_ok = False
        _f=figure()
        _ax=_f.add_subplot(111)

        if self.data.kind is None or self.data.kind == 'linear':
            _ax.plot(self.data.x_org, self.data.y_org,'+-')
            xlabel(r'x')
            ylabel(r'y')


            while self.data.kind is None:
                sys.stdout.write('Replot with log axis?')
                sys.stdout.write('\n\t"ll" -> loglog, "ly" -> semilogy, "lx" -> semilogx: [ll/ly/lx/ENTER]:')
                sys.stdout.flush()
                res=raw_input()
                sys.stdout.write('\n')
                sys.stdout.flush()
                if res == 'll':
                    _ax.set_xscale('log')
                    _ax.set_yscale('log')
                    self.data.kind = 'loglog'
                    draw()
                elif res == 'ly':
                    _ax.set_yscale('log')
                    draw()
                    self.data.kind = 'semilogy'
                elif res == 'lx':
                   _ax.set_xscale('log')
                   draw()
                   self.data.kind = 'semilogx'
                else:
                    self.data.kind = 'linear'

        elif  self.data.kind == 'loglog':
            _ax.loglog(self.data.x_org, self.data.y_org)

        elif self.data.kind == 'semilogy':
            _ax.semilogy(self.data.x_org, self.data.y_org)

        elif self.data.kind == 'semilogx':
            _ax.semilogx(self.data.x_org, self.data.y_org)


        # set useblit True on gtkagg for enhanced performance
        span = selector(_ax)

        self.range[0], self.range[1] = searchsorted(self.data.x_org,(span.xmin, span.xmax))
        self.range[1]  = min(len(self.data.x_org)-1, self.range[1] )

        self.data.x = array(getObjRange(self.data.x_org, self.range[0], self.range[1], 1))
        self.data.y = array(getObjRange(self.data.y_org, self.range[0], self.range[1], 1))
        self.data.we = array(getObjRange(self.data.we_org, self.range[0], self.range[1], 1))

        close(_f)
        return True


    def plotResult(self, appendTo=None):
        if self.ax is None:
            if appendTo is None:
                _f=figure()
                _ax=_f.add_subplot(111)
            else:
                _ax=appedTo
        else:
            if fignum_exists(self.ax.figure.number):
                _ax = self.ax
            else:
                _f=figure()
                _ax=_f.add_subplot(111)

        if self.data.x.shape[0] > 100:
            _every = int(self.data.x.shape[0]/50.)
        else:
            _every = 1
        if self.data.kind is None or self.data.kind == 'linear':
            if self.ax is None:
                _ax.plot(self.data.x_org, self.data.y_org,'--+',markersize=12,linewidth=0.75,markevery=_every)
            else:
                for i in xrange(len(self.ax.get_lines()[:-1])):
                    self.ax.get_lines()[i+1].set_color('k')
            if self.data.x_org.shape <> self.result.xin.shape:
                _ax.plot(self.data.x_org, self.result.yfull)
            _ax.plot(self.result.xin, self.result.yout)

            while self.data.kind is None:
                sys.stdout.write('Replot with log axis?')
                sys.stdout.write( '"ll" -> loglog, "ly" -> semilogy, "lx" -> semilogx: [ll/ly/lx/ENTER]:')
                sys.stdout.flush()
                res=raw_input()
                if res == 'll':
                    _ax.set_xscale('log')
                    _ax.set_yscale('log')
                    self.data.kind = 'loglog'
                    draw()
                elif res == 'ly':
                    _ax.set_yscale('log')
                    draw()
                    self.data.kind = 'semilogy'
                elif res == 'lx':
                   _ax.set_xscale('log')
                   draw()
                   self.data.kind = 'semilogx'

        elif  self.data.kind == 'loglog':
            _ax.loglog(self.data.x_org, self.data.y_org,'--+',markersize=12,linewidth=0.75,markevery=_every)
            if self.data.x_org.shape <> self.result.xin.shape:
                _ax.loglog(self.data.x_org, self.result.yfull)
            _ax.loglog(self.result.xin, self.result.yout)

        elif self.data.kind == 'semilogy':
            _ax.semilogy(self.data.x_org, self.data.y_org,'--+',markersize=12,linewidth=0.75,markevery=_every)
            if self.data.x_org.shape <> self.result.xin.shape:
                _ax.semilogy(self.data.x_org, self.result.yfull)
            _ax.semilogy(self.result.xin, self.result.yout)

        elif self.data.kind == 'semilogx':
            _ax.semilogx(self.data.x_org, self.data.y_org,'--+',markersize=12,linewidth=0.75,markevery=_every)
            if self.data.x_org.shape <> self.result.xin.shape:
                _ax.semilogx(self.data.x_org, self.result.yfull)
            _ax.semilogx(self.result.xin, self.result.yout)

        show()
        draw()
        draw()
        show()
        #rainbow.map(_ax)
        self.ax = _ax

    def plotModelData(self, appendTo=None):
        if self.ax is None:
            if appendTo is None:
                _f=figure()
                _ax=_f.add_subplot(111)
            else:
                _ax=appedTo
        else:
            if fignum_exists(self.ax.figure.number):
                _ax = self.ax
            else:
                _f=figure()
                _ax=_f.add_subplot(111)

        if self.data.x.shape[0] > 100:
            _every = int(self.data.x.shape[0]/50.)
        else:
            _every = 1

        if self.data.kind is None or self.data.kind == 'linear':
            if self.ax is None:
                _ax.plot(self.data.x_org, self.data.y_org,'--+',markersize=12,linewidth=0.75,markevery=_every)
            else:
                for i in xrange(len(self.ax.get_lines()[:-1])):
                    self.ax.get_lines()[i+1].set_color('k')

            self.model.plot(self.data.x_org,'linear',_ax)

            while self.data.kind is None:
                sys.stdout.write('Replot with log axis?')
                sys.stdout.write( '"ll" -> loglog, "ly" -> semilogy, "lx" -> semilogx: [ll/ly/lx/ENTER]:')
                sys.stdout.flush()
                res=raw_input()
                if res == 'll':
                    _ax.set_xscale('log')
                    _ax.set_yscale('log')
                    self.data.kind = 'loglog'
                    draw()
                elif res == 'ly':
                    _ax.set_yscale('log')
                    draw()
                    self.data.kind = 'semilogy'
                elif res == 'lx':
                   _ax.set_xscale('log')
                   draw()
                   self.data.kind = 'semilogx'

        elif  self.data.kind == 'loglog':
            _ax.loglog(self.data.x_org, self.data.y_org,'--+',markersize=12,linewidth=0.75,markevery=_every)
            self.model.plot(self.data.x_org,'loglog',_ax)

        elif self.data.kind == 'semilogy':
            _ax.semilogy(self.data.x_org, self.data.y_org,'--+',markersize=12,linewidth=0.75,markevery=_every)
            self.model.plot(self.data.x_org,'semilogy',_ax)

        elif self.data.kind == 'semilogx':
            _ax.semilogx(self.data.x_org, self.data.y_org,'--+',markersize=12,linewidth=0.75,markevery=_every)
            self.model.plot(self.data.x_org,'semilogx',_ax)

        show()
        draw()
        draw()
        show()
        rainbow.map(_ax)
        self.ax = _ax

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # Definition of fit procedures
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


    #
    # Scipy ODR
    #
    def _fit_odr(self):
        #
        # Fitting function
        #
        def fcn(params, x):
            for i in xrange(len(params)):
                if abs(params[i]) == nan or abs(params[i]) == inf:
                    params[i] = self.model.x0[i]()['value']
                    return [1e100]*len(x)
                if params[i] < self.model.x0[i]()['min']:
                    params[i] = self.model.x0[i]()['min']
                    return [1e100]*len(x)
                if params[i] > self.model.x0[i]()['max']:
                    params[i] = self.model.x0[i]()['max']
                    return [1e100]*len(x)
            return self.model.fcn(params, x, self.model.const)
        #
        # create Model
        #
        model = odr.Model(fcn)
        #
        # Get a list with the fitting start parameters and create the
        # ODR fitting object

        self.solver = odr.ODR(self.data, model, beta0=self.model.params,
                              ifixb=self.model.froozen, maxit=self.maxfev)
        self.solver.set_job(fit_type=self.fitType-1)

        #
        # Run the fitting ...
        #
        output = self.solver.run()
        #
        # If result is not satisfying (output.info > 4) run fitting again
        #
        if(output.info >= 4):
            self.solver = odr.ODR(self.data, model, beta0=output.beta,
                                  ifixb=self.model.froozen, maxit=self.maxfev)
            self.solver.set_job(fit_type=self.fitType-1, restart=0)
            output = self.solver.run()
            if(output.info >= 4):
                if self.verbose:
                    ('ODRPACK still unable to find solution: \n%s\n\t\t'
                     %(output.stopreason) )


        for i in xrange(self.model.get_paramCount()):
            if not self.model.froozen[i]:
                self.model.params[i] = output.beta[i]

        self.storeResult(output)

        return self.result


    def _store_odr(self, fitres):

        result = m_Result()
        result['xin'] = self.data.x
        result['yin'] = self.data.y
        result['xinfull'] = self.data.x_org
        result['yinfull'] = self.data.y_org
        result['yout'] = array(self.model.fcn(list(self.model.params), self.data.x,
                              self.model.const))
        result['yfull'] = array(self.model.fcn(list(self.model.params), self.data.x_org,
                              self.model.const))
        result['params'] = self.model.params
        result['paramdict'] = LP_object.m_Dict()
        [result['paramdict'].set(name.split('.')[-1],v) for name,v in zip(self.model.names,self.model.params)]
        result['paramnames'] = tuple(name.split('.')[-1] for name in self.model.names)
        result['parmaxes'] = tuple(cov for cov in fitres.cov_beta)
        result['parmins'] = tuple(-1*cov for cov in fitres.cov_beta)
        result['fixed'] = tuple(result['paramnames'][i] for i in xrange(len(result['paramnames']))
                                if self.model.x0[i].froozen)
        result['info'] = fitres.info
        result['stopreason'] = fitres.stopreason
        result['fitres'] = fitres

        ret =   str(self.model)
        ret+='\n'
        if hasattr(fitres, 'info'):
            ret += 'Residual Variance: %s\n' %fitres.res_var
            ret += 'Inverse Condition #: %s\n' %fitres.inv_condnum
            ret += 'Reason(s) for Halting:\n'
            for r in fitres.stopreason:
                ret += '  %s\n' % r
        ret+='\n\n'

        result['pprint'] = lambda : sys.stdout.write(ret); sys.stdout.flush()
        result['format'] = ret

        self.result=result

    #
    # Scipy's curve_fit fitting
    #
    def _fit_curve_fit(self):
      #TODO
        pass



    if SHERPA:
        #
        # Sherpa fitting
        #

        def _sherpaSet(self, **kwargs):

            for key,val in kwargs.items():
                try:
                    exec('sherpa.%s("%s")'%(key,val))
                except:
                    try:
                        exec('sherpa.%s("%s",%s)'%(key,val[0],val[1]))
                    except:
                        exec('sherpa.%s("%s","%s")'%(key,val[0],val[1]))

        def _sherpaGet(self, keys):

            for key in LP_object._asarray(keys):
                try:
                    exec('return sherpa.get_%s'%(key))
                except:
                    pass



        def _fit_sherpa(self):
            #
            # Fitting function
            #
            def fcn(params, x):
                return self.model.fcn(params, x, self.model.const)
            #
            # set data
            #
            self.solver=sherpa
            self.solver.set_data(self.solver.Data1D('Default', self.data.x, self.data.y))
            #
            # create model
            #
            self.solver.load_user_model(fcn, "fmdl")
            #
            # set startparameter
            #
            self.solver.add_user_pars("fmdl", self.model.names, self.model.params,
                                        self.model.mins, self.model.maxs,
                                        self.model.units, self.model.froozen)
            self.solver.set_model(fmdl)

            if len(self.kwargs) > 0:
                self.set(**self.kwargs)
            #
            # set method for fitting (levmar, moncar, simplex)
            # if simplex or moncar is used try first to get good
            # starting values with levmar
            fitres,errestres = (None,None)
            for method in LP_object._asarray(self.method):
                if method in ('levmar','simplex','moncar'):
                    self.set(set_method=method)
                    if method=='moncar':
                        self.set(set_method_opt=("maxfev",self.maxfev/100))
                    elif method == 'levmar':
                        self.set(set_method_opt=("maxfev",self.maxfev*50))
                    elif method == 'simplex':
                        self.set(set_method_opt=("maxfev",self.maxfev))
                    else: print '! Fit method unknown: %s' %method
                    fitres=self.solver.run()
                elif method in ('proj','covar','conf') and self.stat <> 'leastsq':

                    #~ try:
                        if method == 'proj':
                            self.solver.set_proj_opt ('tol',1e-1)
                            self.solver.set_proj_opt ('eps',1e-2)
                            self.solver.set_proj_opt ('sigma',3)
                            errestres = self.solver.proj()
                        elif method == 'conf':
                            self.solver.set_conf_opt ('tol',1e-1)
                            self.solver.set_conf_opt ('eps',1e-2)
                            self.solver.set_conf_opt ('sigma',3)
                            errestres = self.solver.conf()
                        elif method == 'covar':
                            self.solver.set_covar_opt ('eps',1e-2)
                            self.solver.set_covar_opt ('sigma',3)
                            errestres = self.solver.covar()
                        else: print '! Error estimation method unknown: %s' %method
                    #~ except:
                       #~ pass

            j=0
            for i in xrange(self.model.get_paramCount()):
                if not self.model.froozen[i]\
                    and self.model.params[i] <> fitres.parvals[j]:
                    self.model.params[i] = fitres.parvals[j]
                    j+=1

            self.storeResult(fitres)

            return self.result

    def _store_deSolver(self, fitres):

        result = m_Result()
        result['xin'] = self.data.x
        result['yin'] = self.data.y
        result['xinfull'] = self.data.x_org
        result['yinfull'] = self.data.y_org
        result['yout'] = array(self.model.fcn(self.model.params, self.data.x,
                              self.model.const))
        result['yfull'] = array(self.model.fcn(list(self.model.params), self.data.x_org,
                              self.model.const))
        result['succeeded'] = (True if fitres.goal_error > fitres.best_error
                               else False)
        result['paramnames'] = tuple(name.split('.')[-1] for name in self.model.names)
        result['params'] = self.model.params
        result['paramdict'] = LP_object.m_Dict()
        [result['paramdict'].set(name.split('.')[-1],v) for name,v in zip(self.model.names,self.model.params)]
        result['fixed'] = tuple(result['paramnames'][i] for i in xrange(len(result['paramnames']))
                                if self.model.x0[i].froozen)
        result['chi2'] = fitres.best_error
        result['nfev'] = fitres.best_individual
        result['method'] = fitres.method
        result['fitres'] = fitres

        ret = '\nNumber of generations: %s - Best generation: %s\n' %(fitres.generation, fitres.best_generation)
        ret +=   str(self.model)
        ret+='\n'

        result['pprint'] = lambda : sys.stdout.write(ret); sys.stdout.flush()
        result['format'] = ret

        self.result=result


    def _fit_deSolver(self):
        #~ #Fit with scipy.optimize
        #~ #
        import DESolver;    reload(DESolver)
        # data, model, population_size, max_generations,
        # method = DE_RAND_1, seed=None,
        # param_names = None, scale=[0.5,1.0], crossover_prob=0.9,
        # goal_error=1e-3, polish=True, verbose=True,
        # use_pp=True, pp_depfuncs=None, pp_modules=None,
        # pp_proto=2, pp_ncpus='autodetect'

        self.solver = DESolver.DESolver(self.data, self.model,
                            self.kwargs.get('popsize',self.model.get_paramCount()*50),
                            self.maxfev, DESolver.DE_BEST_1, scale=[0.5,1.0],
                            crossover_prob=0.8, goal_error=1e-7, polish=True,
                            verbose=False, parallel=False)

        if len(self.kwargs) > 0:
            self.set(**self.kwargs)

        result = m_Result()
        result['pspace'] = copy.copy(self.solver.population.transpose())

        self.solver.Solve()

        if self.solver.best_error < 1e120:
            for i in xrange(self.model.get_paramCount()):
                if not self.model.froozen[i]:
                    self.model.params[i] = self.solver.best_individual[i]

        self.storeResult(self.solver)

        return self.result

    def _deSolverSet(self, **kwargs):
        for key,val in kwargs.items():
            try:
                exec('self.solver.%s = %s'%(key,val))
            except:
                try:
                    exec('self.solver.%s = "%s"'%(key,val))
                except:
                    print 'Value "%s" not set : %s' %(key,val)
        self.solver.initialize()

    def _deSolverGet(self, keys):

        for key in LP_object._asarray(keys):
            try:
                exec('return self.solver.%s'%(key))
            except:
                pass



    def _store_deSolver(self, fitres):

        result = m_Result()
        result['xin'] = self.data.x
        result['yin'] = self.data.y
        result['xinfull'] = self.data.x_org
        result['yinfull'] = self.data.y_org
        result['yout'] = array(self.model.fcn(self.model.params, self.data.x,
                              self.model.const))
        result['yfull'] = array(self.model.fcn(list(self.model.params), self.data.x_org,
                              self.model.const))
        result['succeeded'] = (True if fitres.goal_error > fitres.best_error
                               else False)
        result['paramnames'] = tuple(name.split('.')[-1] for name in self.model.names)
        result['params'] = self.model.params
        result['paramdict'] = LP_object.m_Dict()
        [result['paramdict'].set(name.split('.')[-1],v) for name,v in zip(self.model.names,self.model.params)]
        result['fixed'] = tuple(result['paramnames'][i] for i in xrange(len(result['paramnames']))
                                if self.model.x0[i].froozen)
        result['chi2'] = fitres.best_error
        result['nfev'] = fitres.best_individual
        result['method'] = fitres.method
        result['fitres'] = fitres

        ret = '\nNumber of generations: %s - Best generation: %s\n' %(fitres.generation, fitres.best_generation)
        ret +=   str(self.model)
        ret+='\n'

        result['pprint'] = lambda : sys.stdout.write(ret); sys.stdout.flush()
        result['format'] = ret

        self.result=result


    #
    # NumpyÂ´s leastsq fitting
    #
    def _fit_leastsq(self):
        from scipy import optimize

        def f(params,x,y):
            for i in xrange(len(params)):
                if self.model.x0[i]()['froozen']:
                    params[i] = self.model.x0[i]()['value']
                else:
                    if abs(params[i]) == nan or abs(params[i]) == inf:
                        params[i] = self.model.x0[i]()['value']
                        return [1e100]*len(x)
                    if params[i] < self.model.x0[i]()['min']:
                        params[i] = self.model.x0[i]()['min']
                        return [1e100]*len(x)
                    if params[i] > self.model.x0[i]()['max']:
                        params[i] = self.model.x0[i]()['max']
                        return [1e100]*len(x)
            if self.data.we is not None:
                return (y-self.model.fcn(params,x,self.model.const))**2 * self.data.we
            else:
                return(y-self.model.fcn(params,x,self.model.const))

        params = []
        for p in self.model.x0:
            params.append(p()['value'])


        output = optimize.leastsq(f, params,args=(self.data.x, self.data.y),maxfev=self.maxfev, full_output=0)
        if output[-1] < 4: #rerun from stop
            if not isnan(output[0][0]):
                output = optimize.leastsq(f, params,args=(self.data.x, self.data.y),maxfev=self.maxfev, full_output=0)

        for i in xrange(self.model.get_paramCount()):
            if not self.model.froozen[i]:
                self.model.params[i] = output[0][i]

        self.storeResult(output)

        return self.result

    def _store_leastsq(self, fitres):

        result = m_Result()
        result['xin'] = self.data.x
        result['yin'] = self.data.y
        result['xinfull'] = self.data.x_org
        result['yinfull'] = self.data.y_org
        result['yout'] = array(self.model.fcn(self.model.params, self.data.x,
                              self.model.const))
        result['yfull'] = array(self.model.fcn(list(self.model.params), self.data.x_org,
                              self.model.const))
        result['succeeded'] = (True if fitres[1] < 4 else False)
        result['params'] = self.model.params
        result['paramdict'] = LP_object.m_Dict()
        [result['paramdict'].set(name.split('.')[-1],v) for name,v in zip(self.model.names,
                                                                          self.model.params)]
        result['paramnames'] = tuple(name.split('.')[-1] for name in self.model.names)
        result['fixed'] = tuple(result['paramnames'][i] for i in xrange(len(result['paramnames']))
                                if self.model.x0[i].froozen)
        result['fitres'] = fitres[1]

        ret =   str(self.model)
        ret+='\n'

        result['pprint'] = lambda : sys.stdout.write(ret); sys.stdout.flush()
        result['format'] = ret

        self.result=result

    #
    # Openopt fitting
    #
    def _fit_openopt(self):

        from openopt import NLP


        solver = self.kwargs.get('solver','ralg')
        #~ self.args += args[1:]
        #~ self.kwargs.update(kwargs)
        self.err = [1e10,1e9,1e8,1e7]
        self.i = 0
        params = []
        for p in self.model.x0:
            params.append(p()['value'])

        def f(params):
            for i in xrange(len(params)):
                if self.model.x0[i]()['froozen']:
                    params[i] = self.model.x0[i]()['value']
                else:
                    if abs(params[i]) == nan or abs(params[i]) == inf:
                        params[i] = self.model.x0[i]()['value']
                        self.i = (self.i+1)%4
                        return sum([self.err[self.i]]*len(self.data.x))
                    if params[i] < self.model.x0[i]()['min']:
                        params[i] = self.model.x0[i]()['min']
                        self.i = (self.i+1)%4
                        return sum([self.err[self.i]]*len(self.data.x))
                    if params[i] > self.model.x0[i]()['max']:
                        params[i] = self.model.x0[i]()['max']
                        self.i = (self.i+1)%4
                        return sum([self.err[self.i]]*len(self.data.x))
            if self.data.we is not None:
                return sum((self.data.y-self.model.fcn(params,self.data.x,self.model.const))**2 * self.data.we)
            else:
                return sum(self.data.y-self.model.fcn(params,self.data.x,self.model.const)**2)

        #~ f = lambda p: sum(((self.func(p, self.x) - self.y) / self.weights)**2)
        self.prob = NLP(f, x0 = params)
        self.res = self.prob.solve(solver)

        self.storeResult(self.res)

        return self.result


    def _store_openopt(self, fitres):

        result = m_Result()
        result['xin'] = self.data.x
        result['yin'] = self.data.y
        result['xinfull'] = self.data.x_org
        result['yinfull'] = self.data.y_org
        result['yout'] = array(self.model.fcn(self.model.params, self.data.x,
                              self.model.const))
        result['yfull'] = array(self.model.fcn(list(self.model.params), self.data.x_org,
                              self.model.const))
        result['succeeded'] = (True if fitres.stopcase > -1 else False)
        result['params'] = self.model.params
        result['paramdict'] = LP_object.m_Dict()
        [result['paramdict'].set(name.split('.')[-1],v) for name,v in zip(self.model.names,
                                                                          self.model.params)]
        result['paramnames'] = tuple(name.split('.')[-1] for name in self.model.names)
        result['fixed'] = tuple(result['paramnames'][i] for i in xrange(len(result['paramnames']))
                                if self.model.x0[i].froozen)
        result['fitres'] = fitres.xf

        ret =   str(self.model)
        ret+='\n'

        result['pprint'] = lambda : sys.stdout.write(ret); sys.stdout.flush()
        result['format'] = ret

        self.result=result





class m_Coords:
    def __init__(self,fig, vis=False):
        import sys
        self.f=sys.stdout
        self.figure = fig
        self.data=LP_object.m_Dict()
        print(type(self.data))
        self.event=None
        self.vis = vis
        self.marker=LP_object.m_Dict()
        self.morder=[]
        self.cid1 = self.figure.canvas.mpl_connect('button_press_event',
                                                  self._onclick)
        self.cid2 = self.figure.canvas.mpl_connect('pick_event',
                                                  self._onpick)

    def _onclick(self,event):
        if event.button == 2:
            self._release()
            return True

    def _onpick(self,event):
        from numpy import mean
        if event.mouseevent.button == 2:
            self._release()
            return True
        self.event=event
        self.artist=event.artist
        self.mouse=event.mouseevent
        self.data.set('x',self.mouse.x)
        self.data.set('y',self.mouse.y)
        self.data.set('xcoord',self.mouse.xdata)
        self.data.set('ycoord',self.mouse.ydata)
        self.data.set('xdata',self.artist.get_xdata()[event.ind])
        self.data.set('ydata',self.artist.get_ydata()[event.ind])
        self.data.set('ind',event.ind)
        if event.mouseevent.button == 1:
            if len(self.marker)<4:
                self.figure.axes[0].hold(True)
                line=self.figure.axes[0].plot([mean(self.data.xdata)],[mean(self.data.ydata)],'r.',picker=2)[0]
                self.figure.canvas.draw()
                self.morder.append(line)
                self.marker[line]=LP_object.m_Dict(x=[mean(self.data.xdata)],y=[mean(self.data.ydata)])
            else:
                l=self.morder.pop(0)
                self.figure.axes[0].hold(True)
                line=self.figure.axes[0].plot([mean(self.data.xdata)],[mean(self.data.ydata)],'r.',picker=2)[0]
                self.figure.canvas.draw()
                self.morder.append(line)
                self.marker[line]=LP_object.m_Dict(x=[mean(self.data.xdata)],y=[mean(self.data.ydata)])
                l.remove()
                self.marker.pop(l)

        if event.mouseevent.button == 3:
            if self.marker.has_key(self.artist):
                self.morder.remove(self.artist)
                self.artist.remove()
                self.marker.pop(self.artist)
            else:
                l=self.morder.pop(-1)
                l.remove()
                self.marker.pop(l)
            self.figure.canvas.draw()
#~
        if self.vis:
            self.f.write('\nid=%d, x=%d, y=%d,\nxcoord=%.4e, ycoord=%.4e,\nxdata=%.4e, ydata=%.4e'%(
            int(mean(event.idn)), self.data.x,self.data.y, self.data.xcoord,
                self.data.ycoord, mean(self.data.xdata), mean(self.data.ydata)))
            self.f.flush()


    def _release(self):
        self.figure.canvas.mpl_disconnect(self.cid1)
        self.figure.canvas.mpl_disconnect(self.cid2)

def __getObjRange(obj, start=None, stop=-1, step=1):
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

def mfit(func, ax, start=None, stop=-1, step=1, *args, **kwargs):

    isnotnan = lambda x: (isnan(x))==False

    if ax != Subplot:
        ax=gca()
    if start == 'h' or start == 'H':
        start = len(ax.get_lines())/2
    if stop == 'h' or stop == 'H':
        stop = len(ax.get_lines())/2

    list=__getObjRange(ax.get_lines(), start, stop, step)

    from scipy.optimize import curve_fit
    fit_params = []
    for i,l in enumerate(list):
        old_c = l.get_color()
        old_ls = l.get_linestyle()
        old_m = l.get_marker()
        old_mfc = l.get_markerfacecolor()
        old_ms = l.get_markersize()

        l.set_color('#555555')
        l.set_linestyle(':')
        l.set_marker('o')
        l.set_markerfacecolor('None')
        l.set_markersize(9)
        draw()

        span = selector(gca())

        range = [0,0]
        xdata = l.get_xdata()[isnotnan(l.get_xdata())]
        ydata = l.get_ydata()[isnotnan(l.get_xdata())]
        xdata = xdata[isnotnan(ydata)]
        ydata = ydata[isnotnan(ydata)]

        # sort data
        idxs =argsort(xdata)
        xdata = xdata[idxs]
        ydata = ydata[idxs]
        we = 1.0#/ydata

        range[0], range[1] = searchsorted(xdata,(span.xmin, span.xmax))
        range[1]  = min(len(xdata)-1, range[1] )

        print range[0], range[1], span.xmin, span.xmax, xdata[0], xdata[-1]


        xdata_fit = array(getObjRange(xdata, range[0], range[1], 1))
        ydata_fit = array(getObjRange(ydata, range[0], range[1], 1))
        we = array(getObjRange(we, range[0], range[1], 1))



        if len(kwargs.keys()) > 0:
            def f(x, *args):
                return func(x, *args, **kwargs)
        else:
            def f(x, *args):
                return func(x, *args)

        try:
            p, cov = curve_fit(f, xdata_fit, ydata_fit, p0=args, sigma=we)
        except:
            p, cov = args, None

        print p
        fit_params.append(p)

        del span
        l.set_color(old_c)
        l.set_linestyle(old_ls)
        l.set_marker(old_m)
        l.set_markerfacecolor(old_mfc)
        l.set_markersize(old_ms)

        draw()

        if cov is not None:
            if gca().xaxis.get_scale() == 'linear':
                x_plot = np.linspace(xdata[0], xdata[-1], 300)
            else:
                 x_plot = np.logspace(log10(xdata[0]), log10(xdata[-1]), 300)

            color_idx = (2*i*10/len(list))%10
            plot(x_plot, f(x_plot, *p),':',c='#%s%s%s%s%s%s'%(color_idx,color_idx,color_idx,color_idx,color_idx,color_idx), lw=2)

    show()

    return fit_params

