# -*- coding: utf-8 -*-
#
# Copyright © 2009-2010 CEA
# Pierre Raybaut
# Licensed under the terms of the CECILL License
# (see guiqwt/__init__.py for details)

"""
guiqwt.fit
----------

The `fit` module provides an interactive curve fitting tool allowing:
    * to fit data manually (by moving sliders)
    * or automatically (with standard optimization algorithms
      provided by :py:mod:`scipy`).

Example
~~~~~~~

.. literalinclude:: ../guiqwt/tests/fit.py
   :start-after: SHOW
   :end-before: Workaround for Sphinx v0.6 bug: empty 'end-before' directive

.. image:: images/screenshots/fit.png

Reference
~~~~~~~~~

.. autofunction:: guifit

.. autoclass:: FitDialog
   :members:
   :inherited-members:
.. autoclass:: FitParam
   :members:
   :inherited-members:
.. autoclass:: AutoFitParam
   :members:
   :inherited-members:
"""

from __future__ import division

from PyQt4.QtGui import (QGridLayout, QLabel, QSlider, QPushButton, QLineEdit,
                         QDialog, QVBoxLayout, QHBoxLayout, QWidget,
                         QDialogButtonBox, QTabWidget, QSplitter)
from PyQt4.QtCore import Qt, SIGNAL, QObject, SLOT

import numpy as np
from numpy import inf # Do not remove this import (used by optimization funcs)
import sys

import guidata
from guidata.utils import update_dataset, restore_dataset
from guidata.qthelpers import create_groupbox
from guidata.configtools import get_icon
from guidata.dataset.datatypes import DataSet
from guidata.dataset.dataitems import (StringItem, FloatItem, IntItem,
                                       ChoiceItem, BoolItem)

# Local imports
from guiqwt.config import _
from guiqwt.builder import make
from guiqwt.plot import CurveWidgetMixin
from guiqwt.signals import SIG_RANGE_CHANGED

class AutoFitParam(DataSet):
    xmin = FloatItem("xmin")
    xmax = FloatItem("xmax")
    method = ChoiceItem(_("Method"),
                        [ ("simplex", "Simplex"), ("powel", "Powel"),
                          ("bfgs", "BFGS"), ("l_bfgs_b", "L-BFGS-B"),
                          ("cg", _("Conjugate Gradient")),
                          ("ncg", _("Newton Conjugate Gradient")),
                          ("lq", _("Least squares")), ],
                        default="lq")
    maxfun = IntItem("maxfun", default=20000,
                     help=_("Maximum of function evaluation. for simplex, powel, least squares, cg, bfgs, l_bfgs_b"))
    maxiter = IntItem("maxiter", default=20000,
                     help=_("Maximum of iterations. for simplex, powel, least squares, cg, bfgs, l_bfgs_b"))
    err_norm = StringItem("enorm", default='2.0',
                          help=_("for simplex, powel, cg and bfgs norm used "
                                 "by the error function"))
    xtol = FloatItem("xtol", default=0.0001,
                     help=_("for simplex, powel, least squares"))
    ftol = FloatItem("ftol", default=0.0001,
                     help=_("for simplex, powel, least squares"))
    gtol = FloatItem("gtol", default=0.0001, help=_("for cg, bfgs"))
    norm = StringItem("norm", default="inf",
                      help=_("for cg, bfgs. inf is max, -inf is min"))


class FitParamDataSet(DataSet):
    name = StringItem(_("Name"))
    value = FloatItem(_("Value"), default=0.0)
    min = FloatItem(_("Min"), default=-1.0)
    max = FloatItem(_("Max"), default=1.0).set_pos(col=1)
    steps = IntItem(_("Steps"), default=5000)
    format = StringItem(_("Format"), default="%.3f").set_pos(col=1)
    logscale = BoolItem(_("Logarithmic"), _("Scale"))
    unit = StringItem(_("Unit"), default="").set_pos(col=1)

class FitParam(object):
    def __init__(self, name, value, mini, maxi, logscale=False,
                 steps=5000, format='%.3f', size_offset=0, unit=''):
        self.name = name
        self.value = value
        self.min = mini if logscale==False else max(1e-120,mini)
        self.max = maxi
        self.logscale = logscale
        self.steps = steps
        self.format = format
        self.unit = unit
        self.prefix_label = None
        self.lineedit = None
        self.unit_label = None
        self.slider = None
        self.button = None
        self._widgets = []
        self._size_offset = size_offset
        self._refresh_callback = None
        self.dataset = FitParamDataSet(title=_("Curve fitting parameter"))

    def copy(self):
        """Return a copy of this fitparam"""
        return self.__class__(self.name, self.value, self.min, self.max,
                              self.logscale, self.steps, self.format,
                              self._size_offset, self.unit)

    def create_widgets(self, parent, refresh_callback):
        self._refresh_callback = refresh_callback
        self.prefix_label = QLabel()
        font = self.prefix_label.font()
        font.setPointSize(font.pointSize()+self._size_offset)
        self.prefix_label.setFont(font)
        self.button = QPushButton()
        self.button.setIcon(get_icon('settings.png'))
        self.button.setToolTip(
                        _("Edit '%s' fit parameter properties") % self.name)
        QObject.connect(self.button, SIGNAL('clicked()'),
                        lambda: self.edit_param(parent))
        self.lineedit = QLineEdit()
        QObject.connect(self.lineedit, SIGNAL('editingFinished()'),
                        self.line_editing_finished)
        self.unit_label = QLabel(self.unit)
        self.slider = QSlider()
        self.slider.setOrientation(Qt.Horizontal)
        self.slider.setRange(0, self.steps-1)
        QObject.connect(self.slider, SIGNAL("valueChanged(int)"),
                        self.slider_value_changed)
        self.update(refresh=False)
        self.add_widgets([self.prefix_label, self.lineedit, self.unit_label,
                          self.slider, self.button])

    def add_widgets(self, widgets):
        self._widgets += widgets

    def get_widgets(self):
        return self._widgets

    def set_scale(self, state):
        self.logscale = state > 0
        self.update_slider_value()

    def set_text(self, fmt=None):
        style = "<span style=\'color: #444444\'><b>%s</b></span>"
        self.prefix_label.setText(style % self.name)
        if self.value is None:
            value_str = ''
        else:
            if fmt is None:
                fmt = self.format
            value_str = fmt % self.value
        self.lineedit.setText(value_str)
        self.lineedit.setDisabled(
                            self.value == self.min and self.max == self.min)

    def line_editing_finished(self):
        try:
            self.value = float(self.lineedit.text())
        except ValueError:
            self.set_text()
        self.update_slider_value()
        self._refresh_callback()

    def slider_value_changed(self, int_value):
        if self.logscale:
            #~ total_delta = np.log10(1+self.max-self.min)
            #~ self.value = self.min+10**(total_delta*int_value/(self.steps-1))-1
            #~ total_delta = np.log10(self.max)-np.log10(self.min)
            ratio = int_value/(self.steps-1)
            self.value = self.max**ratio * self.min**(1-ratio)
        else:
            total_delta = self.max-self.min
            self.value = self.min+total_delta*int_value/(self.steps-1)
        self.set_text()
        self._refresh_callback()

    def update_slider_value(self):
        from numpy import isnan, isinf
        if (self.value is None or self.min is None or self.max is None):
            self.slider.setEnabled(False)
            if self.slider.parent() and self.slider.parent().isVisible():
                self.slider.show()
        elif self.value == self.min and self.max == self.min:
            self.slider.hide()
        else:
            self.slider.setEnabled(True)
            if self.slider.parent() and self.slider.parent().isVisible():
                self.slider.show()
            if self.logscale:
                value_delta = max([np.log10(self.value/self.min), 0.])
                total_delta = np.log10(self.max/self.min)
                if not isnan(self.steps*value_delta/total_delta):
                    intval = int(self.steps*value_delta/total_delta)
                else:
                    intval = int(self.min)
            else:
                value_delta = self.value-self.min
                total_delta = self.max-self.min
                intval = int(self.steps*value_delta/total_delta)
            self.slider.blockSignals(True)
            print intval
            print
            sys.stdout.flush()
            self.slider.setValue(intval)
            self.slider.blockSignals(False)

    def edit_param(self, parent):
        update_dataset(self.dataset, self)
        if self.dataset.edit(parent=parent):
            restore_dataset(self.dataset, self)
            if self.value > self.max:
                self.max = self.value
            if self.value < self.min:
                self.min = self.value
            self.update(True)

    def update(self, refresh=True):
        self.unit_label.setText(self.unit)
        self.slider.setRange(0, self.steps-1)
        self.update_slider_value()
        self.set_text()
        if refresh:
            self._refresh_callback()


def add_fitparam_widgets_to(layout, fitparams, refresh_callback, param_cols=1):
    row_contents = []
    row_nb = 0
    col_nb = 0
    for i, param in enumerate(fitparams):
        param.create_widgets(layout.parent(), refresh_callback)
        widgets = param.get_widgets()
        w_colums = len(widgets)+1
        row_contents += [(widget, row_nb, j+col_nb*w_colums)
                         for j, widget in enumerate(widgets)]
        col_nb += 1
        if col_nb == param_cols:
            row_nb += 1
            col_nb = 0
    for widget, row, col in row_contents:
        layout.addWidget(widget, row, col)
    if fitparams:
        for col_nb in range(param_cols):
            layout.setColumnStretch(1+col_nb*w_colums, 5)
            if col_nb > 0:
                layout.setColumnStretch(col_nb*w_colums-1, 1)

class FitWidgetMixin(CurveWidgetMixin):
    def __init__(self, wintitle="guiqwt plot", icon="guiqwt.png",
                 toolbar=False, options=None, panels=None, param_cols=1,
                 legend_anchor='TR', auto_fit=True, show_range=False):
        if wintitle is None:
            wintitle = _('Curve fitting')

        self.x = None
        self.y = None
        self.sigma = None
        self.fitfunc = None
        self.fitnames = []
        self.fitargs = None
        self.fitkwargs = None
        self.fitparams = None
        self.autofit_prm = None

        self.data_curve = None
        self.fit_curves = []
        self.fit_curve = None
        self.legend = None
        self.legend_anchor = legend_anchor
        self.xrange = None
        self.show_xrange = show_range

        self.param_cols = param_cols
        self.auto_fit_enabled = auto_fit
        self.button_list = [] # list of buttons to be disabled at startup

        self.fit_layout = None
        self.params_layouts = []

        CurveWidgetMixin.__init__(self, wintitle=wintitle, icon=icon,
                                  toolbar=toolbar, options=options,
                                  panels=panels)

        self.refresh()

    # QWidget API --------------------------------------------------------------
    def resizeEvent(self, event):
        QWidget.resizeEvent(self, event)
        self.get_plot().replot()

    # CurveWidgetMixin API -----------------------------------------------------
    def setup_widget_layout(self):
        self.tabs = QTabWidget()
        self.connect(self.tabs,SIGNAL('currentChanged(int)'), self.changeFit)
        self.fit_layout = QSplitter(self)
        if self.auto_fit_enabled:
            auto_group = self.create_autofit_group()
            self.fit_layout.addWidget(auto_group)

        self.fit_layout.addWidget(self.tabs)
        self.plot_layout.addWidget(self.fit_layout, 1, 0)

        vlayout = QVBoxLayout(self)
        vlayout.addWidget(self.toolbar)
        vlayout.addLayout(self.plot_layout,0)
        self.setLayout(vlayout)

        #~ self.tabs = QTabWidget()
        #~ self.connect(self.tabs,SIGNAL('currentChanged(int)'), self.changeFit)
        #~ self.fit_layout = QSplitter(self)
        #~ if self.auto_fit_enabled:
            #~ auto_group = self.create_autofit_group()
            #~ self.fit_layout.addWidget(auto_group)
            #~
        #~ self.fit_layout.addWidget(self.tabs)
                #~
        #~ self.plot_widget = QWidget()
        #~ self.plot_widget.setLayout(self.plot_layout)
 #~
        #~ vlayout = QSplitter(Qt.Vertical)
        #~ vlayout.addWidget(self.toolbar)
        #~ vlayout.addWidget(self.plot_widget)
        #~ vlayout.addWidget(self.fit_layout)
        #~ tt = QVBoxLayout()
        #~ tt.addWidget(vlayout)
        #~ self.setLayo

    def create_plot(self, options):
        super(FitWidgetMixin, self).create_plot(options)
        for plot in self.get_plots():
            self.connect(plot, SIG_RANGE_CHANGED, self.range_changed)

    # Public API ---------------------------------------------------------------
    def set_data(self, x, y, fitfunc=None, fitparams=None,
                 fitargs=None, fitkwargs=None, xmin=None, xmax=None, sigma=None, **autofit_params):

        if self.fitparams is not None and fitparams is not None:
            self.clear_params_layout()

        if not isinstance(fitfunc,(tuple, list, np.ndarray,dict)):
            fitfunc = [fitfunc]
            fitparams = [fitparams]
        elif isinstance(fitfunc,dict):
            self.fitnames = fitfunc.keys()
            fitfunc = fitfunc.values()
            #fitparams = fitparams.values()


        if self.fitnames == []:
            self.fitnames = ['Fit %d' %(i+1) for i in xrange(len(fitfunc))]
        self._idxlen=len(fitfunc)
        self.x = x
        self.y = y

        if sigma is None:
            sigma = np.ones(x.shape)
        self.sigma = sigma

        if fitfunc is not None:
            self.fitfunc = fitfunc
        if fitparams is not None:
            self.fitparams = fitparams
        if fitargs is not None:
            self.fitargs = fitargs
        if fitkwargs is not None:
            self.fitkwargs = fitkwargs

        self.autofit_prm = AutoFitParam(title=_("Automatic fitting options"))
        if xmin is None:
            self.autofit_prm.xmin = x.min()
        else:
            self.autofit_prm.xmin = xmin
        if xmax is None:
            self.autofit_prm.xmax = x.max()
        else:
            self.autofit_prm.xmax = xmax

        self.compute_imin_imax()

        if self.fitparams is not None and fitparams is not None:
            for i in xrange(self._idxlen):
                self.params_layouts.append( QGridLayout() )
                params_group = create_groupbox(self, _(""),
                                           layout=self.params_layouts[i])
                self.tabs.addTab(params_group,_('%s') %self.fitnames[i])

            self.populate_params_layout()

        if autofit_params is not None:
            self.set_autofit_param(**autofit_params)

        self.refresh()

    def set_fit_data(self, fitfunc, fitparams, fitargs=None, fitkwargs=None):
        if self.fitparams is not None:
            self.clear_params_layout()
        self.fitfunc = fitfunc
        self.fitparams = fitparams
        self.fitargs = fitargs
        self.fitkwargs = fitkwargs
        self.populate_params_layout()
        self.refresh()

    def clear_params_layout(self):
        for i, param in enumerate(self.fitparams[self.idx]):
            print param, i
            for widget in param.get_widgets():
                if widget is not None:
                    self.params_layouts[self.idx].removeWidget(widget)
                    widget.hide()

    def populate_params_layout(self):
        [add_fitparam_widgets_to(self.params_layouts[i], self.fitparams[i],
                                self.refresh, param_cols=self.param_cols) for i in xrange(self._idxlen)]

    #~ def clear_params_layout(self):
       #~ remove_fitparam_widgets_from(self.params_layout, self.fitparams[(self.idx-1)%self._idxlen])

    def create_autofit_group(self):
        auto_button = QPushButton(get_icon('apply.png'), _("Run"), self)
        self.connect(auto_button, SIGNAL("clicked()"), self.autofit)
        autoprm_button = QPushButton(get_icon('settings.png'), _("Settings"),
                                     self)
        self.connect(autoprm_button, SIGNAL("clicked()"), self.edit_parameters)
        xrange_button = QPushButton(get_icon('xrange.png'), _("Bounds"), self)
        xrange_button.setCheckable(True)
        self.connect(xrange_button, SIGNAL("toggled(bool)"), self.toggle_xrange)

        auto_layout = QVBoxLayout()
        auto_layout.addWidget(auto_button)
        auto_layout.addWidget(autoprm_button)
        auto_layout.addWidget(xrange_button)
        self.button_list += [auto_button, autoprm_button, xrange_button]
        return create_groupbox(self, _("Automatic fit"), layout=auto_layout)

    def get_fitfunc_arguments(self):
        """Return fitargs and fitkwargs"""
        fitargs = self.fitargs
        if self.fitargs is None:
            fitargs = []
        fitkwargs = self.fitkwargs
        if self.fitkwargs is None:
            fitkwargs = {}
        return fitargs, fitkwargs

    def refresh(self, slider_value=None):
        """Refresh Fit Tool dialog box"""
        # Update button states
        enable = self.x is not None and self.y is not None \
                 and self.x.size > 0 and self.y.size > 0 \
                 and self.fitfunc is not None and self.fitparams is not None \
                 and len(self.fitparams) > 0
        for btn in self.button_list:
            btn.setEnabled(enable)

        if not enable:
            # Fit widget is not yet configured
            return

        plot = self.get_plot()

        if self.xrange is None:
            self.xrange = make.range(0., 1.)
            plot.add_item(self.xrange)

        self.xrange.set_range(self.autofit_prm.xmin, self.autofit_prm.xmax)
        self.xrange.setVisible(self.show_xrange)

        if self.data_curve is None:
            self.data_curve = make.curve([], [],
                                         _("Data"), color="b", marker='o',
                                         markerfacecolor='w', markersize=8,
                                         linestyle='DashLine', linewidth=1.)
            plot.add_item(self.data_curve)
        self.data_curve.set_data(self.x, self.y)

        colors = [u'#d4f200',u'#00e639',u'#009fff',u'#7900f2']

        for i in xrange(self._idxlen):
            if len(self.fit_curves) < self._idxlen:
                self.fit_curves.append(make.curve([], [],
                                    _('%s' %self.fitnames[i]), color=colors[i%len(colors)], linestyle='--', linewidth=2))
                plot.add_item(self.fit_curves[-1])

            fitargs, fitkwargs = self.get_fitfunc_arguments()

            yfit = self.fitfunc[i](self.x, *([p.value for p in self.fitparams[i]]+
                        list(fitargs)), **fitkwargs)

            if plot.get_scales()[1] == 'log':
                self.fit_curves[i].set_data(self.x[yfit>1e-80], yfit[yfit>1e-80])
            else:
                self.fit_curves[i].set_data(self.x, yfit)

        self.get_itemlist_panel().show()

        if self.fit_curve is not None:
            self.fit_curve.curveparam.line.color = self.fit_curves[self.idx].curveparam.line.color
            self.fit_curve.curveparam.line.style = 'DashLine'

        self.fit_curve = self.fit_curves[self.idx]
        self.fit_curve.curveparam.line.color = u'#ff0000'
        self.fit_curve.curveparam.line.style = 'SolidLine'

        [item.update_params() for item in self.fit_curves]

        if self.legend is None:
            self.legend = make.legend(anchor=self.legend_anchor)
            plot.add_item(self.legend)

        plot.set_antialiasing(False)

        plot.replot()


        #plot.disable_autoscale()

    def range_changed(self, xrange_obj, xmin, xmax):
        self.autofit_prm.xmin, self.autofit_prm.xmax = xmin, xmax
        self.compute_imin_imax()

    def toggle_xrange(self, state):
        self.xrange.setVisible(state)
        plot = self.get_plot()
        plot.replot()
        if state:
            plot.set_active_item(self.xrange)
        self.show_xrange = state

    def changeFit(self, idx):
        self.idx = idx

        self.refresh()

    def edit_parameters(self):
        if self.autofit_prm.edit(parent=self):
            self.xrange.set_range(self.autofit_prm.xmin, self.autofit_prm.xmax)
            plot = self.get_plot()
            plot.replot()
            self.compute_imin_imax()

    def compute_imin_imax(self):
        self.i_min = self.x.searchsorted(self.autofit_prm.xmin)
        self.i_max = self.x.searchsorted(self.autofit_prm.xmax, side='right')

    def errorfunc(self, params):
        x = self.x[self.i_min:self.i_max]
        y = self.y[self.i_min:self.i_max]
        norm = self.sigma[self.i_min:self.i_max]
        fitargs, fitkwargs = self.get_fitfunc_arguments()
        return (y - self.fitfunc[self.idx](x, *(params.tolist() + list(fitargs)), **fitkwargs))/norm

    def set_autofit_param(self, **kwargs):
        for k,v in kwargs:
            self.autofit_prm.set('_%s'%k,v)

    def autofit(self):
        meth = self.autofit_prm.method
        x0 = np.array([p.value for p in self.fitparams[self.idx]])
        max = np.array([p.max for p in self.fitparams[self.idx]])
        min = np.array([p.min for p in self.fitparams[self.idx]])
        if meth == "lq":
            x = self.autofit_lq(x0,[max,min])
            #~ x = self.autofit_lq(x0)
        elif meth=="simplex":
            x = self.autofit_simplex(x0,[max,min])
            #~ x = self.autofit_simplex(x0)
        elif meth=="powel":
            x = self.autofit_powel(x0,[max,min])
            #~ x = self.autofit_powel(x0)
        elif meth=="bfgs":
            x = self.autofit_bfgs(x0,[max,min])
            #~ x = self.autofit_bfgs(x0)
        elif meth=="l_bfgs_b":
            x = self.autofit_l_bfgs(x0,[max,min])
            #~ x = self.autofit_l_bfgs(x0)
        elif meth=="cg":
            x = self.autofit_cg(x0,[max,min])
            #~ x = self.autofit_cg(x0)
        elif meth=="ncg":
            x = self.autofit_ncg(x0,[max,min])
            #~ x = self.autofit_ncg(x0)
        else:
            return
        for v,p in zip(x, self.fitparams[self.idx]):
            p.value = v
        self.refresh()
        for prm in self.fitparams[self.idx]:
            prm.update(True)

    def get_norm_func(self):
        prm = self.autofit_prm
        err_norm = eval(prm.err_norm)
        def func(params,maxmin):
        #~ def func(params):
            if np.any(np.array(params)>np.array(maxmin[0])):
                return 1e100
            if np.any(np.array(params)<np.array(maxmin[1])):
                return 1e100
            if self.get_plot().get_scales()[1] == 'log':
                err = np.linalg.norm(self.errorfunc(params), err_norm)
            else:
                err = np.linalg.norm(self.errorfunc(params), err_norm)
            return err
        return func

    def get_f_prime(self):
        prm = self.autofit_prm
        err_norm = eval(prm.err_norm)
        from scipy.misc import derivative
        def func(params):
            if self.get_plot().get_scales()[1] == 'log':
                err = derivative(self.errorfunc,np.array(params))
            else:
                err = derivative(self.errorfunc,np.array(params))
            return err
        return func

    def autofit_simplex(self, x0, maxmin):
    #~ def autofit_simplex(self, x0):
        prm = self.autofit_prm
        from scipy.optimize import fmin
        x = fmin(self.get_norm_func(), x0, args=(maxmin,), xtol=prm.xtol, ftol=prm.ftol,
        #~ x = fmin(self.get_norm_func(), x0, xtol=prm.xtol, ftol=prm.ftol,
                maxfun=prm.maxfun, maxiter=prm.maxiter)
        return x

    def autofit_powel(self, x0, maxmin):
        prm = self.autofit_prm
        from scipy.optimize import fmin_powell
        x = fmin_powell(self.get_norm_func(), x0, args=(maxmin,), xtol=prm.xtol, ftol=prm.ftol,
                        maxfun=prm.maxfun, maxiter=prm.maxiter)
        return x

    def autofit_bfgs(self, x0, maxmin):
        prm = self.autofit_prm
        from scipy.optimize import fmin_bfgs
        x = fmin_bfgs(self.get_norm_func(), x0, args=([1e300,1e-300],), gtol=prm.gtol,
                      norm=eval(prm.norm), maxiter=prm.maxiter)
        return x

    def autofit_l_bfgs(self, x0, maxmin):
        prm = self.autofit_prm
        bounds = [(p.min, p.max) for p in self.fitparams[self.idx]]
        from scipy.optimize import fmin_l_bfgs_b
        x, _f, _d = fmin_l_bfgs_b(self.get_norm_func(), x0, args=([1e300,1e-300],), pgtol=prm.gtol,
                          approx_grad=1, bounds=bounds, maxfun=prm.maxfun)
        return x

    def autofit_ncg(self, x0, maxmin):
        '''Only if fprime can be provided'''
        prm = self.autofit_prm
        from scipy.optimize import fmin_ncg
        x = fmin_ncg(self.get_norm_func(), x0, args=(maxmin,), fprime=self.get_f_prime(), epsilon=prm.gtol,
                    maxiter=prm.maxiter)
        return x

    def autofit_cg(self, x0, maxmin):
        prm = self.autofit_prm
        from scipy.optimize import fmin_cg
        x = fmin_cg(self.get_norm_func(), x0, args=(maxmin,), gtol=prm.gtol,
                    norm=eval(prm.norm), maxiter=prm.maxiter)
        return x

    def autofit_lq(self, x0, maxmin):
    #~ def autofit_lq(self, x0):
        prm = self.autofit_prm
        def func(params):
            if np.any(np.array(params)>np.array(maxmin[0])):
                return [1e100]*len(self.x)
            if np.any(np.array(params)<np.array(maxmin[1])):
                return [1e100]*len(self.x)

            return self.errorfunc(params)

        from scipy.optimize import leastsq
        x, _ier = leastsq(func, x0, xtol=prm.xtol, ftol=prm.ftol, maxfev=prm.maxfun)
        return x

    def get_values(self):
        """Convenience method to get fit parameter values"""
        return [[param.value for param in self.fitparams[idx]] for idx in xrange(self._idxlen)]

    def get_fitresult(self):
        fitargs, fitkwargs = self.get_fitfunc_arguments()
        std = [sum(((self.y - self.fitfunc[i](self.x, *(self.get_values()[i] + list(fitargs)), **fitkwargs))/self.y)**2)/len(self.x) for i in xrange(self._idxlen)]
        fitargs, fitkwargs = self.get_fitfunc_arguments()
        yfit = [self.fitfunc[idx](self.x, *([p.value for p in self.fitparams[idx]]+
                                                list(fitargs)), **fitkwargs) for idx in xrange(self._idxlen)]
        return self.fitparams, std, yfit, self.xrange.get_range()


class FitWidget(QWidget, FitWidgetMixin):
    def __init__(self, wintitle=None, icon="guiqwt.png", toolbar=False,
                 options=None, parent=None, panels=None,
                 param_cols=1, legend_anchor='TR', auto_fit=False):
        QWidget.__init__(self, parent)
        FitWidgetMixin.__init__(self, wintitle, icon, toolbar, options, panels,
                                param_cols, legend_anchor, auto_fit)


class FitDialog(QDialog, FitWidgetMixin):
    def __init__(self, wintitle=None, icon="guiqwt.png", edit=True,
                 toolbar=False, options=None, parent=None, panels=None,
                 param_cols=1, legend_anchor='TR', auto_fit=False,
                 show_range=True):
        QDialog.__init__(self, parent)
        self.edit = edit
        self.button_layout = None
        self.idx = 0
        self._idxlen = 2

        FitWidgetMixin.__init__(self, wintitle, icon, toolbar, options, panels,
                                param_cols, legend_anchor, auto_fit,show_range)
        self.setWindowFlags(Qt.Window)

    def setup_widget_layout(self):
        FitWidgetMixin.setup_widget_layout(self)
        if self.edit:
            self.install_button_layout()

    def install_button_layout(self):
        bbox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.connect(bbox, SIGNAL("accepted()"), SLOT("accept()"))
        self.connect(bbox, SIGNAL("rejected()"), SLOT("reject()"))
        self.button_list += [bbox.button(QDialogButtonBox.Ok)]

        self.button_layout = QHBoxLayout()
        self.button_layout.addStretch()
        self.button_layout.addWidget(bbox)

        vlayout = self.layout()
        vlayout.addSpacing(10)
        vlayout.addLayout(self.button_layout)




def guifit(x, y, fitfunc, fitparams, fitargs=None, fitkwargs=None,
           sigma=None, wintitle=None, title=None, xlabel=None, ylabel=None,
           xaxlog=False, yaxlog=False, xmin=None, xmax=None,
           ymin=None, ymax=None, fitxmin=None, fitxmax=None,
           param_cols=1, auto_fit=True, winsize=None, winpos=None,
            **autofit_params):

    """GUI-based curve fitting tool"""
    _app = guidata.qapplication()
    #~ win = FitWidget(wintitle=wintitle, toolbar=True,
                    #~ param_cols=param_cols, auto_fit=auto_fit,
                    #~ options=dict(title=title, xlabel=xlabel, ylabel=ylabel))

    from numpy import min, max
    if(xaxlog==True): typeofxax = "log"
    else: typeofxax="lin"

    if(yaxlog==True): typeofyax = "log"
    else: typeofyax="lin"

    win = FitDialog(edit=True, wintitle=wintitle, toolbar=True,
                    param_cols=param_cols, auto_fit=auto_fit,
                    options=dict(title=title, xlabel=xlabel, ylabel=ylabel),
                    show_range=True)

    if xmin is None:
        if xaxlog:
            xmin = 1e-300 if x.min() < 0.0 else x.min()
        else:
            xmin = x.min()

    if xmax is None:
            xmax = x.max()

    if ymin is None:
        if yaxlog:
            ymin = 1e-300 if (y[x>0]).min() < 0.0 else (y[x>0]).min()
        else:
            ymin = y.min()

    if ymax is None:
            ymax = y.max()

    win.resize(900,600)
    win.get_default_plot().set_scales(typeofxax,typeofyax)
    win.get_default_plot().set_axis_limits(win.get_default_plot().get_active_axes()[0],xmin,xmax)#set x-axis range / find set_axis_limits in in guiqwt curves.py
    win.get_default_plot().set_axis_limits(win.get_default_plot().get_active_axes()[1],ymin,ymax)


    win.set_data(x, y, fitfunc, fitparams, fitargs, fitkwargs, fitxmin, fitxmax, sigma)

    if winsize is not None:
        win.resize(*winsize)
    if winpos is not None:
        win.move(*winpos)
    if win.exec_():

        return win.get_fitresult()


if __name__ == "__main__":
    x = np.linspace(-10, 10, 1000)
    y = np.cos(1.5*x)+np.random.rand(x.shape[0])*.2
    def fit(x, params):
        a, b = params
        return np.cos(b*x)+a
    a = FitParam("Offset", 1., 0., 2.)
    b = FitParam("Frequency", 1.05, 0.1, 10., logscale=True)
    params = [a, b]
    values = guifit(x, y, fit, params, auto_fit=True)
    print values
    print [param.value for param in params]
