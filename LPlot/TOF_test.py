#!/usr/bin/env python
#
#       TOF_test.py.py
#
#       Copyright 2009 Jens Lorrmann <jens@E07-Jens>
#
#       This program is free software; you can redistribute it and/or modify
#       it under the terms of the GNU General Public License as published by
#       the Free Software Foundation; either version 2 of the License, or
#       (at your option) any later version.
#
#       This program is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#       GNU General Public License for more details.
#
#       You should have received a copy of the GNU General Public License
#       along with this program; if not, write to the Free Software
#       Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#       MA 02110-1301, USA.

import os
import numpy
import pylab
import tables
import LP_object
import rainbow

class dataObject(LP_object.m_Object):
	def __init__(self, data, header, name):
		self.data = data
		self.header = header
		self.name = name


class dataHandler(LP_object.m_Object):
    def __init__(self, dir = '.'):
        self.dataObjects = {}
        self.n = 0
        self.des = ''
        self.dTable = None
        if os.path.isdir(dir):
            self.dir = dir

    def get_des(self, file):
        f = open(os.path.join(self.dir,file),'r')
        ls=[]
        tmp = {}
        line = f.readline()
        while line.count('#') > 0:
            if line.count('##dt') > 0:
                line = line.split(';')
                for l in line:
                    l = l.split('=')
                    if len(l) == 2:
                        if l[0][0]=='#':
                            if l[1][0].isdigit() or l[1][0] in ('+','-'):
                                tmp[l[0][2:]] = tables.Float32Col()
                            else:
                                tmp[l[0][2:]] = tables.StringCol(32)
                        else:
                            if l[1][0].isdigit() or l[1][0] in ('+','-'):
                                tmp[l[0]] = tables.Float32Col()
                            else:
                                tmp[l[0]] = tables.StringCol(32)
            line = f.readline()

        tmp['name'] = tables.StringCol(32)
        f.close()

        self.des = tmp

    def load_files(self, dir=None, range=[-1,-1]):
        from scipy.signal import wiener
        from LP_func import m_Progress, smooth, strSort_cmp
        #~ import tempfile
        #~ tFile = tempfile.mktemp
        tFile = '/tmp/test.hdf5'
        if dir is not None:
            self.dir = dir

        if not os.path.isdir(self.dir):
            print '!! ERROR: Directory not found: %s' %(dir)
            return 0

        files = os.listdir(dir)
        files.sort(strSort_cmp)
        if os.path.isfile(tFile):
            os.remove(tFile)
        fileH = tables.openFile(tFile , mode='w')
        self.get_des(files[0])
        self.dTable = fileH.createTable(fileH.root, 'Headers', self.des)
        if range == [-1,-1]:
            range = xrange(0,len(files))
        elif range[1] == -1:
            range = xrange(max(range[0],0), len(files))
        elif range[0] == -1:  
            range = xrange(0, min(max(range[1],0),len(files)))
        else:
            range = xrange(max(range[0],0), min(max(range[1],0),len(files)))
        prog = m_Progress(len(range))
        for file in [files[i] for i in range]:
            name = file.split('.')[0]
            file = os.path.join(self.dir,file)
            append = False
            f = open(file,'r')
            header = {}
            line = f.readline()
            while line.count('#') > 0:
                if line.count('##dt') > 0:
                    append = True
                    line = line.split(';')
                    for l in line:
                        l = l.split('=')
                        if len(l) == 2:
                            if l[0][0]=='#':
                                header[l[0][2:]] = float(l[1])
                            else:
                                header[l[0]] = float(l[1])
                line = f.readline()

            if append:
                data = {}

                for k,v in header.items():
                    self.dTable.row[k] = v
                self.dTable.row['name'] = name
                self.dTable.row.append()

                data['x'] = numpy.linspace( float(header['tstart']), float(header['tend']), float(header['dp']))
                data['y'] = numpy.loadtxt(file)
                data[('x','plot')] = data['x'][numpy.where(data['x']>0)[0][0]:]
                data[('y','plot')] = data['y'][numpy.where(data['x']>0)[0][0]:]
                data[('x','plot1')] = data['x'][numpy.where(data['x']>0)[0][0]:numpy.where(data['x']<7e-3)[0][-1]]
                data[('y','plot1')] = data['y'][numpy.where(data['x']>0)[0][0]:numpy.where(data['x']<7e-3)[0][-1]]
                data[('y','wsmoothed')] = wiener(wiener(wiener(wiener(data[('y','plot1')],300), 150), 15), 2)
                data[('y','bsmoothed1')] = smooth(data[('y','plot1')],20)
                data[('y','bsmoothed2')] = smooth(data[('y','plot1')],200)
                data['xin'] = numpy.array([sum(data['x'][i-10:i+10])/20 for i in xrange(10,data['x'].shape[0]-11,10)])
                data['yin'] = numpy.array([sum(data['y'][i-10:i+10])/20 for i in xrange(10,data['x'].shape[0]-11,10)])

                self.dataObjects[name] = (dataObject(data,header,name))
            prog.progress()
        self.dTable.flush()


    def plot_files(self, x, y, condition = 'All', fnc_x = None, fnc_y = None, kind = 'loglog'):
        xs=[]; ys=[]; names=[]; colors=[]
        if condition == 'All':
            for k in self.dTable.iterrows():
                k = k['name']
                if fnc_x is None:
                    xs.append(self.dataObjects[k].data[x])
                else:
                    xs.append(fnc_x(self.dataObjects[k].data[x]))
                if fnc_y is None:
                    ys.append(self.dataObjects[k].data[y])
                else:
                    ys.append(fnc_y(self.dataObjects[k].data[y]))
                names.append(self.dataObjects[k].name)
        else:
            if condition.count('==') > 0:
                condition = self.get_Filter(condition)
            for k in self.dTable.where(condition):
                k = k['name']
                if fnc_x is None:
                    xs.append(self.dataObjects[k].data[x])
                else:
                    xs.append(fnc_x(self.dataObjects[k].data[x]))
                if fnc_y is None:
                    ys.append(self.dataObjects[k].data[y])
                else:
                    ys.append(fnc_y(self.dataObjects[k].data[y]))
                names.append(self.dataObjects[k].name)
                
        cc = rainbow.rainbow_colors(len(xs))
        [eval('pylab.%s' %kind)(xs[i], ys[i], label=names[i], color=cc.next()) for i in xrange(len(xs))]
        pylab.show()

    def mk_array(self, x, y, condition = 'All'):
        from numpy import asarray
        t1 = {}
        if condition == 'All':
            for k in self.dTable.iterrows():
                k = k['name']
                key = self.dataObjects[k].data[x] if self.dataObjects[k].data.has_key(x) else self.dataObjects[k].header[x]
                value = self.dataObjects[k].data[y] if self.dataObjects[k].data.has_key(y) else self.dataObjects[k].header[y]
                if asarray(value).ndim == 0:
                    if value <> 0:
                        t1[key] = value
                else:
                    if len(value) > 0:
                        t1[key] = value
            keys = numpy.array(sorted(t1.keys()))
            return keys, numpy.array([t1[k] for k in keys])
        else:
            if condition.count('==') > 0:
                condition = self.get_Filter(condition)
            for k in self.dTable.where(condition):
                k = k['name']
                key = self.dataObjects[k].data[x] if self.dataObjects[k].data.has_key(x) else self.dataObjects[k].header[x]
                value = self.dataObjects[k].data[y] if self.dataObjects[k].data.has_key(y) else self.dataObjects[k].header[y]
                if asarray(value).ndim == 0:
                    if value <> 0:
                        t1[key] = value
                else:
                    if len(value) > 0:
                        t1[key] = value
            keys = numpy.array(sorted(t1.keys()))
            return keys, numpy.array([t1[k] for k in keys])


    def add_Data(self):
        hfile = self.dTable._v_file
        hfile.createGroup(hfile.root, 'Data', 'Data from files')
        prog = LPlot.LP_progress(hfile.root.Headers.nrows)
        for row in hfile.root.Headers.iterrows():
            prog.progress()
            gr = hfile.createGroup(hfile.root.Data, row['name'], 'Data from files')
            for k,v in self.dataObjects[row['name']].data.items():
                if isinstance(k, str):
                    array = hfile.createArray(gr, k, v)
                else:
                    name=''
                    for part in k:
                        name += part + '_'
                    name = name[:-2]
                    array = hfile.createArray(gr, name, v)
            tab = hfile.createTable(gr, 'h', self.des)
            for k,v in self.dataObjects[row['name']].header.items():
                tab.row[k] = v
            tab.row.append()
            tab.flush()

    def eval_TOF_batch(self, condition = 'All'):
        if condition == 'All':
            for k in self.dTable.iterrows():
                k = k['name']
                q, t_q, t_max = self.eval_TOF(self.dataObjects[k].data[('x','plot')],self.dataObjects[k].data[('y','smoothed')])
                self.dataObjects[k].data['q'] = q
                self.dataObjects[k].data['t_q'] = t_q
                self.dataObjects[k].data['t_max'] = t_max
        else:
            if condition.count('==') > 0:
                condition = self.get_Filter(condition)
                #~ print '!! WARNING: Testing equality "(\'testValue == Value\')" \n (e.g. \'(T==300)\') doesn`t work reliable !!'
                #~ print '   To test equality use e.g. "(\'(300-T) < 0.00001\')" \n as condition'
            for k in self.dTable.where(condition):
                k = k['name']
                q, t_q, t_max = self.eval_TOF(self.dataObjects[k].data[('x','plot')],self.dataObjects[k].data[('y','smoothed')])
                self.dataObjects[k].data['q'] = q
                self.dataObjects[k].data['t_q'] = t_q
                self.dataObjects[k].data['t_max'] = t_max


    def eval_TOF(self, t, j):
        from scipy.integrate import trapz
        start = numpy.where(t>5e-6)[0][0]
        if numpy.where(j[start:] < 0)[0].shape[0] > 0:
            offsetRange = [start-200+numpy.where(j[start:] < 0)[0][0], start+numpy.where(j[start:] < 0)[0][0]]
        else:
            offsetRange = [j.shape[0]-200, j.shape[0]]

        print offsetRange, numpy.average(j[offsetRange[0]:offsetRange[1]]), t[offsetRange[1]]

        j -= numpy.average(j[offsetRange[0]:offsetRange[1]])
        q = numpy.array([trapz(j[:i],t[:i]) for i in xrange(0,offsetRange[1],10)])
        q /= q[-1]
        t_q = numpy.array([t[i] for i in xrange(0,offsetRange[1],10)])
        t_max = []
        for i in xrange(9):
            t_max.append(t_q[numpy.where(q>(i+1)/10.)[0][0]])

        return q, t_q, t_max


    def get_Filter(self, str):
        parts = str.split('&')
        returnStr = ''
        for part in parts:
            if part.count('==') > 0:
                test, value = part.split('==')
                test = test.replace(' ','')
                value = value.replace(' ','')
                if value[0].isdigit() or value[0] in ('+','-'):
                    if test.count('(') > 0: test = test[1:]
                    if test.count(')') > 0: test = test[:-1]
                    if value.count('(') > 0: value = value[1:]
                    if value.count(')') > 0: value = value[:-1]
                    part = '(sqrt((%s-%s)*(%s-%s)) < 0.0001*%s)' %(value,test,value,test,value)

            returnStr = returnStr + part + ' & '

        return returnStr[:-3]

    def fitit(self, x, y, fitType = 3, method='simplex', statmodel='leastsq'):
        import LP_fitting
        from LP_func import HiraoFit
        import phc

        data = LP_fitting.m_Data(x,y)
        TOFModel = LP_fitting.m_Model(HiraoFit, [{'v_d':[1e-2,9e-5,8e-1,'m/s']},
                                         {'D':[1e-8,1e-11,8e-4,'m^2/s']},
                                         {'n0':[6e12,5e11,5e15,'no']}],[1e-6])

        fit = LP_fitting.m_Fit(data, TOFModel,fitType=fitType, method=method)
        if fitType == 3:
            LP_fitting.sherpa.set_stat(statmodel)

        res = fit.run()
        v,D,n = res.params
        res['t_cal'] = 1e-6 / v * (1 - numpy.sqrt(phc.pi*D/(1e-6*v)) - 3*D/(2*1e-6*v))/\
                (1 - D/(2*1e-6*v))
        res['t_12'] = 1e-6/v
        res.pprint()
        if fitType == 3:
            LP_fitting.sherpa.plot_fit_resid()

        return res, fit






