#!/usr/bin/python
# -*- coding: iso-8859-1 -*-
#  Januar 2009 (JL)


import numpy,copy,types,sys
import LP_object;           reload(LP_object)
import LP_lookup;           reload(LP_lookup)
import LP_saveload;         reload(LP_saveload)
import LP_func;             reload(LP_func)
import LP_progress;         reload(LP_progress)


axes_lookup = {'j' : r'current density $[$A/m$^2$$]$',
                   't' : r'time $[$s$]$',
                   'td' : r'delay time $[$s$]$',
                   'tdmax' : r'delay time + t$_{max}$ $[$s$]$',
                   'delay' : r'time $[$s$]$',
                   'odelay' : r'time $[$s$]$',
                   'mu': r'mobility $[$m$^2$/Vs$]$',
                   'n' : r'carrier density $[$m$^{-3}$$]$',
                   'n0' : r'carrier density $[$m$^{-3}$$]$',
                   'T' : r'temperatur $[$K$]$',
                   'lambda' : r'wavelength $[$nm$]$',
                   'Ft' : r'filter',
                   'A' : r'voltage slope $[$V/s$]$',
                   'Vp' : r'max. voltage$[$V$]$',
                   'Vmax' : r'max. voltage$[$V$]$',
                   'Voff' : r'offset voltage$[$V$]$',
                   'V' : r'voltage$[$V$]$',
                   'Up' : r'max. voltage$[$V$]$',
                   'Umax' : r'max. voltage$[$V$]$',
                   'Uoff' : r'offset voltage$[$V$]$',
                   'U' : r'voltage$[$V$]$',
                   'F' : r'Field$[$V/m$]$',
                   'tp' : r'extraction time$[$s$]$'}

def _conv(obj, dtype=None):
    """
        Convert an object to the preferred form for input to the odr routine.
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
    from numpy import array
    if array(obj).ndim == 0:
        return array([obj])
    else:
        return array(obj)

def _ndim(obj):
    """
    Get dimension of object
    """
    from numpy import array
    try:
        if isinstance(obj,dict):
            return len(obj)
        return array(obj).ndim
    except:
        return len(obj)
def _canarray(obj):
    """
    Check if dimensions sizes are equal
    """
    from numpy import array
    try:
        tst = array(obj).ndim
        return True
    except:
        return False

class m_Data(LP_object.m_Dict):
    """
        Base Data Class
    """
    def __init__(self, derived_class=LP_object.m_Dict,*args, **kwargs):
        derived_class.__init__(self, **kwargs)
        c = 0
        for arg in args:
            print(arg)
            if _asarray(arg).ndim == 1 or _asarray(arg).ndim == 2:
                self.set('def%d'%c, copy.deepcopy(_conv(arg)))
                c+=1
            else:
                print('WARNING: m_Data can only store 1-dimensional or 2-dimensional data')

    def isEmpty(self):
        return self.items() == []

    def size(self):
        return len(self.keys())



class m_Header(LP_object.m_Dict):
    """
        Base Header Class
    """
    def __init__(self, derived_class=LP_object.m_Dict,*args, **kwargs):
        derived_class.__init__(self, **kwargs)
        c = 0
        for arg in args:
            self.set('def%d'%c, arg)

    def __str__(self):
        ret ='##'
        for key,val in self.items():
            ret += '%s=%s;' %(key,val)
        return ret

    def isEmpty(self):
        return self.items() == []

    def size(self):
        return len(self.keys())


class m_dataObject(LP_object.m_Object):
    """
        Base Data Class with header Informations
    """

    def __init__(self, data=None, header=None, derived_class=LP_object.m_Object, **kwargs):
        derived_class.__init__(self, **kwargs)
        if data is None:
            self.data = m_Data()
        else:
            self.data=data
        if header is None:
            self.header = m_Header()
        else:
            self.header = header
        self.parent =None

        for arg,val in kwargs.items():
            self.header.set(arg,val)

    def clear(self):
        self.data.clear()
        self.header.clear()

    def setParent(self, parent):
        self.parent=parent


    def getData(self, name):
        if name is not None:
            try:
                return self.data.get(name)
            except:
                raise AttributeError, str(name)
        else:
            return self.data

    def setData(self, name, value):
        try:
            self.data.set(name, value)
        except:
            raise AttributeError, str(name)

    def getHeader(self, name = None):
        if name is not None:
            try:
                return self.header.get(name)
            except:
                raise AttributeError, str(name)
        else:
            return self.header

    def setHeader(self, key, value):
        from numpy import nan
        from LP_func import isNumeric
        #~ try:
        if self.parent is not None:
            if self.parent.header.has_key(key):
                if isinstance(self.parent.getHeader(key),dict):
                    self.parent.setHeader(key+'/d0', [nan])

                dat = self.parent.getHeader(key)
                if len(self.parent.getHeader(key)) <= self.getHeader('counter'):

                    dat.extend([nan] * (self.header['counter'] -len(dat) + 1))


                dat[self.getHeader('counter')] = value
                self.parent.setHeader(key, dat)
            else:
                self.parent.setHeader(key, [value])

        self.header.set(key, value)
        #~ except:
            #~ raise AttributeError, str(name)

    def headersize(self):
        return self.header.size()

    def datasize(self):
        return self.data.size()

    def save(self, dir, fname):
        from os import path
        size = 0
        for val in self.data.getValues():
            if size < _asarray(val).shape[0]:
                size = _asarray(val).shape[0]

        f = open(path.join(dir,fname), 'w')
        try:
            f.write('##remark=*~*LPlot dataset - © Jens Lorrmann 2009*~* %s' %(self.header['remark']))
            f.write('##date=%s\n##' %(self.header['date']))
        except:
            f.write('##remark=*~*LPlot dataset - © Jens Lorrmann 2009*~*\n')
            f.write('##date=%s\n##' %('---'))
        for key,value in self.header.getItems():
            if key <> 'remark' and key <> 'date':
                key = key.replace('/','__')
                if LP_func.isNumeric(value):
                    f.write('%s=%.5g;' %(str(key),value))
                else:
                    f.write('%s=%s;' %(str(key),str(value)))
        f.flush()
        f.write('\n##Columns=')
        for key in self.data.getKeys():
            key = key.replace('/d0','')
            f.write('%s ' %(key))
        f.write('\n')
        f.flush()
        for i in xrange(size):
            for key in self.data.getKeys():
                if i < _asarray(self.data[key]).shape[0]:
                    f.write('%.5g\t' %(self.data[key][i]))
                else:
                    f.write('%.5g\t' %(numpy.nan))
            f.write('\n')
        f.flush()
        f.close()


class m_Data2D(m_dataObject):
    """
        Data with Header Class for 1 dimensional data

        Parameter:
        ----------
        data:       instance of m_Data
        x:          vector with x data
        y:          vector with y data
        z:          vector with z data

    """
    def __init__(self, data=None, x=None, y=None, z=None, **kwargs):
        if data is None:
            if x is None or y is None or z is None:
                print 'WARNING: Initialisation failed! Either data or x AND y have to have values'
            else:
                if _asarray(z).ndim < 2:
                    print 'WARNING: Initialisation failed! z data needs tow be 2-dimensional array'
                else:
                    from numpy import meshgrid
                    if _asarray(x).ndim == 1  and _asarray(y).ndim == 1:
                        x,y = meshgrid(x,y)
                    m_dataObject.__init__(self, data=m_Data(x=x,y=y,z=z), **kwargs)
        else:
            m_dataObject.__init__(self, data = data, **kwargs)


class m_Data1D(m_dataObject):
    """
        Data with Header Class for 1 dimensional data

        Parameter:
        ----------
        data:       instance of m_Data
        x:          vector with x data
        y:          vector with y data

    """

    def __init__(self, data=None, x=None, y=None, **kwargs):
        if data is None:
            if x is None or y is None:
                print 'WARNING: Initialisation failed! Either data or x AND y have to have values'
            else:
                m_dataObject.__init__(self, data=m_Data(x=x,y=y), **kwargs)
        else:
            m_dataObject.__init__(self, data = data, **kwargs)


class m_dataHandler(LP_object.m_Object,LP_saveload.m_IgorSave):
    def __init__(self, objects=None, derived_class1=LP_object.m_Object,derived_class2=LP_saveload.m_IgorSave,**kwargs):
        derived_class1.__init__(self, **kwargs)
        derived_class2.__init__(self)
        self.objects = None
        self.objects = LP_object.m_Dict()
        self.header = m_Header(**kwargs)
        self.lookup = LP_lookup.m_Lookup()
        self.order = []
        if objects is not None:
            for object in objects:
                object.parent=None
            self.add(copy.deepcopy(objects))

    def _makeHeader(self,dataObjects):
        # if header is empty the global header is
        # initialiesed with the values from the first dataObject
        from numpy import inf
        for dO in dataObjects:
            if self.header.getKeys() <> dO.header.getKeys():
                for key in dO.header.getKeys():
                    if key not in self.header.getKeys():
                        self.header[key] = [inf]*len(dataObjects)

    def _appendToHeader(self, dataObjects):
        from numpy import nan
        for dO in dataObjects:
            for key in self.header.getKeys():
                if key not in dO.header.getKeys():
                    try:
                        dO.header[key] = self.header[key][dO.header['counter']]
                    except:
                        dO.header[key] = None

                if len(self.header[key]) <= dO.header['counter']:
                    self.header[key].extend([nan] \
                                           *(dO.header['counter'] \
                                           -len(self.header[key]) \
                                           + 1))

                self.header[key][dO.header['counter']] = dO.header[key]

#                if _ndim(dOHeader[key])==0:
#                    if _ndim(sHeader[key])==0:
#                        if dOHeader[key] <> sHeader[key]:
#                            sHeader[key] = _asarray(sHeader[key])
#                            sHeader[key]=append(sHeader[key], copy.copy(dOHeader[key]))
#                    else:
#                        if dOHeader[key] not in sHeader[key]:
#                            try:
#                                sHeader[key]=append(sHeader[key], copy.copy(dOHeader[key]))
#                            except:
#                                if _ndim(sHeader[key])==0:
#                                    sHeader[key] = _asarray(sHeader[key])
#                                sHeader[key]=append(sHeader[key],copy.copy(dOHeader[key]))
#
#                else:
#                    if isinstance(dOHeader[key],dict):
#                        if isinstance(_asarray(sHeader[key])[0],dict):
#                            self.__makeHeader(sHeader,dOHeader[key])
#                        else:
#                            sHeader[key+'/0']=copy.deepcopy(sHeader[key])
#                            self.__makeHeader(sHeader,dOHeader[key])
#
#                    elif _ndim(sHeader[key])==0:
#                        if not all([a==b for a,b in zip(_asarray(dOHeader[key]),_asarray(sHeader[key]))]):
#                            sHeader[key] = _asarray(sHeader[key])
#                            for val in _asarray(dOHeader[key]):
#                                if val not in sHeader[key]:
#                                    sHeader[key]=append(sHeader[key], copy.copy(val))

    def __checkData(self,dO):
        # if header is empty check is not necessary
        if self.header.isEmpty():
            return True
        # check for every header key as if it has
        # a counterpart in the global (handler) header
        for key,val in dO.header.items():
            if self.header.has_key(key):
                if val == self.header[key]:
                    return True
        return False

    def delete(self, names):
        for name in reversed(_asarray(names)):
            row=self.lookup.dTable.getWhereList('name=="%s"'%name)[0]
            self.__dict__.pop('o%d' %self.objects[name].header.counter)
            self.lookup.dTable.removeRows(row)
            self.objects.pop(name)
            self.order.remove(name)

        self.updateHeader()

    def deleteWhere(self, con, fnc = lambda x:x, **kwargs):
        names = self.lookup.readWhere(con,'name',fnc=fnc,**kwargs)
        if names <> []:
            self.delete(names[0])

    def add(self, dOs, **kwargs):
        from string import join
        for key,val in kwargs.items():
            self.header.set(key,val)
        if LP_object.isinstanceof(_asarray(dOs)[0],m_dataObject):
            for d in _asarray(dOs):
                counter = len(self.objects.keys())
                d.setHeader('counter',counter)
                if not d.header.has_key('name'):
                    d.setHeader('name','def_%d' %(counter))
                # Add an extra counter to the names to avoid
                # compatibility problems with IGOR makros

                if not self.header.has_key('name'):
                    self.setHeader('name', [])
                c=0
                name = copy.copy(d.header['name'])
                while True:
                    if name not in self.header['name']:

                        self.header['name'].append(name)
                        break

                    name = d.header['name']+'_'+str(c)
                    c+=1

                d.setParent(self)
                d.setHeader('name', name)
                self.objects.set(d.header['name'],d)
                self.order.append(d.header['name'])
                self.__dict__['o%d' %counter]=self.objects[d.header['name']]

            self._makeHeader(_asarray(dOs))
            self._appendToHeader(_asarray(dOs))
            self.updateHeader()
        else:
            print '%s kann nicht durch m_dataHandler verwaltet werden.'
            print 'Objekte muessen von der Klasse "LP_data.m_dataObject" sein.' %str(_asarray(dOs)[0].__class__).split("'")[1]

    def size(self):
        return len(self.objects)

    def printObjects(self, con, keys='name', fnc=lambda x: True,**kwargs):
        from numpy import array,append
        i = 0
        s = []
        keys=array([key for key in _asarray(keys) if key != 'name'])
        keys=append(array(['name']), keys)
        vals=self.lookup.readWhere(con,keys,fnc,trans=True,**kwargs)
        for name in self.order:
            s.append('%d:\t' %i)
            val=[val for val in vals if val[0]==name]
            for key,v in zip(keys,val[0]):
                if type(val) is types.FloatType:
                    s.append('%s = %.2g,\t' %(key,v))
                elif type(val) is types.IntType:
                    s.append('%s = %d,\t' %(key,v))
                else:
                    s.append('%s = %s,\t' %(key,v))
            i+=1
            s.append('\n')

        sys.stdout.write(''.join(s))
        sys.stdout.flush()

    def getObjects(self, con, keys=None, fnc=lambda x: True,**kwargs):
        if keys is None:
            oH = m_dataHandler()
            Os=[]

            vals=self.lookup.getWhere(con,fnc,**kwargs)
            for val in vals:
                self.objects[val].parent=None
                Os.append(copy.deepcopy(self.objects[val]))
            if len(Os) > 0:
                oH.add(Os)
                return oH
        else:
            oHs={}
            for key in reversed(_asarray(keys)):
                if LP_func.isAscii(_asarray(self.header.get(key))[0]):
                    self.sortObjects(key,LP_func.strSort_cmp,**kwargs)
                else:
                    self.sortObjects(key,cmp,**kwargs)
            cons=self.lookup.getConditions(con,keys,fnc,**kwargs)
            for i,con in enumerate(cons):
                dat=LP_object.m_Dict()
                conParts=_asarray(con.replace(' ','').split('&'))
                for conp in conParts:
                    if conp.count('==') > 0:
                        if not conp.split('==')[0].lstrip('(') in keys:
                            con+='&('+conp+')'
                #~ con += '&'+con
                objs=self.getObjects(con,None,fnc,**kwargs)
                if objs is not None:
                    objs=copy.copy(objs)
                    name = (con.replace('(','').replace(')','')\
                            .replace('&','_').replace('==','').replace(' ',''))
                    oHs[name]=objs

            return oHs



    def sortObjects(self, keys='name', fnc_sort=lambda x,y: cmp(x,y),**kwargs):
        oH = m_dataHandler()
        Os=[]
        vals=self.lookup.getWhereSorted('',keys,fnc_sort=fnc_sort,**kwargs)
        for val in vals:
            Os.append(self.objects[val])
        oH.add(Os)
        objs=oH.getSorted()
        self.__init__(objs)

    def map(self,fnc, con='', prog=True,fnc_cmp=lambda x:x, **kwargs):
        vals=self.lookup.getWhere(con,fnc_cmp,**kwargs)
        if prog:
            progBar = LP_progress.m_progressBar(len(vals),'% Mapping function to data ...')
        for val in vals:
            fnc(self.objects[val],**kwargs)
            #~ progBar.update(name=self.objects[val].header.name)
            if prog:
                progBar.update()


    def getDataSlice(self, con, keys, params=None, sortby=None, name = 'def', fnc=lambda x: True,
                    fnc_sort=lambda x,y: cmp(x,y),**kwargs):
        dH=m_dataHandler()
        dOs=[]
        if params is not None:
            cons=self.lookup.getConditions(con,_asarray(params),fnc,**kwargs)
        else:
            cons = con

        for i,con in enumerate(cons):
            dat=LP_object.m_Dict()

            objs=self.getObjects(con,None,fnc,**kwargs)
            if objs is not None:
                objs=copy.copy(objs)
                if sortby is not None:
                    for key in reversed(_asarray(sortby)):
                        objs.sortObjects(key,fnc_sort,**kwargs)

                #name = ['%svs' %(d.replace('/','_')) for d in keys]
                #nadd = self.lookup.condition_to_string(con).replace('=','').replace(', ','_')
                #for key,value in kwargs.items():
                #    nadd=nadd.replace(key,value)
                objs_name = name
                if params is not None:
                    for param in _asarray(params):
                        objs_name = objs_name + '_%s%s' %(param, str(objs.getSorted()[0].getHeader(param)))

                if objs.header.has_key(_asarray(keys)[0]):
                    dat=LP_object.m_Dict()
                    for d in _asarray(keys):
                        for obj in objs.getSorted():
                            if LP_object.isinstanceof(obj.header[d],LP_object.m_Dict):
                                for k in obj.header[d].getKeys():
                                    if not dat.has_key(d+'/'+k):
                                        dat[d+'/'+k]=[]
                                    dat[d+'/'+k].append(obj.header[d].get(k))
                            else:
                                if not dat.has_key(d):
                                    dat[d]=[]
                                dat[d].append(obj.header.get(d))
                    dO = m_dataObject(m_Data(**dat),m_Header(**{'name':objs_name}))
                    for p in params:
                        dO.setHeader(p, objs.header[p][0])
                    dOs.append(dO)
        dH.add(dOs)
        return dH

    def updateHeader(self):
        del self.lookup
        self.lookup = None
        self.lookup = LP_lookup.m_Lookup()
        self.lookup.initLookup(self.header)
        for d in self.getSorted():
            self.lookup.add(d.header)

    def clear(self):
        for i in xrange(len(self.objects)-1,-1,-1):
            self.objects.pop(i)

    def getSorted(self, con='', i='all', fnc=lambda x: True,**kwargs):
        if con == '':
            if i == 'all':
                return [self.objects[n] for n in self.order]
            else:
                return self.objects[self.order[i]]
        else:
            vals = self.lookup.getWhere(con,fnc,**kwargs)
            Objs = []
            for n in self.order:
                if n in vals:
                    Objs.append(self.objects[n])

            return Objs


    def getHeader(self, name):
        try:
            return self.header.get(name)
        except:
            raise AttributeError, str(name)

    def setHeader(self, name, value):
        try:
            self.header.set(name,value)
        except:
            raise AttributeError, str(name)

    def getData(self, name):
        try:
            data = []
            for dO in self.getSorted():
                data.append(dO.data.get(name))
            return data
        except:
            raise AttributeError, str(name)

    def setData(self, name, value):
        if _ndim(value) < 2:
            try:
                for dO in self.getSorted:
                    dO.data.set(name,value)
            except:
                raise AttributeError, str(name)
        else:
            try:
                i=0
                for dO in self.objects:
                    dO.data.set(name,value[i])
                    i+=1
            except:
                raise AttributeError, str(name)


    def load(self, dir, name=None, con='', node=None, **kwargs):
        from os import path
        objs=self.loadit(path.join(dir,name),condition=con, node=node, **kwargs)
        self.add(objs)

    def save(self, dir, name=None, hdf=False, strip='',node=None, **kwargs):
        from os import path
        size = 0
        for dO in self.objects.values():
            for val in dO.data.values():
                if size < _asarray(val).shape[0]:
                    size = _asarray(val).shape[0]

        if hdf:
            return self.saveit(path.join(dir,name),strip=strip,node=node, **kwargs)

        else:
            if not path.isdir(path.abspath(dir)):
                import os
                os.makedirs(path.abspath(dir))

            counter = 1
            for dO in self.getSorted():
                if name is not None:
                    name1 = '%s%d%s' %(path.splitext(name)[0],
                                       counter,
                                       path.splitext(name)[1])
                else:
                    name1 = '%s.dat' %(dO.header['name'].split('_')[0])
                dO.save(dir,name1)
                counter += 1

    def plot(self, xa='t', ya='j', param=None,  label=None, kind ='plot',
             style=None,  fnc_x = lambda x: x, fnc_y = lambda y: y,
             fnc_z = lambda z: z, axis_label = None, con='',
             fnc_filt = lambda x: True, appto = None, **kwargs):
        """
        Plot Data:

        Plot the data the projectmanger holds. For data that is spread over several
        measurments (e.g. Temp vs. n) use "sort" first.

        Parameter:
        xa:             data to be used for the X axis
        ya:             data to be used for the Y axis
        param:          Plot is parametric for
        kind:           kind of plot ('plot', 'semilogx', ...)
        axis_label:     label of the axes ([0]: X, [1]: Y)
        style:          defines the style of the plot lines
                        ( [linsetyle,  marker, linewidth, markersize];
                        e.g. ['--', 'o', 2, 9] ):

                        "linestyle" defines the style of the plotted lines
                        [ - | -- | -. | : ]

                        "linewidth" defines the width of the plotted line

                        "marker" defines the style of the plotted markers
                        [ + | * | , | . | 1 | 2 | 3 | 4
                        | < | > | D | H | ^ | _ | d
                        | h | o | p | s | v | x | |
                        | TICKUP | TICKDOWN | TICKLEFT | TICKRIGHT ]

                        "markersize" defines the size of the plotted marker

        fnc_x:          function for X-values
        fnc_y:          function for Y-values
        con:      filter data by con; e.g. '(T==300)&(Vp<4)'
        fnc_filt:       function to filter data;
                        the function works on all header data
                        e.g. to filter by name:

                        def filter_by_name(value):
                            num = re.search('[0-9]*', value).group()
                            return int(value) > 20 and int(value) < 40

        appto:       figure to append data to; e.g. pylab.gca()
        """

        import pylab
        import rainbow
        from numpy import array
        axs=[]
        cons = [con]
        style = list(style) if style is not None else []

        # try to guess what to set on the axes if not explicit given

        # set linestyle, linewidth, marker and markersize


        if axis_label is None:
            axis_label = [r'default x', r'default y']
            x_test = xa if not xa.count('/') > 0 else xa[:xa.find('/')]
            y_test = ya if not ya.count('/') > 0 else ya[:ya.find('/')]
            if axes_lookup.has_key(x_test):
                axis_label[0] = axes_lookup[x_test]
            if axes_lookup.has_key(y_test):
                axis_label[1] = axes_lookup[y_test]

        if param is not None:
            cons=self.lookup.getConditions(con,_asarray(param),fnc_filt,**kwargs)
        if cons == []:
            cons = ['',]
        if kind not in ('plot', 'semilogx','semilogy','loglog'):
            self.__Plot2D(xa, ya, _asarray(param)[0], con, _asarray(param)[1:],
                          label, kind, fnc_x, fnc_y, fnc_z, axis_label,
                          fnc_filt, appto, axs, **kwargs)
        else:

            if self.header.has_key(xa) and self.header.has_key(ya):
                sl = len(style)
                default_style = ['--','o',1.5, 9]
                style.extend(default_style[sl:])
                if not appto:
                    f=pylab.figure()
                    axs.append(f.add_subplot(111))
                else:
                    if type(appto) in [list,tuple]:
                        axs=appto
                    else:
                        axs.append(appto)

                ax_num=-1
                old_num_ls = len(axs[ax_num].get_lines())
                for i,con in enumerate(cons):
                    x = (self.lookup.readWhere(con,xa))
                    y = (self.lookup.readWhere(con,ya))

                    if len(x[0]) > 0 and len(y[0]) > 0:
                        if any(array(x[0])>0) or any(array(y[0])>0):

                            if label is not None:
                                try:
                                    lab='%s=%g'%(label,self.lookup.readWhere(con,label)[0][0])
                                except:
                                    lab='%s=%g'%(label,0.0)
                            else:
                                lab = self.lookup.keys_to_string(con)

                            args = [fnc_x( x[0], *(self.__fnc_params(self, fnc_x, con)) ),
                                    fnc_y( y[0], *(self.__fnc_params(self, fnc_y, con)) )]

                            print lab
                            kwargs = dict(label = lab, linestyle = style[0], linewidth = style[2],
                                          marker = style[1], markersize = style[3],
                                          markeredgecolor = None)

                            if kind == 'semilogy':
                                axs[ax_num].semilogy( *args, **kwargs )

                            elif kind == 'loglog':
                                axs[ax_num].loglog( *args, **kwargs )

                            elif kind == 'semilogx':
                                axs[ax_num].semilogx( *args, **kwargs )

                            elif kind == 'plot':
                                axs[ax_num].plot( *args, **kwargs )

                            elif kind == 'plot_date':
                                axs[ax_num].plot_date( *args, **kwargs )
                                axs[ax_num].get_figure().autofmt_xdate()

                    if axis_label is not None:
                        axs[ax_num].set_xlabel(axis_label[0])
                        axs[ax_num].set_ylabel(axis_label[1])

                print default_style, style
                if sl == 0:
                    rainbow.map_markers(axs[ax_num], old_num_ls, len(axs[ax_num].get_lines()))
                else:
                    rainbow.map(axs[ax_num], old_num_ls, len(axs[ax_num].get_lines()), mec='None')

                ncol = len(axs[ax_num].get_lines())/5+1
                pylab.legend(loc=0, ncol=ncol)
                pylab.draw()

            else:
                default_style = ['-','',2, 1]
                style.extend(default_style[len(style):])

                for i,con in enumerate(cons):
                    vals=self.lookup.getWhere(con)
                    if vals <> []:
                        if not appto:
                            f=pylab.figure()
                            axs.append(f.add_subplot(111))
                        else:
                            if type(appto) in [list,tuple]:
                                axs=appto
                            else:
                                axs.append(appto)

                        ax_num=i%len(axs)
                        old_num_ls = len(axs[ax_num].get_lines())


                        self.map(self.__doplot,con,xa=xa,ya=ya,style=style,kind=kind,
                                 ax=axs[ax_num],fnc_x=fnc_x, fnc_y=fnc_y,label=label)

                        if axis_label is not None:
                            axs[ax_num].set_xlabel(axis_label[0])
                            axs[ax_num].set_ylabel(axis_label[1])

                        rainbow.map(axs[ax_num], old_num_ls, len(axs[ax_num].get_lines()))
                        lab = self.lookup.keys_to_string(con)
                        axs[ax_num].set_title(lab)
                        ncol = len(axs[ax_num].get_lines())/5+1
                        axs[ax_num].legend(loc=0, ncol=ncol)
                        if len(axs[ax_num].get_lines())==0:
                            pylab.close(axs[ax_num].get_figure())
                        pylab.draw()

        return axs

    def __fnc_params(self, dO, fnc, con=''):
        args=[]
        fnc_params = fnc.func_code.co_varnames
        if len(fnc_params)>1:

            for param in fnc_params[1:]:
                param = param.replace('__','/')
                if dO.header.has_key(param):
                    args.append(dO.lookup.readWhere(con,param)[0])

                elif dO.data.has_key(param):
                    args.append(dO.getData(param))
                else:
                    import inspect
                    s_code = inspect.getsource(fnc).replace(' ','')
                    if s_code[s_code.rfind(param)-1:s_code.rfind(param)+1].find('*','/') > 0:
                        args.append(1.0)
                    else:
                        args.append(0.0)

                    print ' !WARNING: Parameter %s not found. Using %d instead.' %(param, args[-1])

        return args


    def __doplot(self, dO,xa,ya,style,kind,ax,fnc_x,fnc_y,label):
        from pylab import isscalar
        if dO.data.has_key(xa) and dO.data.has_key(ya):
            x = (dO.getData(xa))
            y = (dO.getData(ya))
            try:
                if isscalar(label):
                    lab='%s=%g'%(label,dO.getHeader(label))
                else:
                    lab=''
                    for l in label:
                        lab+='%s=%g, '%(l,dO.getHeader(l))

                    lab=lab[:-2]
            except:
                lab=None

            if x is not None and y is not None:
                args = [fnc_x( x, *(self.__fnc_params(dO, fnc_x)) ),
                        fnc_y( y, *(self.__fnc_params(dO, fnc_y)) )]

                kwargs = dict(label = lab, linestyle = style[0], linewidth = style[2],
                              marker = style[1], markersize = style[3],
                              markeredgecolor = None)

                if kind == 'semilogy':
                    ax.semilogy( *args, **kwargs )

                elif kind == 'loglog':
                    ax.loglog( *args, **kwargs )

                elif kind == 'semilogx':
                    ax.semilogx( *args, **kwargs )

                elif kind == 'plot':
                    ax.plot( *args, **kwargs )


    def __Plot2D(self, xa, ya, param, con, params, label, kind, fnc_x, fnc_y, fnc_z,
                 axis_label, fnc_filt, appto, axs,**kwargs):

        import pylab
        from matplotlib.image import NonUniformImage
        from numpy import linspace
        from scipy.interpolate import splrep, splev
        from mpl_toolkits.mplot3d import Axes3D

        if len(params)>0:
            cons=self.lookup.getConditions(con,_asarray(params),fnc_filt,**kwargs)
        else:
            cons = _asarray(con)

        for i,con in enumerate(cons):
            vals=self.lookup.getWhere(con)
            if vals <> []:
                if self.objects[vals[0]].data.has_key(xa) and \
                   self.objects[vals[0]].data.has_key(ya):

                    if not appto:
                        f=pylab.figure()
                        if kind in ['imshow', 'contour', 'contourf']:
                            axs.append(f.add_subplot(111))
                        else:
                            axs.append(f.gca(projection='3d'))
                    else:
                        if type(appto) in [list,tuple]:
                            axs=appto
                        else:
                            axs.append(appto)

                    ax_num=i%len(axs)

                    if self.objects[vals[0]].data.has_key(param):

                        ma = fnc_z(self.objects[vals[0]].getData(param))
                        x_axis = fnc_x(self.objects[vals[0]].getData(xa))
                        y_axis = fnc_y(self.objects[vals[0]].getData(ya))


                    else:
                        def generateMatrix(dO,xa, ya, param, kind, ma, xs, ys):
                            # generates a matrix for the Image or contoure plot
                            if dO.data.has_key(xa) and dO.data.has_key(ya) and \
                               self.header.has_key(param):

                                ma.append(fnc_z(dO.getData(ya)))
                                xs.append(fnc_x(dO.getData(xa)))
                                ys.append(fnc_y(dO.getHeader(param)))


                        ma=[];xs=[];ys=[]
                        self.map(generateMatrix,con,xa=xa,ya=ya,param=param,kind=kind,ma=ma,xs=xs,ys=ys)

                        x_min, x_max = max(LP_func.transpose(xs)[0]), min(LP_func.transpose(xs)[-1])
                        x_axis = linspace(x_min, x_max,400)
                        for i in xrange(len(xs)):
                            ma[i] = splev(x_axis, splrep(xs[i], ma[i], k=3))
                            xs[i]=x_axis

                        y_axis = ys

                    if kind == 'imshow':
                        im = NonUniformImage(axs[ax_num], interpolation='bilinear' )
                        im.set_cmap(kwargs.get('cmap',None))
                        im.set_data( x_axis, y_axis, ma )
                        if kwargs.has_key('vmin'):
                            im.set_clim(vmin=kwargs['vmin'])
                        if kwargs.has_key('vmax'):
                            im.set_clim(vmax=kwargs['vmax'])
                        axs[ax_num].images.append( im )
                        #~ xlabel( r'Wavelength [nm]' )
                        #~ ylabel( r'Delay [ps]' )
                        pylab.show()
                        if kwargs.has_key('bar'):
                            bar = kwargs['bar']
                            kwargs.pop('bar')
                        else:
                            bar = True
                        if bar:
                            axs[ax_num].get_figure().colorbar(im)
                        axs[ax_num].set_xlim(x_axis[0],x_axis[-1])
                        axs[ax_num].set_ylim(y_axis[0],y_axis[-1])
                        pylab.draw()
                    elif kind == 'contour':
                        N=kwargs.get('N', 8)
                        X, Y = pylab.meshgrid(x_axis, y_axis)
                        CS=axs[ax_num].contour(X, Y, ma, N, **kwargs)
                        if kwargs.has_key('labels'):
                            labels = kwargs['labels']
                            kwargs.pop('labels')
                            fmt = {}
                            for l, s in zip( CS.levels, labels ):
                                fmt[l] = s
                        elif kwargs.has_key('fmt'):
                            fmt = kwargs('fmt')
                        else:
                            fmt = '%1.2f'
                        if kwargs.has_key('fontsize'):
                            fontsize = kwargs['fontsize']
                        else:
                            fontsize = 12
                        axs[ax_num].clabel(CS, CS.levels, inline=1, fmt = fmt, fontsize = fontsize)
                        pylab.show()
                        axs[ax_num].set_xlim(x_axis[0],x_axis[-1])
                        axs[ax_num].set_ylim(y_axis[0],y_axis[-1])
                        pylab.draw()
                    elif kind == 'contourf':
                        N=kwargs.get('N', 8)
                        X, Y = pylab.meshgrid(x_axis, y_axis)
                        CS=axs[ax_num].contourf(X, Y, ma, N, **kwargs)
                        axs[ax_num].get_figure().colorbar(CS)
                        pylab.show()
                        axs[ax_num].set_xlim(x_axis[0],x_axis[-1])
                        axs[ax_num].set_ylim(y_axis[0],y_axis[-1])
                        pylab.draw()
                    elif kind == 'surf':
                        X, Y = pylab.meshgrid(x_axis, y_axis)
                        CS=axs[ax_num].plot_surface(X, Y, pylab.array(ma), **kwargs)
                        #axs[ax_num].get_figure().colorbar(CS, shrink=0.5, aspect=5)
                        pylab.show()
                        #axs[ax_num].set_xlim(x_axis[0],x_axis[-1])
                        #axs[ax_num].set_ylim(y_axis[0],y_axis[-1])
                        pylab.draw()
                    elif kind == 'contour3d':
                        N=kwargs.get('N', 8)
                        X, Y = pylab.meshgrid(x_axis, y_axis)
                        CS=axs[ax_num].contourf(X, Y, ma, N, **kwargs)
                        if kwargs.has_key('labels'):
                            labels = kwargs['labels']
                            kwargs.pop('labels')
                            fmt = {}
                            for l, s in zip( CS.levels, labels ):
                                fmt[l] = s
                        elif kwargs.has_key('fmt'):
                            fmt = kwargs('fmt')
                        else:
                            fmt = '%1.2f'
                        if kwargs.has_key('fontsize'):
                            fontsize = kwargs['fontsize']
                        else:
                            fontsize = 12
                        axs[ax_num].clabel(CS, CS.levels, inline=1, fmt = fmt, fontsize = fontsize)
                        pylab.show()
                        axs[ax_num].set_xlim(x_axis[0],x_axis[-1])
                        axs[ax_num].set_ylim(y_axis[0],y_axis[-1])
                        pylab.draw()


                    lab = self.lookup.keys_to_string(con)
                    axs[ax_num].set_title(lab)
                    pylab.show()
                    pylab.draw()

