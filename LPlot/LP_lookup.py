#!/usr/bin/python
# -*- coding: utf-8 -*-
#  Januar 2009 (JL)
from numpy import dtype, array, append, delete, nan, inf, where, isnan, isinf
from LP_func import isNumeric, isFloat
import tempfile
import tables

import LP_object   ;reload(LP_object)
import LP_func     ;reload(LP_func)

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


class m_Lookup(LP_object.m_Object):
    """
        Class Lookup
        -------------

        Class helps to search and sort fast for components in LPlot lists of
        objects, e.g. dataObjects or fileObjects.
        Class uses temporary (python) tables for that.
    """
    lookups=[]
    def __init__(self,derived_class=LP_object.m_Object):
        derived_class.__init__(self)
        self.tFname = tempfile.mktemp()
        self.fileH = tables.openFile(self.tFname , mode='w')
        m_Lookup.lookups.append(self.fileH)
        self.des=None

    def __del__(self):
        from os import remove
        try:
            self.fileH.flush()
            self.fileH.close()
            m_Lookup.lookups.remove(self.fileH)
            del self.fileH
            remove(self.tFname)
        except:
            pass


    def __getDescription(self,items):
        tmp = {}
        for k,v in items:
            k=k.replace('/','__')
            k=k.replace('__d0','')
            value = v
            try:
                if any(isnan(value)):
                    value = where( isnan(value), 0.0, v )
            except:
                pass
            if inf in v:
                value = where( isinf(value), 1e300, v )
            if -inf in v:
                value = where( isinf(value), -1e300, v )

            if not isNumeric(value):
                value = '*'*64

            if _ndim(value)>1 :
                if _asarray(value).dtype == object:
                    if isinstance(value,tuple):
                        value=list(value)
                    default=None
                    for i in xrange(len(value)):
                        if value[i] is not None:
                            default = value[i]
                            break
                    for i in xrange(len(value)):
                        if value[i] is None:
                            value[i] = default

                if LP_func.isNumeric(_asarray(value)[0]):
                        tmp[k] = tables.Col.from_dtype(dtype((dtype('float64'),
                                                _asarray(value).shape)),dflt=-inf)
                else:
                    tmp[k] = tables.Col.from_dtype(dtype((_asarray(value).dtype,
                                            _asarray(value).shape)))

            elif _asarray(value).dtype <> object:
                if LP_func.isNumeric(_asarray(value)[0]):
                    tmp[k] = tables.Col.from_dtype(dtype(dtype('float64')),dflt=-inf)
                else:
                    tmp[k] = tables.Col.from_dtype(dtype(_asarray(value).dtype))


        return tmp


    def add(self,obj):
        items = obj.getItems()

        if 'lookup' not in self.fileH.root._v_children.keys():
            self.initLookup(obj)

        for k, v in items:
            k=k.replace('/','__')
            k=k.replace('__d0','')
            #~ try:
            if self.des.has_key(k):
                if _ndim(v) > 0:
                    self.dTable.row[k] = v
                else:
                    self.dTable.row[k] = v
            else:
                #~ print '!! WARNING: No key %s found in lookup table!' %(k)
                pass
            #~ except:
                #~ pass

        self.dTable.row.append()
        self.dTable.flush()

    def condition_to_string(self, condition):
        conDict=self.__getKeysandValues(condition)
        retStr=''
        for test,values in conDict.items():
            for value in values:
                sep,value=value
                if not(len(values)>1 and sep<>'=='):
                    retStr += '%s%s%s, ' %(test, sep, value)
        return retStr.rstrip(', ').replace('==','=')


    def keys_to_string(self, condition, keys=None):
        conDict=self.__getKeysandValues(condition)
        retStr=''
        tkeys = conDict.keys()
        if keys is not None:
            [tkeys.append(key) for key in keys if key not in tkeys]
        res = self.getValuesWhere(condition, tkeys)
        for i in xrange(len(tkeys)):
            for r in res[i]:
                retStr += '%s=%s, ' %(tkeys[i], r)
        return retStr.rstrip(', ')

    def __getKeysandValues(self, condition):
        import re
        ret={}
        parts=re.findall('["\.\+\-a-zA-Z0-9_\/\ ]*[!=<>]*["\.\+\-a-zA-Z0-9_\/\ ]*',condition)
        for part in parts:
            if part <> '':
                try:
                    sep=re.search('[!=<>]+',part).group()
                except:
                    sep = '=='
                if sep == '=': sep = '=='
                test, value = part.split(sep)
                if LP_func.isNumeric(test):
                    value, test = test,value

                #~ test = test.replace(' ','')
                test = test.replace(',','.')
                test = test.replace('/','__')
                #~ test = test.replace(' ','')
                #~ value = value.replace(' ','')
                value = value.replace(',','.')
                value = value.replace('/','__')
                #~ value = value.replace(' ','')

                if test not in self.dTable.description._v_names:
                    value, test = test,value

                if test in self.dTable.description._v_names:
                    if ret.has_key(test):
                        if [sep,str(LP_func.atof(value))] not in ret[test]:
                            ret[test].append([sep,value])
                    else:
                        ret[test]=[[sep,value]]

        return ret

    def __filter(self, condition, **kwargs):
        from numpy import log10,floor,sqrt

        condition=condition.replace('/','__')
        conDict=self.__getKeysandValues(condition)

        for test,values in conDict.items():
            for value in values:
                sep,value=value
                part = '%s%s%s' %(test,sep,value)
                if (not LP_func.isNumeric(value)
                    and value not in self.dTable.read()[test]):
                    value=kwargs[value]
                if (LP_func.isNumeric(value) and kwargs.has_key(value)
                    and LP_func.atoi(value) not in self.dTable.read()[test]):
                    value=kwargs[value]

                repl = '%s%s%s' %(test,sep,value)

                if sep == '==':
                    if LP_func.isNumeric(value):
                        if LP_func.atof(value) <> 0.0:
                            dev = 10**(floor(log10(abs(LP_func.atof((value))))))
                        else:
                            dev = 1.
                        repl = '(sqrt((%s-%s)*(%s-%s))/%e < 1e-6)' \
                                %(value,test,value,test,dev)
                if LP_func.isAscii(value):
                    repl = '%s%s"%s"' %(test,sep,value)
                    #~ part = '%s%s%s' %(test,sep,value)

                condition=condition.replace(part,repl)

        return condition


    def initLookup(self, obj):
        if isinstance(obj, dict):
            items = obj.getItems()
        else:
            items = obj.header.getItems()

        self.des = self.__getDescription(items)
        self.dTable = self.fileH.createTable( self.fileH.root,'lookup',self.des)

    def getWhere(self,condition, fnc=lambda x: True, **kwargs):
        condition = self.__filter(condition,**kwargs)
        if condition!='':
            return array([entry['name'] for entry
                        in self.dTable.where(condition)
                        if fnc(entry)])
        else:
            return array([entry['name'] for entry
                        in self.dTable.read()
                        if fnc(entry)])


    def getWhereSorted(self,condition, keys, fnc=lambda x: True,
                        fnc_sort=lambda x,y: cmp(x,y),**kwargs):
        keys=_asarray(keys)
        keys=[key for key in keys if key != 'name']
        keys=append(keys,'name')
        ret=self.readWhereSorted(condition,keys,fnc,fnc_sort,**kwargs)
        return ret[-1]

    def readWhere(self,condition,keys,fnc=lambda x: True,trans=False,**kwargs):
        condition = self.__filter(condition,**kwargs)
        ret = []
        if condition!='':
            for k in _asarray(keys):
                k=k.replace('/','__')
                k=k.replace('/d0','')
                if k in self.dTable.description._v_names:
                    ret.append(array([entry[k] for entry
                                    in self.dTable.where(condition)
                                    if fnc(entry)]))
        else:
            for k in _asarray(keys):
                k=k.replace('/','__')
                k=k.replace('/d0','')
                if k in self.dTable.description._v_names:
                    ret.append(array([entry[k] for entry
                                    in self.dTable.read()
                                    if fnc(entry)]))
        if trans:
            return LP_func.transpose(ret)
        else:
            return ret

    def readWhereSorted(self,condition,keys,fnc=lambda x: True,
                        fnc_sort=lambda x,y: cmp(x,y),trans=False,**kwargs):
        if _asarray(keys).shape[0]>1:
            keys=_asarray(keys)
        else:
            keys=append(_asarray(keys),'counter')

        ret = self.readWhere(condition,keys,fnc,
                        trans=False,**kwargs)

        if trans:
                return LP_func.transpose(self.__sort(ret,fnc_sort))
        else:
                return self.__sort(ret,fnc_sort)


    def __sort(self,xs,fnc=lambda x,y: cmp(x,y)):
        from numpy import append
        from LP_func import isAscii, strSort_cmp
        ret=[]
        tmp_x=[x for x in xs[0]]
        tmp_x_sort=[x for x in xs[0]]
        if isAscii(str(xs[0])):
            tmp_x_sort.sort(strSort_cmp)
        else:
            tmp_x_sort.sort(fnc)

        tmp_y=[y.tolist() for y in xs[1:]]

        for x in tmp_x_sort:
            r=[]
            i = tmp_x.index(x)
            r.append(x)
            for y in tmp_y:
                r.append(y[i])
                y.pop(i)
            ret.append(r)
            tmp_x.pop(i)
        ret = LP_func.transpose(ret)
        return ret

    def read(self,keys,trans=False):
        return self.readWhere('',keys,trans=trans)

    def getValuesWhere(self,condition,keys,fnc=lambda x: True,**kwargs):
        tmp=[]
        vals=self.readWhere(condition,keys,fnc,False,**kwargs)
        for val in _asarray(vals):
            tmp_val=[]
            for v in _asarray(val):
                if v not in tmp_val:
                    tmp_val.append(v)
            tmp.append(tmp_val)
        if tmp == []:
            tmp.append([])
        return tmp

    def readAll(self):
        tmp={}
        items=self.getItems()
        for key,values in self.getItems():
            tmp[key]=[]
            for v in _asarray(values):
                if v not in tmp[key]:
                    tmp[key].append(v)
        return tmp

    def getConditions(self,condition,keys,fnc=lambda x: True,**kwargs):
        if keys is None:
            return [condition]
        keys = [k.replace('/','__') for k in _asarray(keys)]
        tmp=self.getValuesWhere(condition,keys,fnc,**kwargs)
        cons = self.__getConditions(keys,tmp)
        for i in xrange(len(cons)):
            #~ conParts=_asarray(condition.replace(' ','').split('&'))
            conParts=_asarray(condition.split('&'))
            for conp in conParts:
                if conp <> '':
                    cons[i]+='&('+conp+')'
            cons[i].rstrip('&')

        return cons

    def getConditionKeys(self,condition,keys,fnc=lambda x: True,**kwargs):

        keys = [k.replace('/','__') for k in _asarray(keys)]
        tmp=self.getValuesWhere(condition,keys,fnc,**kwargs)
        cons = self.__getConditions(keys,tmp)
        for i in xrange(len(cons)):
            #~ conParts=_asarray(condition.replace(' ','').split('&'))
            conParts=_asarray(condition.split('&'))
            for conp in conParts:
                if conp <> '':
                    cons[i]+='&('+conp+')'
            cons[i].rstrip('&')

        return cons

    def __getConditions(self,keys,tmp):
        max=1
        cons=[]
        ks = []
        for i,k in enumerate(keys):
            if k in self.dTable.description._v_names:
                ks.append(keys[i])
        for k in xrange(len(tmp)):
            max*=len(tmp[k])
        for i in xrange(max):
            ret=''
            for k in xrange(len(ks)):
                l=1
                for t in xrange(k+1,len(ks)):
                    l*=len(tmp[t])
                j=i/l%len(tmp[k])
                if LP_func.isAscii(tmp[k][j]):
                    ret += '(%s==%s)&' %(ks[k],tmp[k][j])
                else:
                    ret += '(%s==%s)&' %(ks[k],tmp[k][j])

            if self.getWhere(ret[:-1]) != [[]]:
                cons.append(ret[:-1])
        return cons

    def getKeys(self):
        return self.dTable.cols._v_colnames

    def getValues(self):
        keys=self.getKeys()
        vals=[]
        for key in keys:
            vals.append(self.dTable.read()[key])
        return vals

    def getItems(self):
        return zip(self.getKeys(),self.getValues())

