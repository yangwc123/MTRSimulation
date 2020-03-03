#!/usr/bin/env python
from numpy import dtype, array, asarray
import types
from copy import deepcopy


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

def _asarray(obj):
    if array(obj).ndim == 0:
        return array([obj])
    else:
        return array(obj)

def _ndim(obj):
    """
    Get dimension of object
    """
    try:
        if isinstance(obj,dict):
            return len(obj)
        return array(obj).ndim
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

class m_Object(object):
    """
        Base Object Class
    """

    def __init__(self, **kwargs):
        self.__dict__ = m_Dict(self.__dict__)
        self.__dict__.__init__(self,**kwargs)

    def __setitem__(self, key, value):
        self.__dict__.__setitem__(key, value)

    def __getitem__(self, key):
        return self.__dict__.__getitem__(key)

    def __call__(self):
        return self.__dict__

    def __str__(self):
        # Get string for variables and functions
        vStr = 'Variables:\n'
        mStr = 'Methods:\n'
        cStr = 'Classes:\n'
        for item in self.__dict__.items():
            # check if item is a variable
            if array(item[1]).dtype <> object or isinstance(item[1],dict):
                vStr += self._vStr(item)
            elif isinstance(item[1], types.FunctionType):
                mStr += self._mStr(item)
            elif isinstance(item[1], types.MethodType):
                mStr += self._mStr(item)
            else:
                cStr += self._cStr(item,)

        return vStr + mStr + cStr


    def _vStr(self, item, height = 1, vStr = ''):
        try:
            if isinstance(item[1], dict):
                if height == 1:
                    vStr += '  %s:\n%s|--' %(item[0],' '*(height)*2)
                else:
                    vStr += '%s:\n%s|--' %(item[0],' '*(height)*2)
                height += 1
                for item in item[1].items():
                    vStr += self._vStr(item, height)
                return vStr
            elif array(item[1]).shape == ():
                #~ if height > 1:
                    #~ height = 0
                return '%s%s : %s (%s)\n' %(' '*height*2, item[0], item[1],
                                          dtype(type(item[1])).name)
            else:
                #~ if height > 1:
                    #~ height = 0
                return '%s%s : %s %s (%s)\n' %(' '*height*2, item[0], item[1][:5],
                                          array(item[1]).shape,
                                          array(item[1]).dtype.name)
        except:
            return 'Not resolved!'

    def _mStr(self, item, height = 1):
        try:
            return '%s%s : %s (%s)\n' %(' '*height*2, item[0], item[1].func_name,
                                        item[1])
        except:
            return 'Not resolved!'

    def _cStr(self, item, height = 1):
        try:
            if array(item[1]).shape == ():
                return '%s%s : %s\n' %(' '*height*2, item[0],
                                          str(type(item[1])).lstrip("<class''> "))[:-2]
            else:
                return '%s%s : %s %s\n' %(' '*height*2, item[0],
                                          str(type(item[1][0])).lstrip("<class''> ")[:-2],
                                          array(item[1]).shape)
        except:
            return 'Not resolved!'

    def set(self, key, value):
        self.__setitem__(key, value)

    def get(self, key):
        return self.__getitem__(key)

    def has_key(self, key):
        return self.__dict__.has_key(key)


class m_Dict(dict,object):

    def __init__(self, *args,**kwargs):
        dict.__init__(self,  **kwargs)
        object.__init__(self)
        self.__dict__.__init__(self, **kwargs)
        self.order = []
        if kwargs.has_key('dvar'):
            self.dvar = kwargs['dvar']
        else:
            self.dvar = 'd0'


    def __setitem__(self, key, value):
        if isinstance(key, str):
            if key.count('/') > 0:
                obj = self._getNode(key)
                if obj['key'].count('/') >0:
                    obj['node'].__makeNode(obj['key'][:obj['key'].find('/')])

                obj= self._getNode(key)
                obj['node'].__setitem__(obj['key'],value)
            else:
                dict.__setitem__(self,key,value)
                self.__dict__.__setitem__(key,value)

        else:
            dict.__setitem__(self,key,value)
            self.__dict__.__setitem__(key,value)

    def __getitem__(self, key):
        ret = []
        if _ndim(key) == 0:
            return self._getitem(key)
        else:
            for k in (key):
                ret.append(self._getitem(k))

            return ret

    def delete(self, key):
        obj = self._getitem(key)
        del obj

    def __delitem__(self, key):
        if isinstance(key, str):
            if key.count('/') > 0:
                obj = self._getNode(key)

                obj['node'].__delitem__(obj['key'])
            else:
                dict.__delitem__(self,key)
                self.__dict__.__delitem__(key)

        else:
            dict.__delitem__(self,key)
            self.__dict__.__delitem__(key)


    def _getitem(self,key):
        if isinstance(key, str):
            if key.count('/') > 0:
                obj = self._getNode(key)
                if obj['key'] == self.dvar and key[-2:] <> self.dvar:
                    return obj['node']
                else:
                    return obj['node'].__getitem__(obj['key'])
            else:
                return dict.__getitem__(self,key)

        else:
            return dict.__getitem__(self,key)


    def __makeNode(self, key):
        from string import join
        obj = self
        if key.count('/') > 0:
            keys = key.split('/')
            if self.has_key(keys[0]):
                obj = self.__getitem__(keys[0])
                tmp = deepcopy(obj)
                obj = m_Dict()
                obj.__setitem__(self.dvar,tmp)
            else:
                self.__setitem__(keys[0], m_Dict())
                obj = self.__getitem__(keys[0])

            obj.__makeNode(join(keys[1:],'/'))
        else:
            if self.has_key(key):
                tmp = deepcopy(self.__getitem__(key))
                self.__setitem__(key, m_Dict())
                self.__getitem__(key).__setitem__(self.dvar,tmp)
            else:
                self.__setitem__(key, m_Dict())

    def _getNode(self, key):
        from string import join
        obj= self
        if key.count('/')>0:
            keys = key.split('/')
            obj=obj._getNode(keys[0])
            if obj['key']==self.dvar:
                return obj['node']._getNode(join(keys[1:],'/'))
            else:
                return {'node':obj['node'], 'key':key}
        else:
            try:
                if isinstance(obj.__getitem__(key),dict):
                    return {'node':obj.__getitem__(key), 'key':self.dvar}
                else:
                    return {'node':obj, 'key':key}
            except:
                return {'node':obj, 'key':key}


    def has_key(self,key):
        keys = self.getKeys()
        for k in keys:
            for i in xrange(k.count('/')):
                if k.rsplit('/',i+1)[0] not in keys:
                    keys.append(k.rsplit('/',i+1)[0])

        try:
            self.__getitem__(key)
            return key in keys
        except:
            return False


    def set(self, key, value):
        if self.has_key(key):
            if isinstance(self.__getitem__(key),dict):
                try:
                    return self.__setitem__(key+'/'+self.dvar, value)
                except:
                    return self.__setitem__(key, value)
        self.__setitem__(key, value)

    def get(self, key,getWhole=False):
        if isinstance(self.__getitem__(key),dict) and not getWhole:
            try:
                return self.__getitem__(key+'/'+ self.dvar)
            except:
                self.__getitem__(key)
        return self.__getitem__(key)

    def isEmpty(self):
        return dict.__len__(self)==0

    def getKeys(self):
        keys=[]
        for key,value in dict.items(self):
            if isinstance(value,dict):
                try:
                    tmp=value.getKeys()
                except:
                    tmp=value.keys()
                for k in tmp:
                    keys.append(str(key)+'/'+str(k))
            else:
                keys.append(key)
        return keys

    def getValues(self):
        values=[]
        for key in self.getKeys():
            values.append(self.get(key))
        return values

    def getItems(self):
        return zip(self.getKeys(),self.getValues())

    def update(self, *args, **kwargs):
        dict.update(self, *args, **kwargs)
        self.__dict__.update(*args, **kwargs)
