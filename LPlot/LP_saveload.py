#!/usr/bin/python
# -*- coding: utf-8 -*-
#  Januar 2009 (JL)

import pylab, numpy, tables, os,copy, sys
import LP_object;       reload(LP_object)
import LP_func;         reload(LP_func)
import LP_progress;     reload(LP_progress)
import re

#~ import LPlot
#~ import LP_fitting
#~ import LP_plot
#~ import LP_files
#~ import LP_func
#~ import LP_data



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


class m_ioHandler(LP_object.m_Object):
    """Saving Klasse.
    Kann FileHandler, DataHandler oder Plot-Objekte speichern und laden.

    Parameter:
    -----------
    name            Name des Projekts
    description     Kurze Beschreibung des Projekts (default: none)
    fileHandler     Ein Object der Klasse m_FileHandler (default: none)
    DataHandlers    Ein Array von Objecten der Klasse m_DataHandler (default: [])
    plotObject      Ein Array von Objecten der Klasse m_Plot (default: [])
    fileSlices      Ein Array von Objecten der Klasse m_FileHandler (default: [])
                    fileSlices bilden Untergruppen aller Dateien im Project"""


    def __init__(self,  obj, fname,derived_class=LP_object.m_Object):
        derived_class.__init__(self)
        self.obj = None
        self.obj = obj
        self.fname = fname
        self.h5file = None
        self.des = None

    def close(self):
        self.h5file.flush()
        self.h5file.close()

    def _saveT(self,obj, where, des, counter = None, parent_key = None):
        """Function do_save:
        Kümmert sich um das SPEICHERN mit den richtigen Parametern, etc.

        Parameter:
        ----------
        obj:    Objekt, das gespeichert werden soll
        where:  Pfad unter dem das Objekt gespeichert wird
        maxl:   Maximale Länge für eine Liste von Daten"""

        if isinstance(obj, dict):
            items = obj.items()
        else:
            items = obj.__dict__.items()
        for (key, value) in items:
            if not isinstance(value, dict):
                try:
                    if des.has_key(key):
                        if parent_key is not None:
                            nkey = parent_key + '/' + key
                        else:
                            nkey = key
                        if des[key].dflt <> '' and des[key].dtype == '|S64' :
                            where.row[nkey] = where._v_pathname.rsplit('/',1)[0] \
                                            + '/' + des[key].dflt + '_' + str(counter)
                        elif des[key].dflt <> '' and des[key].dtype == '|S63' :
                            where.row[nkey] = where._v_pathname.rsplit('/',1)[0] \
                                            + '/' + des[key].dflt
                        else:
                            where.row[nkey] = value
                except:
                    print 'Error: Key %s can not save object of type %s' %(key,type(value))
                    print '       Expected: %s' %(des[key].dtype)
            else:
                if parent_key is not None:
                    nkey = parent_key + '/' + key
                else:
                    nkey = key
                self._saveT(value,where,des[key],counter,nkey)

    def _saveA(self,obj, where, counter):
        """Function do_save:
        Kümmert sich um das SPEICHERN mit den richtigen Parametern, etc.

        Parameter:
        ----------
        obj:    Objekt, das gespeichert werden soll
        where:  Pfad unter dem das Objekt gespeichert wird
        maxl:   Maximale Länge für eine Liste von Daten"""

        if isinstance(obj, dict):
            items = obj.items()
        else:
            items = obj.__dict__.items()
        inc = False
        for (key, value) in items:
            if _ndim(value) > 0:
                if _canarray(value):
                    if _asarray(value).dtype <> 'object' and _asarray(value).shape[0] > 0:
                        self.h5file.createArray(where, str(key + '_' + str(counter)), _asarray(value))
                        inc = True
                        self.h5file.flush()
                else:
                    _type = tables.Atom.from_dtype(numpy.dtype((_asarray(value[0]).dtype)))
                    VLArray = self.h5file.createVLArray(where, str(key + '_' + str(counter)), _type)
                    for row in value:
                        VLArray.append(_asarray(row))
                    inc = True
                    self.h5file.flush()
        if inc: counter += 1
        return counter


    def _walk_saving(self, obj = None,  where = None):
        """Function start_saveing:
        Speichert das Projekt rekursiv.

        Parameter:
        ----------
        obj:    Objekt, das gespeichert werden soll
        where:  Pfad, unter dem gespeichert werden soll
        node:   Knoten, in dem gespeichert werden soll"""

        # Erstelle die Tabellen-Beschreibung zum speichern der Dateien
        des, objs = self.get_description(obj)
        self.des = des

        if des is not None and objs is not None:
            if des <> {}:
                groupH = where

                if not groupH._v_children.has_key('vars'):
                    tableH = self.h5file.createTable(groupH, 'vars', des.copy())
                else:
                    tableH = groupH.vars

                counter = 0
                for kind in _asarray(obj):
                    self._saveT(kind, tableH, des, counter)
                    tableH.row.append()
                    tableH.flush()
                    counter = self._saveA(kind, groupH, counter)
                    self.h5file.flush()

            if objs <> {}:
                for key, value in objs.items():
                    groupH1 = self.h5file.createGroup(groupH,  key)

                    if isinstance(_asarray(value)[0], LP_object.m_Object) and \
                       _asarray(value)[0] is not None:
                        self._walk_saving(value,  groupH1)

    def save(self):
        """Function save:
        Speichert das Projekt rekursiv.

        Parameter:
        ----------
        obj:    Objekt, das gespeichert werden soll
        where:  Pfad, unter dem gespeichert werden soll
        node:   Knoten, in dem gespeichert werden soll"""

        #
        # Lösche hdf5-Datei, wenn es sie schon gibt
        # Aus irgend einem Grund kann pytables schon vorhandene
        # Dateien nicht öffnen
        #
        if os.path.isfile(self.fname):
            os.remove(self.fname)
        # Öffne die Datei
        self.h5file = tables.openFile(self.fname, mode = "w",
                    title = 'LPlot Saving: Saved Object is "%s"'
                    %(str(_asarray(self.obj)[0].__class__).lstrip("<class''> ")
                    [:-2]))

        # Alle Zeiger und Referenzen der Plots müssen zwischengespeichert werden
        #~ tmp_saveH = []
        #~ for slice in self.project.slices:
            #~ for plot in slice.plotHandlers:
                #~ tmp_saveH.append(m_saveHandler(plot))

        print '=' * 60
        print self.h5file
        print '=' * 60
        # Das zu speichernde Objekt ist das bei der Initialisierung übergebene Objekt
        # Erstelle die Root-Gruppe
        groupH = self.h5file.root
        self._walk_saving(self.obj, groupH)


        #~ for slice in xrange(len(self.project.slices)):
            #~ for plot in xrange(len(self.project.slices[slice].plotHandlers)):
                #~ tmp_saveH[slice+plot].reset(self.project.slices[slice].plotHandlers[plot])

    def _loadT(self, row,  names, obj=None):
        type = row['self']
        imp = type.split('.')[0]
        eval('import ' + imp)
        eval('reload ' + imp)
        if obj is None:
            obj = eval(type + '()')
        for name in names:
            obj.set[name] = row[name]


    def load(self, obj = None,  groupH = None,  type = None):
        """Function save:
        Läd aus der Datei "fname" ein Projekt rekursiv.

        Parameter:
        ----------
        obj:    Objekt, das geladen werden soll
        groupH: Pfad, aus dem geladen werden soll
        type:   Datentype des zu ladenden Objekts"""

        # Öffne Datei zum schreiben
        if self.h5file is None:
            # Öffne die Datei
            self.h5file = tables.openFile(self.fname, mode = "r")

            print '=' * 60
            print self.h5file
            print '=' * 60
            # Das zu speichernde Objekt ist das bei der Initialisierung übergebene Objekt
            #~ obj= self.obj
            groupH = self.h5file.root
            self.obj = self._loadT(groupH.vars, groupH.vars.colpathnames)
            obj = self.obj



        for row in groupH.vars:
            self._loadT(groupH.vars, groupH.vars.colpathnames)

        for group in groupH:
            getO = m_Object()
            if isinstance(group, tables.group.Group):
                Gname = (group._v_pathname.split('/')[-1])
                if type is None:
                    obj1 = obj.get(Gname)
                    if obj1 is None:
                        obj.set(Gname,  getO.get_Obj(Gname))
                        obj1 = obj.get(Gname)
                else:
                    if type is not None:
                        if pylab.array(obj).size == int(Gname.split('_')[-1]):
                            tmp = getO.get_Obj(type)
                            obj.append(tmp)
                    obj1 = obj[int(Gname.split('_')[-1])]
                if isinstance(obj1,  tuple) or isinstance(obj1,  list):
                    type1 = Gname
                else:
                    type1 = None
                a = self.project
                self.load(obj1, group,  type1)

    def get_description(self, obj):
        """Funktion description.
        Gibt ein Dictionary als Tabellen-Beschreibung für pytables zurück"""
        tmp = {}
        objs = {}
        if not isinstance(obj,tables.file.File):
            if isinstance(obj,  list) or isinstance(obj, tuple):
                return self.get_description(obj[0])
            else:
                if isinstance(obj, dict):
                    items = obj.items()
                else:
                    items = obj.__dict__.items()
                if len(items) > 0:
                    tmp['self'] = tables.StringCol(64,
                            dflt=str(_asarray(obj)[0].__class__).lstrip("<class''> ")[:-2])
                for (key, value) in items:
                    # to get maximum size for string column value is setto a
                    # default string with length 64
                    if isinstance(value, str):
                        value = '*'*64
                    if _canarray(value):
                        if _ndim(value) == 0 and _asarray(value).dtype <> object:
                            tmp[key] = tables.Col.from_dtype(
                                       numpy.dtype((_asarray(value).dtype)))

                        elif _asarray(value).dtype <> object:
                            tmp[key] = tables.StringCol(64, dflt=key)
                            #~ tmp[key] = tables.Col.from_dtype(
                                        #~ numpy.dtype((numpy.array(value).dtype,
                                        #~ numpy.array(value).shape)))
                        elif value is None:
                            tmp[key] = tables.StringCol(64)
                        elif isinstance(value, dict):  # instance is a dictionary
                            val = self.get_description(value)
                            del tmp['self']
                            if val[0] is not None:
                                if val[0].has_key('self'):
                                    del val[0]['self']
                                tmp[key] = val[0]
                            else:
                                tmp[key] = tables.StringCol(64)

                        elif key <> 'h5file':
                            tmp[key] = tables.StringCol(63, dflt=str(key))
                            objs[key] = value
                    else:
                        tmp[key] = tables.StringCol(64, dflt=str(key))

        if tmp <> {} or objs <> {}:
            return tmp, objs
        else:
            return None, None


    def clear(self):
        """Function clear:
        Löscht alle Objekte des Projekts
        """
        self.project = None


class m_dumb(object):
    """Klasse m_dumb:
    Default Klasse, die vor dem Laden von Daten einem nicht initialisiertem Objekt zugewisen werden kann.
    Dann können die geladenen Daten leicht in das"""
    def __init__(self):
        self.index = {'Plot' : LP_plot.m_Plot(),
                      'plotHandler' : LP_plot.m_Plot(),
                      'pData' : LP_plot.m_plotData(),
                      'plotObject' : LP_plot.m_Plot(),
                      'plotData' : LP_plot.m_plotData(),
                      'fileHandler' : LP_files.m_fileHandler(),
                      'fileObject' :LP_files.m_fileObject(),
                      'dataHandler' : LP_data.m_dataHandler(),
                      'dataObjects' : LP_data.m_Data(),
                      'matrix' : LP_data.m_Data.matrix(),
                      'Project' : LPlot.m_Project(),
                      'slice' : LPlot.m_Project.m_slice()}

    def get_Obj(self, name):
        if str(name).count('_') == 1:
            name = str(name).split('_')[1]
        if str(name)[-1]== 's':
            name = str(name)[:-1]
        if self.index.has_key(str(name)):
            return self.index[str(name)]
        else:
            return None

class m_saveplotHandler(object):
        """Projekt Klasse.
        Speichert temporär alle Plot-Handler während des speicherns"""
        def __init__(self, plot):
            self.ax1 = []
            self.ax2 = []
            self.lines = []
            self.legend = []
            self.pData = []
            self.fncs = []
            self.ax1.append(plot.get('ax1'))
            self.ax2.append(plot.get('ax2'))
            self.lines.append(plot.get('lines'))
            self.legend.append(plot.get('legend'))
            self.fncs.append(plot.get('fncs'))
            self.pData.append(plot.get('pData'))

            plot.set('ax1',None)
            plot.set('ax2',None)
            plot.set('lines',None)
            plot.set('legend',None)
            plot.set('fncs',None)
            for pD in plot.pData:
                pD.set('line',None)
                pD.set('ax',None)

        def reset(self, plot):
            plot.set('ax1',self.ax1)
            plot.set('ax2',self.ax2)
            plot.set('lines',self.lines)
            plot.set('legend',self.legend)
            plot.set('fncs',self.fncs)
            plot.set('pData',self.pData)

class m_IgorSave(object):

    def __init__(self):#,derived_class=LP_object.m_Object):
        #~ derived_class.__init__(self)
        self.strip=''
        pass

    def __getDescription(self,items,des=None):
        #try:
            from numpy import dtype
            if des is None:
                tmp = {}
            else:
                tmp = des
            for key,value in items:
                key=key.replace('/','__').replace(self.strip,'').lstrip('_')
                #key=key.replace('_','__').replace(self.strip,'')
                key=key.replace('__d0','')
                if isinstance(value, str):
                    value = '*'*64

                if _ndim(value)>0 and not _asarray(value).dtype == 'object':
                    if _asarray(value).dtype == object:
                        if numpy.nan in value:
                            value = where( numpy.isnan(value), 0., v )
                        if numpy.inf in value:
                            value = where( numpy.isinf(value), 1e300, v )
                        if -numpy.inf in value:
                            value = where( numpy.isinf(value), -1e300, v )
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
                        tmp[key] = tables.Col.from_dtype(dtype((dtype('float64'),
                                                _asarray(value).shape)),dflt=-numpy.inf)
                    else:
                        tmp[key] = tables.Col.from_dtype(dtype((_asarray(value).dtype,
                                                _asarray(value).shape)))
                elif _asarray(value).dtype <> object:
                    if LP_func.isNumeric(_asarray(value)[0]):
                        tmp[key] = tables.Col.from_dtype(dtype(dtype('float64')),dflt=-numpy.inf)
                    else:
                        tmp[key] = tables.Col.from_dtype(dtype(_asarray(value).dtype))
                    #~ tmp['name'] = tables.StringCol(32)

#            tmp['prefix'] =tables.StringCol(64)
            return tmp
        #except:
        #    pass

    def __saveHeader(self,header,where):
        if isinstance(header, dict):
            items = header.getItems()
        else:
            items = header.__dict__.items()
        for key, value in items:
            key=key.replace('/','__').replace(self.strip,'').lstrip('_')
            #key=key.replace('_','__').replace(self.strip,'')
            key=key.replace('__d0','')
            if not isinstance(value, dict):
                try:
                    if _ndim(value) > 0:
                        where.row[key] = _asarray(value)
                    else:
                        where.row[key] = value
                except:
                    pass

#            if key == 'name':
#                name = re.findall('^[_a-zA-Z0-9]*',value)[0]
#                where.row['prefix'] = name.replace(self.strip,'') + '_'

        where.row.append()
        where.flush()

    def __saveData(self,obj,where,name=None,counter=None):
        if isinstance(obj, dict):
            items = obj.getItems()
        else:
            items = obj.data.getItems()
        if name is None:
            name = re.findall('^[_a-zA-Z0-9]*',obj.header['name'])[0]
            counter=str(obj.header['counter'])
        for key,value in items:

            key=key.replace('/','__').replace(self.strip,'').lstrip('_')
            #key=key.replace('_','__').replace(self.strip,'')
            key=key.replace('__d0','')

            if key not in where._v_attrs.items:
                where._v_attrs.items.append(key)
                where._v_attrs.items=where._v_attrs.items

            key = (name.replace(self.strip,'')+'_'+key).lstrip('_')

            if value == []:
                value = [numpy.nan]*100

            if not isinstance(value, dict):
                array = self.fileH.createArray(where, key, value)

    def saveit(self,fname, strip='', node=None, mode='w' ):
        import time
        import warnings
        warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)
        from os import path,remove
        if strip is not '':
            self.strip=strip

        else:
            if self.o0.header['name'][:3] == 'def':
                self.strip = 'ef_'

        if path.isdir(path.split(path.abspath(fname))[0]):
            if path.isfile(path.abspath(fname)) and mode=='w':
                remove(path.abspath(fname))

        if node is None:
            node='node'

        self.fileH = tables.openFile(path.abspath(fname) , mode=mode)
        i=1
        while node in self.fileH.root._v_children.keys():
            node = '%s_%d' %(node,i)
            i+=1

        self.nGroup = self.fileH.createGroup(self.fileH.root, node, 'Data from python session')
        # Add some informations
        self.nGroup._v_attrs.date = time.ctime()
        self.nGroup._v_attrs.strip = self.strip
        self.nGroup._v_attrs.htype = str(self.__class__).lstrip("<class '").rstrip("'> ")
        self.nGroup._v_attrs.otype = str(self.getSorted()[0].__class__).lstrip("<class '").rstrip("'> ")
        self.nGroup._v_attrs.groups= []
        self.nGroup._v_attrs.tables = []

        # Create and fill the header tables and data group
        maxHeader={}
        for dat in self.getSorted():
            #~ if not maxHeader.has_key(dat.headersize()):
                maxHeader[dat.headersize()] = dat.header
        header=maxHeader[max(maxHeader)]



        # Use the table description from the data object
        if hasattr(self, 'lookup'):
            self.hTable = self.fileH.createTable(self.nGroup,'header',
                            self.lookup.des)
        else:
            self.hTable = self.fileH.createTable(self.nGroup,'header',
                            self.__getDescription(header.getItems()))

        # workaround to store all the data
        self.nGroup._v_attrs.tables.append('header')
        self.nGroup._v_attrs.tables = self.nGroup._v_attrs.tables

        # Add some informations

        self.dGroup = self.fileH.createGroup(self.nGroup, 'data', 'Data from python session')
        self.nGroup._v_attrs.groups.append('data')
        self.nGroup._v_attrs.groups = self.nGroup._v_attrs.groups
        self.dGroup._v_attrs.items=[]

        progBar = LP_progress.m_progressBar(len(self.objects.values()),'% Saving data ...')
        for obj in self.getSorted():
            self.__saveHeader(obj.header,self.hTable)
            if len(obj.data) > 0:
                self.__saveData(obj,self.dGroup)
            progBar.update()
        self.fileH.flush()
        self.fileH.close()



    def __loadHeader(self,obj,where,counter):

        for key in  where.colnames:
            dkey=key.replace('__','/').lstrip('/')
            #dkey=dkey.replace('_','/').lstrip('/')
            obj.setHeader(dkey, where.read()[key][counter])


    def __loadData(self,obj, where, counter):
        from string import atof

        name = re.findall('^[_a-zA-Z0-9]*',obj.header['name'])[0]
        counter=str(obj.header['counter'])
        data = where._v_children
        for key in  where._v_attrs.items:
            dkey=key.replace('__','/').replace(name.replace(self.strip,'').rstrip('_'),'').lstrip('/')
            dkey=dkey.replace('/'+counter,'')

            if not data.has_key('%s_%d' %(key,0)):
                key = key.replace(name.replace(self.strip,'')+'_', '')
                fkey = (name.replace(self.strip,'')+'_'+key).lstrip('_')
            else:
                fkey = key
            if dkey[0] == '_':
                dkey=dkey[1:]
            try:
                obj.setData(dkey,data['%s_%d' %(fkey,atof(counter))][:])
            except:
                try:
                    obj.setData(dkey,data[fkey][:])
                except:
                    pass
#            else:
#                print 'Key %s not found. %s, %s' %(fkey,dkey,key)



    def loadit(self, fname, condition='', node=None, **kwargs):
        from os import path
        condition=condition.replace('/','__')
        objs=[]
        if node is None:
            node='node'

        self.fileH = tables.openFile(path.abspath(fname) , mode='r')
        self.nGroup = self.fileH.root._v_groups[node]
        self.htype = self.nGroup._v_attrs.htype
        self.otype = self.nGroup._v_attrs.otype

        # Here I know that every node has a header table and a data group
        self.hTable = self.nGroup._v_children[self.nGroup._v_attrs.tables[0]]
        self.dGroup = self.nGroup._v_children[self.nGroup._v_attrs.groups[0]]

        exec('import %s' %self.htype.split('.')[0])
        exec('import %s' %self.otype.split('.')[0])

        self.strip = self.nGroup._v_attrs.strip
        print self.strip

        import LP_data

        if condition <> '':
            rows = self.hTable.getWhereList(condition,**kwargs)
        else:
            rows = xrange(self.hTable.nrows)
        progBar = LP_progress.m_progressBar(len(rows),'% Loading data ...')
        for i in rows:
            dO = None
            name = '_'.join(self.hTable.read()['name'][i].split('_')[:-1])+'.dat'

            dO=LP_data.m_dataObject()

            self.__loadHeader(dO, self.hTable, i)
            self.__loadData(dO, self.dGroup, i)

            objs.append(dO)
            progBar.update()

        progBar.finish()
        self.fileH.close()
        return objs




