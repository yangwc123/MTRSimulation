#!/usr/bin/python
# -*- coding: utf-8 -*-
#  Januar 2009 (JL)

import LP_object                  ;reload(LP_object)
import LP_fitting                 ;reload(LP_fitting)
import LP_plot                    ;reload(LP_plot)
import LP_files                   ;reload(LP_files)
import LP_func                    ;reload(LP_func)
import LP_data                    ;reload(LP_data)
import LP_saveload                ;reload(LP_saveload)
from numpy import asarray

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


class m_Project(LP_object.m_Object):
    """Projekt Klasse.
    Kann FileHandler, DataHandler oder Plot-Objekte aufnehmen, speichern und laden.

    Parameter:
    -----------
    name            Name des Projekts
    description     Kurze Beschreibung des Projekts (default: none)
    fileHandler     Ein Object der Klasse m_FileHandler (default: none)
    DataHandlers    Ein Array von Objecten der Klasse m_dataHandler (default: [])
    plotObject      Ein Array von Objecten der Klasse m_Plot (default: [])
    fileSlices      Ein Array von Objecten der Klasse m_FileHandler (default: [])
            fileSlices bilden Untergruppen aller Dateien im Project"""

    class m_slice(LP_object.m_Object):
        def __init__(self, name = None, description = None, fileHandler = None, dataHandlers = None, plotHandlers = None):
            self.name = None
            self.name = name
            self.description = None
            self.description = description
            self.fileHandler = None
            self.fileHandler = fileHandler
            self.dataHandlers = None
            self.dataHandlers = []
            if dataHandlers is not None:
                if isinstance(dataHandlers,LP_data.m_dataHandler):
                    self.dataHandlers.append(dataHandlers)
                elif isinstance(dataHandlers,list):
                    for dH in dataHandlers:
                        self.dataHandlers.append(dH)
            self.plotHandlers = []
            if plotHandlers is not None:
                if isinstance(plotHandlers,LP_plot.m_Plot):
                    self.plotHandlers.append(plotHandlers)
                elif isinstance(plotHandlers,list):
                    for pH in plotHandlers:
                        self.plotHandlers.append(pH)

        def add(self, obj):
            """Function add:
            Fügt ein Objekt oder eine Liste von Objekten der Klassen
                - m_FileHandler
                - m_dataHandler
                - m_Plot
            hinzu.

            Parameter:
            ----------
            obj:            Objekt oder eine Liste von Objekten der Klassen
                                - m_FileHandler
                                - m_dataHandler
                                - m_Plot"""

            if asarray(obj).ndim == 0:
                if LP_object.isinstanceof(obj, LP_files.m_fileHandler) or \
                   LP_object.isinstanceof(obj, LPlot.LP_files.m_fileHandler):
                    self.fileHandler = obj
                elif LP_object.isinstanceof(obj,LP_data.m_dataHandler) or \
                   LP_object.isinstanceof(obj, LPlot.LP_data.m_dataHandler):
                    self.dataHandlers.append(obj)
            else:
                if LP_object.isinstanceof(obj[0], LP_files.m_fileData) or \
                   LP_object.isinstanceof(obj, LPlot.LP_files.m_fileData):
                    self.fileHandler = LP_files.m_fileHandler(obj)
                elif LP_object.isinstanceof(obj[0],LP_data.m_Data) or \
                   LP_object.isinstanceof(obj, LPlot.LP_data.m_Data):
                    self.dataHandlers.append(LP_data.m_dataHandler(obj))
                else:
                    for o in asarray(obj):
                        if LP_object.isinstanceof(o,LP_data.m_dataHandler) or \
                           LP_object.isinstanceof(obj, LPlot.LP_data.m_dataHandler):
                                self.dataHandlers.append(o)
                        elif LP_object.isinstanceof(o,LP_plot.m_Plot) or \
                            LP_object.isinstanceof(obj, LPlot.LP_plot.m_Plot) :
                            self.plotHandlers.append(o)
                        else:
                            print '%s kann nicht im Projekt verwaltet werden.\n Objekte muessen von der Klasse "LP_files.m_fileHandler", "m_dataHandler" oder "m_Plot" sein.' %str(o.__class__).split("'")[1]

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

    def __init__(self, name = None, description = None, fileHandler = None):
        self.name = None
        self.name = name
        self.description = None
        self.description = description
        self.fileHandler = None
        self.fileHandler = fileHandler
        self.slices = None
        self.slices = []
        self.n_slices = 0

    def new_Slice(self, name = None,  description = None):
        self.slices.append(self.m_slice(name, description))
        self.n_slices = len(self.slices)
        self.__dict__['_'+str(len(self.slices)-1)]=self.slices[-1]


    def get_slice_count(self):
        return self.n_slices


    def add_toSlice(self, obj, sliceNumber = None, name = None,  description = None):
        """
        Function add:
        Fügt ein Objekt oder eine Liste von Objekten der Klassen
            - m_FileHandler
            - m_dataHandler
            - m_Plot
        der mit "sliceNumber" übergebenen Untergruppe hinzu.

        Parameter:
        ----------
        obj:            Objekt oder eine Liste von Objekten der Klassen
                            - m_FileHandler
                            - m_dataHandler
                            - m_Plot
        sliceNumber:     Nummer des Slices, zu der das Objekt hinzugefügt
                         werden soll. Für "None" wird ein neuer Slice angelegt
                         (default: -1)

        )"""

        if len(self.slices) <= sliceNumber or sliceNumber is None:
            if len(self.slices) >= sliceNumber:
                self.new_Slice(name=name, description=description)
                self.slices[-1].add(obj)
            else:
                print "Error: sliceNumber too high (sliceNumber > %d )" %len(self.slices)
        else:
            self.slices[sliceNumber].add(obj)

    def save(self, fname):
        """Function save:
        Speichert das Projekt.

        Parameter:
        ----------
        fname:    Name der Datei, in der das Project gespeichert werden soll"""

        ioH = LP_saveload.m_ioHandler(self,  fname)
        ioH.save()
        ioH.close()


    def load(self, fname):
        """Function load:
        Läd aus der Datei "fname" ein Projekt.

        Parameter:
        ----------
        fname:    Name der Datei, aus der das Project geladen werden soll"""
        ioH = LP_saveload.m_ioHandler(self,  fname)
        ioH.load()
        ioH.close()



    def clear(self):
        """Function clear:
        Löscht alle Objekte des Projekts
        """
        self.description = None
        self.fileHandler = []
        self.fileHandler = None
        for i in reversed(xrange(len(self.dataHandlers))):
            self.dataHandlers.pop(i)
        for i in reversed(xrange(len(self.plotHandlers))):
            self.plotObjects.pop(i)
        for i in reversed(xrange(len(self.plotHandlers))):
            self.fileSlices.pop(i)



