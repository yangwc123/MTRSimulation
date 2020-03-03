#!/usr/bin/python
# -*- coding: utf-8 -*-
#  Januar 2009 (JL)

import os, tables, string, numpy, pylab, types, sys, copy

import LP_func      ;reload(LP_func)
import LP_data      ;reload(LP_data)
import LP_progress;         reload(LP_progress)

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

class m_fileData(LP_data.m_dataObject):
    """Datei Klasse.
    Enthält alle Daten einer Datei.

    Parameter:
    -----------
    dir         Pfad zum Elternverzeichnis der Datei
    fname       Name der Datei"""

    def __init__(self, fname = None, dir = None, derived_class=LP_data.m_dataObject):
        derived_class.__init__(self,name=fname.split('.')[0])
        self.fname=fname
        if dir is None:
            self.dir = '.'
        else:
            self.dir = dir
        self.is_plotfile = False
        self.data_loaded = False
        self.start_row = 0
        self.last_row = 10
        self.cols = {}
        self.head = []
        self.headsize = 0
        self.__change_eol()

    def __change_eol(self):
        from os import path
        #~ try:
        lines = []              # whole file is saved here and converted to tab seperated file
        change = False
        file = open(path.join(self.dir, self.fname), 'rb')
        for line in file.xreadlines():
            if line.count('\r') > 0 or line.count('\n\n') > 0:
                line=line.replace('\r','\n')
                line=line.replace('\n\n','\n')
                change = True
            if line[0] == '#' or line[0] in string.ascii_letters:
                line=line.replace('\r','\n')
                self.head.append(line)
            elif line.count(',') > 0:
                line=line.replace(',','\t')
                change = True
            elif line.count(' ') > 0:
                if line[0]==' ':
                    line=line[1:]
                line=line.replace(' ','\t')
                change = True
            if line[0]=='\t':
                line=line[1:]
                change = True

            lines.append(line)

            if len(lines) > 30 and not change:
                break
        if change:
            file.close()
            file = open(os.path.join(self.dir, self.fname), 'wb')
            file.writelines(lines)
            file.close()

    def check_fileType(self):
        from string import ascii_letters

        for line in self.head:
            # file is a Monte Carlo Simulation file
            if line[0:8] == '# MoC/UP':
                return m_simFile( self.fname, self.dir)
            # file is out of a measurment in Wuerzburg
            elif line.count('#') > 0 and line.count('='):
                return m_bhFile(self.fname, self.dir)
            elif line.count('#') > 0 and line.count('=') == 0:
		return m_bhFile(self.fname, self.dir)
            # fileheader only contains column names
            elif line[0] in ascii_letters:
                return m_shFile(self.fname, self.dir)
            # else file doesn't have header

        return m_nhFile(self.fname, self.dir)

    def _check_c_number(self):
        from os import path
        column_string = ['x', 'y', 'z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'
                             'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q']
        #check how many columns we have
        line = open(path.join(self.dir, self.fname), 'r').readlines()[-1]
        if line[0] == ' ': line = line[1:]
        if line.count('\t') > 0:
            line = line.split('\t')
        elif line.count(',') > 0:
            line = line.split(',')
        else:
            line = line.split(' ')

        n = 0
        for part in line:
            if part <> '' and part <>' ':
                n += 1

        for i in xrange(n):
            # check if the value is a int or double
            self.cols["%s" %column_string[i]] = i

    def matrix_data(self):
        return None

    def load_data(self):
        """Funktion load_data.
        Liest die Werte der eindimensional gespeicherten Daten aus
        """
        d= numpy.loadtxt(os.path.join(self.dir,self.fname),skiprows=self.headsize)

        self.start_row = self.find_start_stop(d,self.input)[0]
        if self.start_row < 0: self.start_row = 0
        self.last_row = self.find_start_stop(d,self.input)[1]
        if self.last_row == 0: self.last_row = d.shape[0]

        # if file has only one column with data I try to add a timeseries
        # from the header parameters "tstart", "tend" and "dp"
        if len(self.cols.values()) == 1:
            from numpy import linspace
            self.data[self.cols.keys()[0]]=d[self.start_row:self.last_row]
            if (self.header.has_key('tstart')
                and  self.header.has_key('tend')
                and self.header.has_key('dp')):
                self.cols['y']=1
                self.cols['t']=2
                self.data['y']=linspace(self.getHeader('tstart'),
                                self.getHeader('tend'), self.getHeader('dp'))
                self.data['t']=self.data['y'][:]
            self.data_loaded = True
        else:
            for name,column in self.cols.items():
                self.data[name]=d[self.start_row:self.last_row,column]
                self.data_loaded = True


    def getindex_hi_sort(self, value, first, last):
        if first >= last:
            return first
        middle  = ( first + last )/2
        if self.data["N"][middle] <= value :
            last = middle
        else:
            first = middle + 1

        return self.getindex_hi_sort(value, first, last)

    def getindex_low_sort(self, value, first, last):
        if first >= last:
            return first
        middle  = ( first + last )/2
        if self.data["N"][middle] >= value :
            last = middle
        else:
            first = middle + 1

        return self.getindex_low_sort(value, first, last)

    def _get_end(self, x, start):
        from numpy import isnan
        c = 0
        for i in xrange(len(x)-1,0,-1):
            if not isnan(x[i]):
                c += 1
                if c == 10:
                    return i+c-1
            else:
                c = 0
        return len(x)

    def _get_start(self, x):
        from numpy import isnan
        c = 0
        for i in xrange(len(x)):
            if not isnan(x[i]):
                c += 1
                if c == 5:
                    return i-c+2
            else:
                c = 0
        return len(x)

    def get(self, key, dict=None):
        raise AttributeError, str(key)
        #~ if dict is None:
            #~ if self.__dict__.has_key(key):
                #~ return self.__getattribute__(key)
            #~ elif self.header.has_key(key):
                #~ return self.header[key]
            #~ elif self.data.has_key(key):
                #~ return self.data[key]
            #~ elif self.cols.has_key(key):
                #~ return self.cols[key]
            #~ else:
                #~ print '%s: Kein Attribut mit der Bezeichnung "%s"' %(string.split(os.path.join(self.dir,self.fname),"/")[-1], str(key))
        #~ else:
            #~ if dict == 'h':
                #~ return self.header[key]
            #~ elif dict == 'd':
                #~ return self.data[key]
            #~ elif dict == 'c':
                #~ return self.cols[key]
            #~ else:
                #~ print '%s Kein Attribut mit der Bezeichnung "%s"' %(string.split(os.path.join(self.dir,self.fname),"/")[-1], str(key))

    def has_key(self, key):
        if self.__dict__.has_key(key):
            return True
        elif self.header.has_key(key):
            return True
        elif self.data.has_key(key):
            return True
        elif self.cols.has_key(key):
            return True
        else:
            return False

    def __call__(self):
        return (os.path.join(self.dir,self.fname))



class m_simFile(m_fileData):
    """Datei Klasse.
    Enthält alle Daten einer Datei.

    Parameter:
    -----------
    dir         Pfad zum Elternverzeichnis der Datei
    fname       Name der Datei"""

    def __init__(self, fname = None, dir = None):
        m_fileData.__init__(self, fname, dir)
        self.globals = ''
        if dir is not None:
            self.readHeader()

    #Read Header and get parameters
    def readHeader(self):
        """Funktion readHeader.
        Liest alle Parameter aus dem header der Datei aus und speichert sie in der Tabelle h"""

        self.globals = string.split(string.split(self.head[0],'(')[3],')')[0]
        header1 = string.split(self.head[1],';')
        header2 = string.split(self.head[2],';')
        header3 = string.split(self.head[3],';')
        self.headsize = 4

        if string.split(string.split(header3[0],'=')[1], ' ')[0] == 't':

            self.header["nx"] = LP_func.atoi(string.split(self.globals,',')[0])
            self.header["ny"] = LP_func.atoi(string.split(self.globals,',')[1])
            self.header["nz"] = LP_func.atoi(string.split(self.globals,',')[2])

            for value in header1:
                # check if the value is a int or double
                if len(string.split(value,'=')) == 2:
                    key = string.split(value,'=')[0].replace('#','')
                    value = string.split(value,'=')[1].replace('#','')
                    # check if value is allready in the class-param-list
                    if key not in self.header:
                        if LP_func.isInt(value):
                            self.header[key] = LP_func.atoi(value)
                        else:
                            self.header[key] = LP_func.atof(value)



            for value in header2:
                # check if the value is a int or double
                if len(string.split(value,'=')) == 2:
                    key = string.split(value,'=')[0].replace('#','')
                    value = string.split(value,'=')[1].replace('#','')
                    # check if value is allready in the class-param-list
                    if key not in self.header:
                        if LP_func.isInt(value):
                            self.header[key] = LP_func.atoi(value)
                        else:
                            self.header[key] = LP_func.atof(value)

            column_string = string.split(header3[0],';')[0]
            column_string = string.split(string.split(column_string,'=')[1], ' ')
            while column_string.count(''):
                column_string.remove('')


            for value in header3[1:]:
                # check if the value is a int or double
                if len(string.split(value,'=')) == 2:
                    if string.split(value,'=')[1] <> 'nan':
                        key = string.split(value,'=')[0].replace('#','')
                        value = string.split(value,'=')[1].replace('#','')
                        # check if value is allready in the class-param-list
                        if key not in self.header:
                            if LP_func.isInt(value):
                                self.header[key] = LP_func.atoi(value)
                            elif string.split(value,'+-')[0] <> 'nan':
                                self.header[key] = LP_func.atof(string.split(value,'+-')[0])
                                if len(string.split(value,'+-')) > 1 and string.split(value,'+-')[1][1:] <> 'nan':
                                    self.header["%serror" %(key[:-1])] = LP_func.atof(string.split(value,'+-')[1][1:])
                    elif len(string.split(value,'=')) == 2:
                        self.header[key] = numpy.nan

            i = 0
            for value in column_string:
                if value not in self.cols:
                    # check if the value is a int or double
                    self.cols["%s" %value] = i
                    i += 1

             ## Get sort of carrier by filename ToDo: sort of carrier in header
            self.header["carriersort"] = string.split(os.path.split(self.fname)[1],'-')[1][0]

            self.is_plotfile = True
        #~ except:
            #~ try:
                #~ print "Error: reading Header for file: %s!" %self.fname
            #~ except:
                #~ print "Error: reading Header!"

    def matrix_data(self,x):
        """Funktion matrix_data.
        Liest die Werte der zweidimensional gespeicherten Daten aus (N_x, N_E, N_z)"""

        if not self.data_loaded:
            self.load_data()
        if x =='x':
            file = '%s_N_x.%s' %(string.split(self.fname,'.')[-2],string.split(self.fname,'.')[-1])
        elif x =='E':
            file = '%s_N_E.%s' %(string.split(self.fname,'.')[-2],string.split(self.fname,'.')[-1])
        elif x =='z':
            file = '%s_N_z.%s' %(string.split(self.fname,'.')[-2],string.split(self.fname,'.')[-1])
        else:
            file = ''
            'Parameter:["x","E","z"]'
        try:
            if x =='x':
                self.m["xd"] = numpy.loadtxt(os.path.join(self.dir,file),skiprows=4)
                self.data["xd"] = LP_func.meanValue(self.m["xd"][1:,1:],self.m["xd"][1:,0])[self.start_row:self.last_row]
                index = self.getindex_hi_sort(0.95,0,len(self.data["N"])-1)
                if index == 0:
                    index = self.getindex_hi_sort(self.data["N"][0]-0.05,0,len(self.data["N"])-1)
                if index > len(self.data["N"])-1: index = len(self.data["N"])-1

                self.header["xdmean"] = pylab.mean(pylab.nan_to_num(self.data["xd"][index-10:index]))
            elif x =='E':
                self.m["Ed"] = numpy.loadtxt(os.path.join(self.dir,file),skiprows=4)
                self.data["Ed"] = LP_func.meanValue(self.m["Ed"][1:,1:],self.m["Ed"][1:,0])[self.start_row:self.last_row]
                index = self.getindex_hi_sort(0.95,0,len(self.data["N"])-1)
                if index == 0:
                    index = self.getindex_hi_sort(self.data["N"][0]-0.05,0,len(self.data["N"])-1)
                if index > len(self.data["N"])-1: index = len(self.data["N"])-1

                self.header["Edmean"] = pylab.mean(pylab.nan_to_num(self.data["Ed"][index-10:index]))
            elif x =='z':
                self.m["zd"] = numpy.loadtxt(os.path.join(self.dir,file),skiprows=4)
                self.data["zd"] = LP_func.meanValue(self.m["zd"][1:,1:],self.m["zd"][1:,0])[self.start_row:self.last_row]
                index = self.getindex_hi_sort(0.95,0,len(self.data["N"])-1)
                if index == 0:
                    index = self.getindex_hi_sort(self.data["N"][0]-0.05,0,len(self.data["N"])-1)
                if index > len(self.data["N"])-1: index = len(self.data["N"])-1

                self.header["zdmean"] = pylab.mean(pylab.nan_to_num(self.data["zd"][index-10:index]))
        except:
            print 'Error: File: %s, Averaging not possible!' %(self.fname)

    def load_data(self):
        """Funktion load_data.
        Liest die Werte der eindimensional gespeicherten Daten aus"""
        #~ try:
        if self.fname[-7:] != 'rec.dat':
            d = numpy.loadtxt(os.path.join(self.dir,self.fname),skiprows=4)
            if numpy.any(d[:,self.cols["x"]] > 0):
                self.start_row = self._get_start(d[:,self.cols["x"]])
                if self.start_row < 0: self.start_row = 0
                self.last_row = self._get_end(d[:,self.cols["x"]], self.start_row+5)


                for name,column in self.cols.items():
                    self.data['%s'%name]=d[self.start_row:self.last_row,column]

                #calulated values
                self.header["nn"] = self.header["no_of_carriers"]/(1.*self.header["nx"]*self.header["ny"]*self.header["nz"])
                self.header["nnn"] = self.header["no_of_carriers"]/(1.*self.header["nx"]*self.header["ny"])
                self.header["L"] = self.header["L"]*self.header["ax"]
                self.header["nx"] = self.header["nx"]*self.header["ax"]
                self.header["ny"]= self.header["ny"]*self.header["ay"]
                self.header["nz"] = self.header["nz"]*self.header["az"]
                self.header["n0"] = self.header["no_of_carriers"]/(self.header["L"]*self.header["ny"]*self.header["nz"])

                try:
                    self.header["DsigmaH"] = self.header["Dsigma"]/(8.862e-5*self.header["T"])
                except:
                    self.header["DsigmaH"] = None
                try:
                    self.header["AsigmaH"] = self.header["Asigma"]/(8.862e-5*self.header["T"])
                except:
                    self.header["AsigmaH"] = None
                if self.header["F"] <> 0.0 and self.header["t_transit_mean"] <> 0.0:
                    self.header["mu_mean"] = self.header["L"]/(self.header["t_transit_mean"]*self.header["F"])*1e4
                self.header["c"] = 8.854e-12*self.header["Depsilon"]*self.header["ny"]*self.header["nz"]/(self.header["L"])

                self.data["j1"] = self.data["v"]*self.header["n0"]*1.6022e-19*self.data["N"]
                self.data["j2"] = self.calc_j2()
                self.data["I1"] = self.data["j1"]*self.header["ny"]*self.header["nz"]
                self.data["I2"] = self.data["j2"]*self.header["ny"]*self.header["nz"]

                index = self.getindex_hi_sort(0.95,0,len(self.data["N"])-1)
                if index == 0:
                    index = self.getindex_hi_sort(self.data["N"][0]-0.15,0,len(self.data["N"])-1)

                if index > len(self.data["N"])-1: index = len(self.data["N"])-1

                self.header["t_transit_85"] = self.data["t"][index]
                self.header["j_mean"] = pylab.mean(pylab.nan_to_num(self.data["j1"][index-15:index]))
                self.header["I_mean"] = pylab.mean(pylab.nan_to_num(self.data["I1"][index-15:index]))
                self.header["v_mean"] = pylab.mean(pylab.nan_to_num(self.data["v"][index-15:index]))
                self.header["Ea"] = pylab.mean(pylab.nan_to_num(self.data["deltaE"][index-15:index]))
                self.header["range"] = pylab.sum(pylab.nan_to_num(self.data["deltax"][:index]))
                self.header["mu_mean_j"] = pylab.mean(pylab.nan_to_num(self.data["v"][index-50:index]/self.header["F"]))*1e4
                self.header["mu_85"] = self.header["L"]/(self.header["t_transit_85"]*self.header["F"])*1e4
                self.header["Vg"] = self.header["no_of_carriers"]*1.6022e-19/(17e-5*self.header["nz"]*self.header["ny"])
                self.header["Vds"] = self.header["F"]*self.header["range"]
                if self.header["mode"] == 1:
                    if self.header["F"] <> 0.0 and self.header["t_transit_mean"] <> 0.0:
                        self.header["mu_mean"] = self.header["range"]/(self.header["t_transit_mean"]*self.header["F"])*1e4
                        self.header["mu_85"] = self.header["range"]/(self.header["t_transit_85"]*self.header["F"])*1e4
                    self.header["n2d"] = 1/(pylab.mean(pylab.nan_to_num(self.data["z"][index-15:index]+1))*1e-9*self.header["L"]*self.header["ny"])


                self.header["meanE"] = pylab.mean(pylab.nan_to_num(self.data["E"][index-15:index]))



                #~ Es = ['E', 'Eall', 'activationE', 'upE', 'deltaE']
                #~ for E in Es:
                    #~ try:
                        #~ self.header["t_%s" %E], Et = self.get_transportE(E, index-120, index)
                    #~ except ZeroDivisionError:
                        #~ print "dummy string"
                        #~ pass

##              else:
##                  print 'No Data in file %s' %self.fname

                self.data_loaded = True
        else:
            self.data["n_g"]=d[:,self.cols["n_g"]]
            self.data["nev_g"]=d[:,self.cols["nev_g"]]
            self.data["n_ng"]=d[:,self.cols["n_ng"]]
            self.data["nev_ng"]=d[:,self.cols["nev_ng"]]
            self.data_loaded = True

        #~ except:
            #~ print 'Error loading data from file %s' %self.fname
            #~ continue


    def calc_j2(self):
        x_temp = []
        j_temp = 0
        for i in xrange(len(self.data["deltax"])):
            if i == 0:
                x_temp.append(numpy.nan_to_num(self.data["deltax"][i]))
            else:
                x_temp.append(x_temp[i-1] + numpy.nan_to_num(self.data["deltax"][i]))

        j_temp= LP_func.derive(self.data["t"],x_temp,4)
        j_temp = j_temp*self.header["n0"]*1.6022e-19*self.data["N"]
        return pylab.array([pylab.where(j > 0, j, pylab.nan) for j in j_temp])

    def get_transportE(self, column, start, stop):
        try:
            b = self.data[column][start:stop]
            a = self.data["t"][start:stop]
            m,c = pylab.polyfit(pylab.log(a),b,1)
            E_t1 = m*pylab.log(a[100])+c
            E_t = m*pylab.log(a)+c
            return E_t1, E_t
        except KeyError:
            return 0, 0
            pass


class m_bhFile(m_fileData):
    """Datei Klasse.
    Enthält alle Daten einer Messung.

    Parameter:
    -----------
    dir         Pfad zum Elternverzeichnis der Datei
    fname       Name der Datei"""

    def __init__(self, fname = None, dir = None):
        m_fileData.__init__(self, fname, dir)
        self.headsize = 0
        self.find_start_stop = lambda x,y: [0,0]
        self.input = []
        if dir is not None:
            self.readHeader()

    #Read Header and get parameters
    def readHeader(self):
        """
        Funktion readHeader.
        Liest alle Parameter aus dem header der Datei aus und speichert sie in
        der Tabelle h
        """
        import re
        column_string = ''
        for h in self.head:
            self.headsize += 1
            h = h.replace('##','')
            h = h.replace('#','')
            h = h.replace('\r','')
            h = h.replace('\n','')
            if h.count('=') > 0:
                h = string.split(h,';')
                for value in h:
                    # check if the value is a int or double
                    if len(string.split(value,'=')) == 2:
                        key = string.split(value,'=')[0]
                        value = string.split(value,'=')[1]
                        # check if value is allready in the class-param-list
                        key=key.replace('__','/')
                        if (re.search('[a-zA-Z0-9]',value) is not None <> ''
                            and re.search('[a-zA-Z0-9]',key)):
                            if key not in self.header and (LP_func.isAscii(value)
                                or LP_func.isNumeric(value)):
                                if LP_func.isAscii(value):
                                    # if line starts with 'columns' then there
                                    # are the column namens defined
                                    # so this needs extra handling later
                                    if key.lower() == 'columns':
                                        column_string = string.split(value, ' ')
                                        while column_string.count(''):
                                            column_string.remove('')
                                    else:
                                        self.header[key] = value
                                elif LP_func.isInt(value):
                                    self.header[key] = LP_func.atoi(value)
                                elif LP_func.isFloat(value):
                                    self.header[key] = LP_func.atof(value)
                                else:
                                    self.header[key] = value

        if column_string != '':
            i = 0
            for value in column_string:
                if value not in self.cols:
                    # check if the value is a int or double
                    self.cols["%s" %value] = i
                    i += 1
        else:
            self._check_c_number()

        self.is_plotfile = True

class m_shFile(m_fileData):
    """Datei Klasse.
    Enthält alle Daten einer Messung.

    Parameter:
    -----------
    dir         Pfad zum Elternverzeichnis der Datei
    fname       Name der Datei"""

    def __init__(self, fname = None, dir = None):
        m_fileData.__init__(self, fname, dir)
        self.headsize = 0
        self.find_start_stop = lambda x,y: [0,0]
        self.input = []
        if dir is not None:
            self.readHeader()


    #Read Header and get parameters
    def readHeader(self):
        """
        Funktion readHeader.
        Liest alle Parameter aus dem header der Datei aus und speichert
        sie in der Tabelle h

        """
        # file has small header
        # I asume that header only contains column names

        for h in self.head:
            self.headsize += 1
            h = h.replace('##','')
            h = h.replace('#','')
            h = h.replace('\r','')
            h = h.replace('\n','')
            #check how column names are seperated
            spa = h.count(' ')
            com = h.count(',')
            tab = h.count('\t')
            if tab > spa:
                if tab > com:
                    column_string = string.split(h,'\t')
                else:
                    column_string = string.split(h,',')
            else:
                if spa > com:
                    column_string = string.split(h,' ')
                else:
                    column_string = string.split(h,',')
            i = 0
            for value in column_string:
                if value not in self.cols:
                    # check if the value is a int or double
                    self.cols["%s" %value] = i
                    i += 1

            self.is_plotfile = True

class m_nhFile(m_fileData):
    """Datei Klasse.
    Enthält alle Daten einer Messung.

    Parameter:
    -----------
    dir         Pfad zum Elternverzeichnis der Datei
    fname       Name der Datei"""

    def __init__(self,  fname = None, dir = None):
        m_fileData.__init__(self, fname, dir)
        self.headsize = 0
        self.find_start_stop = lambda x,y: [0,0]
        self.input = []
        if dir is not None:
            self.readHeader()


    #Read Header and get parameters
    def readHeader(self, h_file=None):
        """
        Funktion readHeader.
        Liest alle Parameter aus der gemeinsamen Headerdatei h_file

        """
        # file has small header
        # I asume that header only contains column names
        from os import path
        if h_file is None or not path.isfile(h_file):
            print '!! WARNING: No file with data informations is given !!'
            print ' Columns will be set: "x", "y", "a", "b", "c", ...'

            self._check_c_number()

            self.is_plotfile = True
        else:
            f = open(path.join(self.dir, h_file), 'r')
            for line in f.readlines():
                if line.count(':') > 0 and line.count('#') == 0:
                    key = line.split(':')[0].replace(' ','')
                    value = line.split(':')[-1]
                elif line.count('=') > 0 and line.count('#') == 0:
                    key = line.split('=')[0].replace(' ','')
                    value = line.split('=')[-1]
                    no = ''
                    for l in value:
                        if l.isdigit() or l in ('.','e','-','+') :
                            no += l


                self.header[key] = LP_func.atof(no)

class m_fileHandler(LP_data.m_dataHandler):
    """Datei Handler Klasse.
    Enthält eine Liste von Objekten des Typs m_fileData.

    Parameter:
    -----------
    fileObjects     Eine Liste mit Objekten des Typs m_fileData (default: [])

    """

    def __init__(self,objects=None,derived_class=LP_data.m_dataHandler,**kwargs):
        derived_class.__init__(self,objects)

    def load_files(self, dir=None, filter='.*', numberRange=[0,-1,1],
                   arglist = None, valuelist = None):
        """Funktion load_files.
        Erstellt für alle Dateien des Verzeichnisses "dir" ein Objekt der
        Klasse m_fileData.

        Parameter:
        -----------
        dir         Verzeichnis in dem die Dateien gespeichert sind
                    (default: None)
        filter      Filter für Dateinamen (Regex expression!)
        numberRange To cut a slice from the list of files, [start,stop,step]
                    (default: [0,-1,1])
        arglist     Liste der Filterparameter
        valuelist   Liste der Parameterwerte"""

        if dir == None:
            dir = os.getcwd()
        if os.path.isdir(dir):
            dir = os.path.abspath(dir)
            files = LP_func.filterStrings(os.listdir(dir),filter)
            files.sort(LP_func.strSort_cmp)
        else:
            print 'Path "%s" not found' %dir
            return 0
        fOs=[]
        if len(files) == 0:
            print 'No files in "%s" match the filter "%s"' %(dir, filter)
            return 0
        else:
            progBar = LP_progress.m_progressBar(len(LP_func.getObjRange(files,numberRange[0],
                                                                  numberRange[1],
                                                                  numberRange[2])),
                                                                  '% Reading headers ...')
            for file in LP_func.getObjRange(files,numberRange[0],
                                                  numberRange[1],
                                                  numberRange[2]):

                #~ try:
                if os.path.isfile(os.path.join(dir,file)):
                    fO = self.__load_file(dir,file)
                    toAdd = True
                    if arglist is not None:
                        for i in xrange(len(_asarray(arglist))):
                            if fO.get(_asarray(arglist)[i]) <> valuelist[i]:
                                add = False
                    if toAdd:
                        fOs.append(fO)
                progBar.update()

            progBar.finish()

            self.add(fOs)
                #~ except:
                    #~ print "Error: Loading file %s" %file
                    #~

    def __load_file(self, dir=None, filename=None):
        fO = m_fileData(os.path.basename(filename),dir)
        fO = fO.check_fileType()
        return fO


    def getDoubles(self,keys=None):
        """Funktion get_doubles.
        Gibt eine Liste mit doppelten m_fileDatas zurück

        Rückgabewert:
        -------------
        Eine Liste mit Objekten vom Typ m_fileData"""
        self.doubles = None
        self.doubles = []
        if keys is None:
            keys = ['A', 'T', 'Vmax', 'Voff','delay','Ft']

        cons = self.lookup.getConditions('',keys)
        for con in cons:
            names = self.lookup.getValues(con,'name')
            if names is not [[]] and len(names[0])>1:
                for name in _asarray(names[1:]):
                    self.doubles.append(name)
        return self.doubles


    def load_data(self,condition=''):
        if condition <> '':
            self.__init__(copy.copy(self.getObjects(condition).getSorted()))
        progBar = LP_progress.m_progressBar(len(self.objects.values()),'% Loading data ...')
        for fO in self.objects.values():
            if not fO.data_loaded:
                fO.load_data()
            progBar.update()
        progBar.finish()

    def reset_data(self):
        for fO in self.objects.values():
            fO.data.clear()
            fO.data_loaded = False

    def removeObjects(self, condition, fnc=lambda x: True):
        vals=self.lookup.getWhere(condition,fnc)
        for val in vals:
            del self.objects[val]

    def setHeaderAll(self, key, values, islist = False):
        if islist is False:
            for fO in self.objects.values():
                fO.setHeader(key, values)
        else:
            for i in xrange(len(self.getSorted())):
                self.getSorted()[i].setHeader(key,values[i])

    def setDataAll(self, key, values, islist = False):
        if islist is False:
            for fO in self.objects.values():
                fO.setData(key, values)
        else:
            for i in xrange(len(self.getSorted())):
                self.getSorted()[i].setData(key,values[i])

    def clear(self):
        for i in xrange(len(self.objects)-1,-1,-1):
            self.objects.pop(i)
            self.__init__()

