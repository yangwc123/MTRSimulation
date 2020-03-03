#!/usr/bin/python
# -*- coding: iso-8859-1 -*-
#  Januar 2009 (JL)

import pylab, string

import LP_data

class m_Plot(object):
		def __init__(self, dataObjects = None, plot = 'plot', ax1 = None, ax2 = None, lines = [], usetetx = False, legend = True, xlabel = None, ylabel1 = None, ylabel2 = None, titel = None, save = False):
			import rainbow
			self.rainb = rainbow
			self.pData = []
			self.plot = plot
			self.ax1 = ax1
			self.ax2 = ax2
			self.lines = None
			self.lines = []
			self.legend = None
			self.ltexts = None
			self.tetxts = None
			self.texts = []
			self.xlabel = xlabel
			self.ylabel1 = ylabel1
			self.ylabel2 = ylabel2
			self.titel = titel
			self.x1 = None
			self.y1 = None
			self.x1 = []
			self.y1= []
			self.usetex = False
			self.fncs = [None,None,None,None]
			self.save = save
			self.muster = '%s / %s'
			self.order = [0,1]
			self.x1 = [0,0]
			self.y1 = [0,0]
			self.ccycle = None
			#['#660000','#990000','#CC0000','#FF0000','#FF5511','#FF6A00','#FF9500','#FFBF00','#FFDA00','#DDDA00'
			#	,'#808000','#609000','#209000','#009040', '#009080','#005080','#000080','#000099','#0000BB','#0000FF']
			if isinstance(dataObjects, LP_data.m_dataHandler):

				for dO in dataObjects.dataObjects:
					if isinstance(dO.param1, int):
						p1 = '%d' %dO.param1
					elif isinstance(dO.param1, float):
						p1 = '%.2e' %dO.param1
					elif isinstance(dO.param1, str):
						p1 = '%s' %dO.param1
					else:
						p1 = '%s' %pylab.array2string(pylab.array(dO.param1))
					if isinstance(dO.param2, int):
						p2 = '%d' %dO.param2
					elif isinstance(dO.param2, float):
						p2 = '%.2e' %dO.param2
					elif isinstance(dO.param2, str):
						p2 = '%s' %dO.param2
					else:
						p2 = '%s' %pylab.array2string(pylab.array(dO.param2))
					pD = m_plotData(dO.x, dO.y , [dO.x_axis, dO.y_axis], plot =  self.plot, param = list((p1, p2)))
					self.pData.append(pD)

			elif isinstance(dataObjects, LP_data.m_Data):
				if isinstance(dataObjects.param1, int):
					p1 = '%d' %dataObjects.param1
				elif isinstance(dataObjects.param1, float):
					p1 = '%.2e' %dataObjects.param1
				elif isinstance(dataObjects.param1, str):
					p1 = '%s' %dataObjects.param1
				else:
					p1 = '%s' %pylab.array2string(pylab.array(dataObjects.param1))
				if isinstance(dataObjects.param2, int):
					p2 = '%d' %dataObjects.param2
				elif isinstance(dataObjects.param2, float):
					p2 = '%.2e' %dataObjects.param2
				elif isinstance(dataObjects.param2, str):
					p2 = '%.s' %dataObjects.param2
				else:
					p2 = '%s' %pylab.array2string(pylab.array(dataObjects.param2))
				pD = m_plotData(dataObjects.x, dataObjects.y, [dataObjects.x_axis, dataObjects.y_axis], plot =  self.plot, param = list((p1, p2)))
				self.pData.append(pD)

			elif dataObjects is not None:
				self.pData.append(dataObjects)

			self.ccycle = self.rainb.rainbow_colors(len(self.pData)+2)

			pylab.ioff()

		def toggle_figsize(self):
			if self.ax1.get_figure().get_size_inches()[0] < 4:
				rc('figure',figsize=(2.96,1.85))
				self.ax1.set_size_inches(2.96,1.85)
			else:
				rc('figure',figsize=(4.5,2.815))
				self.ax1.set_size_inches(4.5,2.815)

		def set_colors(self,c='all'):
			if c == 'all':
				self.ccycle.reset()
				for pD in self.pData:
					pD.color = self.ccycle.next()
			if isinstance(c, list) or isinstance(c, tuple):
				ccycle = self.rainb.rainbow_colors(c[1] - c[0]+2)
				for i in xrange(c[0],c[1]+1,1):
					self.pData[i].color = ccycle.next()
					if self.ax1 is not None:
						self.pData[i].line.set_color(self.pData[i].color)

		def reset_colors(self):
			for pD in self.pData:
				pD.color = '#ffffff'

			for pD in self.pData:
				pD.color = '#ffffff'

		def set_style(self, c, color=None, linestyle = None, linesize = None, marker = None, markersize = None):
			if c < len(self.pData) or c == 'all':
				if color is not None:
					self.pData[c].color = color
					self.lines[c].set_color(color)
				if c <> 'all':
					if linestyle is not None:
						self.pData[c].style = linestyle
						if self.ax1 is not None:
							self.pData[c].line.set_linestyle(linestyle)
					if linesize is not None:
						self.pData[c].linesize = linesize
						if self.ax1 is not None:
							self.pData[c].line.set_linewidth(linesize)
					if marker is not None:
						self.pData[c].marker = marker
						if self.ax1 is not None:
							self.pData[c].line.set_marker(marker)
					if markersize is not None:
						self.pData[c].markersize = markersize
						if self.ax1 is not None:
							self.pData[c].line.set_markersize(markersize)
				else:
					for i in xrange(len(self.lines)):
						if linestyle is not None:
							self.pData[i].style = linestyle
							if self.ax1 is not None:
								self.pData[i].line.set_linestyle(linestyle)
						if linesize is not None:
							self.pData[i].linesize = linesize
							if self.ax1 is not None:
								self.pData[i].line.set_linewidth(linesize)
						if marker is not None:
							self.pData[i].marker = marker
							if self.ax1 is not None:
								self.pData[i].line.set_marker(marker)
						if markersize is not None:
							self.pData[i].markersize = markersize
							if self.ax1 is not None:
								self.pData[i].line.set_markersize(markersize)
			elif isinstance(c,list):
					for i in xrange(c[0],c[1]+1,1):
						if linestyle is not None:
							self.pData[i].style = linestyle
							if self.ax1 is not None:
								self.pData[i].line.set_linestyle(linestyle)
						if linesize is not None:
							self.pData[i].linesize = linesize
							if self.ax1 is not None:
								self.pData[i].line.set_linewidth(linesize)
						if marker is not None:
							self.pData[i].marker = marker
							if self.ax1 is not None:
								self.pData[i].line.set_marker(marker)
						if markersize is not None:
							self.pData[i].markersize = markersize
							if self.ax1 is not None:
								self.pData[i].line.set_markersize(markersize)
			else:
				print 'So viele Linien gibt es nicht!'

		def set_range(self,xmin=None,xmax=None,ymin=None,ymax=None):
			if xmin is None:
				xmin = self.ax1.get_xlim()[0]
			if xmax is None:
				xmax = self.ax1.get_xlim()[1]
			if ymin is None:
				ymin = self.ax1.get_ylim()[0]
			if ymax is None:
				ymax = self.ax1.get_ylim()[1]
			self.ax1.set_xlim(xmin,xmax)
			self.ax1.set_ylim(ymin,ymax)
			self.x1[0] = xmin
			self.x1[1] = xmax
			self.y1[0] = ymin
			self.y1[1] = ymax

		def resize_fonts(self,which,s):
			if self.usetex:
				pylab.rc('text', usetex=True)
			else:
				pylab.rc('text', usetex=False)
			from matplotlib.font_manager import fontManager, FontProperties
			font = FontProperties(size=s)
			if which == 'legend':
				self.set_legend(font=font)
			if which == 'label':
				self.set_label(font=font)
			if which == 'texts':
				for t in self.ax1.texts:
					t.set_fontproperties(font)

		def set_legend(self, muster = None, order = None, font = None, loc = 0):
			tmp = []
			if order is not None:
				self.order = order
			if muster is not None:
				self.muster = muster
			for pD in self.pData:
				if pD.set_legend and pD.do_plot:
					if pD.param[self.order[0]] is not None:
						if pD.param[self.order[1]] is not None and self.muster.count('%s') == 2:
							if pD.param[self.order[0]].count('e') > 0:
								if string.atoi(string.split(pD.param[self.order[0]],'e')[1]) <> 0:
									p1 = '%s $\cdot$ 10$^{%s}$' %(string.split(pD.param[self.order[0]],'e')[0], string.split(pD.param[self.order[0]],'e')[1])
								else:
									p1 = '%s' %(string.split(pD.param[self.order[0]],'e')[0])
							else:
								if pD.param[self.order[0]].count('.') > 0:
									p1 = '%s.%s' %(string.split(pD.param[self.order[0]],'.')[0], string.split(pD.param[self.order[0]],'.')[1][:1])
								else:
									p1 = '%s' %(pD.param[self.order[0]])

							if pD.param[self.order[1]].count('e') > 0:
								if string.atoi(string.split(pD.param[self.order[1]],'e')[1]) <> 0:
									p2 = '%s $\cdot$ 10$^{%s}$' %(string.split(pD.param[self.order[1]],'e')[0], string.split(pD.param[self.order[1]],'e')[1])
								else:
									p2 = '%s' %(string.split(pD.param[self.order[1]],'e')[0])
							else:
								if pD.param[self.order[1]].count('.') > 0:
									p2 = '%s.%s' %(string.split(pD.param[self.order[1]],'.')[0], string.split(pD.param[self.order[1]],'.')[1][:1])
								else:
									p2 = '%s' %(pD.param[self.order[1]])
							tmp.append(r'%s' %self.muster %(p1, p2))
						else:
							if pD.param[self.order[0]].count('e') > 0:
								if string.atoi(string.split(pD.param[self.order[0]],'e')[1]) <> 0:
									p1 = '%s $\cdot$ 10$^{%s}$' %(string.split(pD.param[self.order[0]],'e')[0], string.split(pD.param[self.order[0]],'e')[1])
								else:
									p1 = '%s' %(string.split(pD.param[self.order[0]],'e')[0])
							else:
								if pD.param[self.order[0]].count('.') > 0:
									p1 = '%s.%s' %(string.split(pD.param[self.order[0]],'.')[0], string.split(pD.param[self.order[0]],'.')[1][:3])
								else:
									p1 = '%s' %(pD.param[self.order[0]])

							tmp.append(r'%s' %self.muster %(p1))

			self.ltexts = tmp
			if self.legend is not None and self.ax1 is not None:
				self.legend = self.ax1.legend(self.lines, tuple(self.ltexts),loc=loc,prop=font)
			elif self.ax1 is not None:
				self.legend = self.ax1.legend(self.lines, tuple(self.ltexts),loc=loc,prop=font)

		def set_text(self, text, x, y, font = None):
			if font is None:
				from matplotlib.font_manager import fontManager, FontProperties
				font = FontProperties(size = 'smaller')
			self.texts.append(text)
			self.ax1.text(x, y, text, transform = self.ax1.transAxes, fontproperties = font)

		def update_legend(self, font = None):
			self.legend = self.ax1.legend(self.lines, tuple(self.ltexts),loc=0,prop=font)


		def toggle_legend(self):
			if self.ax1 is not None:
				if self.legend.get_visible():
					self.legend.set_visible(False)
					self.ax1.get_legend().set_visible(False)
					self.show()
				else:
					self.legend.set_visible(True)
					self.ax1.get_legend().set_visible(True)
					self.show()
			else:
				print 'Kein offener Plot!'

		def print_legend(self):
			for i in xrange(len(self.ltexts)):
				print '%d: %s' %(i,self.ltexts[i])

		def hide_from_legend(self, c):
			if isinstance(c,list):
			   for i in reversed(xrange(c[0],c[1]+1,1)):
					try:
						if len(self.ltexts) >= i:
							self.ltexts.pop(i)
						if len(self.lines) >= i:
							self.lines.pop(i)
						self.pData[i].set_legend = False
					except:
						print 'Liniennummer "%d" is zu hoch' %i
			else:
				try:
					if len(self.ltexts) >= c:
						self.ltexts.pop(c)
					if len(self.lines) >= c:
						self.lines.pop(c)
					self.pData[c].set_legend = False
				except:
					print 'Liniennummer "%d" is zu hoch' %c
			self.set_legend()
			self.show()

		def unhide_from_legend(self, c):
			if isinstance(c,list):
			   for i in reversed(xrange(c[0],c[1]+1,1)):
				try:
					self.lines.insert(c,self.pData[i].line)
					self.pData[i].set_legend = True
				except:
					print 'Liniennummer "%d" is zu hoch' %i
			else:
				try:
					self.lines.insert(c,self.pData[c].line)
					self.pData[c].set_legend = True
				except:
					print 'Liniennummer "%d" is zu hoch' %c
			self.set_legend()
			self.show()


		def hide_from_plot(self, c):
			if isinstance(c,list):
			   for i in reversed(xrange(c[0],c[1]+1,1)):
					try:
						if self.pData[i].do_plot and self.pData[c].set_legend:
							self.hide_from_legend(i)
						self.pData[i].line.remove()
						self.pData[i].do_plot = False
					except:
						print 'Liniennummer "%d" is zu hoch' %i
			else:
				try:
					if self.pData[c].do_plot and self.pData[c].set_legend:
						self.hide_from_legend(c)
					self.pData[c].line.remove()
					self.pData[c].do_plot = False
				except:
					print 'Liniennummer "%d" is zu hoch' %c
			self.show()

		def delete_from_plot(self, c):
			if isinstance(c,list):
			   for i in reversed(xrange(c[0],c[1]+1,1)):
					try:
						if self.pData[i].do_plot and self.pData[c].set_legend:
							self.hide_from_legend(i)
						self.pData[i].line.remove()
						self.pData.pop(i)
					except:
						print 'Liniennummer "%d" is zu hoch' %i
			else:
				try:
					if self.pData[c].do_plot and self.pData[c].set_legend:
						self.hide_from_legend(c)
					self.pData[c].line.remove()
					self.pData.pop(c)
				except:
					print 'Liniennummer "%d" is zu hoch' %c
			self.show()

		def unhide_from_plot(self, c):
			if isinstance(c,list):
			   for i in reversed(xrange(c[0],c[1]+1)):
					try:
						self.pData[i].do_plot = True
						self.unhide_from_legend(i)
					except:
						print 'Liniennummer "%d" is zu hoch' %c
			else:
				try:
					self.pData[c].do_plot = True
					self.unhide_from_legend(c)
				except:
					print 'Liniennummer "%d" is zu hoch' %c
			self.plot_data()
			self.show()

		def save_fig(self, name, fmt = None):
			if self.usetex:
				pylab.rc('text', usetex=True)
			else:
				pylab.rc('text', usetex=False)
			fig = self.ax1.get_figure()
			if fmt is not None:
				fig.savefig('%s.%s' %(name,fmt))
			else:
				fig.savefig('%s.png' %name)

		def show(self):
			if self.usetex:
				pylab.rc('text', usetex=True)
			else:
				pylab.rc('text', usetex=False)
			pylab.draw()
			pylab.show()

		def reset(self):
			self.ax1 = None
			self.ax2 = None
			self.legend = None
			self.lines = None
			self.lines = []
			self.ltexts = None
			self.ltexts = []
			for pD in self.pData:
				pD.clear()

		def reshow(self):
			self.ax1 = None
			self.ax2 = None
			self.ltexts = None
			self.ltexts = []
			self.lines = None
			self.lines = []
			self.plot_data()
			self.show()


		def show_label(self, font = None):
			if not self.xlabel == '':
				self.ax1.set_xlabel(self.xlabel,fontproperties=font)
			if not self.ylabel1 == '':
				self.ax1.set_ylabel(self.ylabel1,fontproperties=font)
			if not self.title == '':
				self.ax1.set_title(self.title,fontproperties=font)
			if not self.ylabel2 == '':
				self.ax2.set_ylabel(self.ylabel2,fontproperties=font)

		def set_label(self, label = None, can='x'):
			if isinstance(label,list):
				self.xlabel = label[0]
				if len(label) == 2:
					self.ylabel1 = label[1]
				if len(label) == 3:
					self.ylabel2 = label[2]
			elif label is not None:
				if can in ['x', 'y', 'y1', 'y2']:
					if can == 'x':
						self.xlabel = label
					elif can == 'y' or can =='y1':
						self.ylabel1 = label
					elif can == 'y2':
						self.ylabel2 = label
			if self.ax1 is not None and self.xlabel is not None:
				self.ax1.set_xlabel(self.xlabel)
			if self.ax1 is not None and self.ylabel1 is not None:
				self.ax1.set_ylabel(self.ylabel1)
			if self.ax2 is not None and self.ylabel2 is not None:
				self.ax2.set_xlabel(self.ylabel2)

		def set_fnc(self, fnc, can='x'):
			if isinstance(fnc,list):
				for i in xrange(len(fnc)):
					self.fncs[i] = fnc
			else:
				if can in ['x', 'y', 'param1', 'param2']:
					if can == 'x':
						self.fncs[0] = fnc
					elif can == 'y':
						self.fncs[1] = fnc
					elif can == 'param1':
						self.fncs[2] = fnc
					elif can == 'param2':
						self.fncs[3] = fnc
				else:
					print 'Kein Kandidat mit Bezeichnung "%s" gefunden. Kandidaten sind "x", "y", "param1", "param2".' %can


		def remove(self, c):
			if c < len(self.pData):
				try:
					self.pData[c].line.remove()
				except: print ''
				try:
					self.pData.pop(c)
					if self.legend is not None:
						self.remove_from_legend(c)
				except:
					print ''

		def append(self, data,arg=None,value=None):
			if isinstance(data, LP_data.m_dataHandler):
				for dO in data.dataObjects:
					if isinstance(dO.param1, int):
						p1 = '%d' %dO.param1
					elif isinstance(dO.param1, float):
						p1 = '%.2e' %dO.param1
					elif isinstance(dO.param1, str):
						p1 = '%s' %dO.param1
					else:
						p1 = '%s' %pylab.array2string(pylab.array(dO.param1))
					if isinstance(dO.param2, int):
						p2 = '%d' %dO.param2
					elif isinstance(dO.param2, float):
						p2 = '%.2e' %dO.param2
					elif isinstance(dO.param2, str):
						p2 = '%.s' %dO.param2
					else:
						p2 = '%s' %pylab.array2string(pylab.array(dO.param2))
					pD = m_plotData(dO.x, dO.y, [dO.x_axis, dO.y_axis], plot =  self.plot, param = list((p1, p2)))
					if arg is not None:
						if not isinstance(arg, list) and not isinstance(arg, tuple):
							arglist = (arg,)
							value = (value,)
						for i in xrange(len(arg)):
							pD.set(arg[i],value[i])
					self.pData.append(pD)


			elif isinstance(data, LP_data.m_Data):

				if isinstance(data.param1, int):
					p1 = '%d' %data.param1
				elif isinstance(data.param1, float):
					p1 = '%.2e' %data.param1
				elif isinstance(data.param1, str):
					p1 = '%s' %data.param1
				else:
					p1 = '%s' %pylab.array2string(pylab.array(data.param1))
				if isinstance(data.param2, int):
					p2 = '%d' %data.param2
				elif isinstance(data.param2, float):
					p2 = '%.2e' %data.param2
				elif isinstance(data.param2, str):
					p2 = '%.s' %data.param2
				else:
					p2 = '%s' %pylab.array2string(pylab.array(data.param2))
				pD = m_plotData(data.x, data.y, [data.x_axis, data.y_axis], plot =  self.plot, param = list((p1, p2)))
				self.pData.append(pD)

			else:
				if isinstance(data.param1, int):
					p1 = '%d' %data.param1
				elif isinstance(data.param1, float):
					p1 = '%.2e' %data.param1
				elif isinstance(data.param1, str):
					p1 = '%s' %data.param1
				else:
					p1 = '%s' %pylab.array2string(pylab.array(data.param1))
				if isinstance(data.param2, int):
					p2 = '%d' %data.param2
				elif isinstance(data.param2, float):
					p2 = '%.2e' %data.param2
				elif isinstance(data.param2, str):
					p2 = '%.s' %data.param2
				else:
					p2 = '%s' %pylab.array2string(pylab.array(data.param2))
				pD = m_plotData(data.x, data.y, [data.x_axis, data.y_axis], plot =  self.plot, param = list((p1, p2)))
				self.pData.append(pD)

			self.ccycle = self.rainb.rainbow_colors(len(self.pData)+2)

		def print_pData(self):
			i = 0
			for pD in self.pData:
				print '%d: Color = %s,\tLinestyle = %s\tMarkersytle = %s,\tParam1 = %s,\tParam2 = %s' %(i, pD.color, pD.style,pD.marker,pD.param[0],pD.param[1])
				i += 1

		#~ def plot_new_ones(self):
			#~ p = []
			#~ for pO in self.pData:
				#~ if not pO.plotted and pO.do_plot:
					#~ p.append(pO)
			#~ self.plot_data(pOlist = p)

		def is_plotted(self):
			try:
				self.ax1.get_figure().show()
				return True
			except:
				return False


		def plot_data(self,c = None, pOlist = None, trange = None):
			if not self.is_plotted():
				fig = pylab.figure()
				fig.clf()
				self.ax1 = fig.add_subplot(111)
			else:
				fig = self.ax1.get_figure()
				while len(self.ax1.get_lines()):
					 self.ax1.get_lines()[-1].remove()
				if self.ax2 is not None:
					while len(self.ax2.get_lines()):
						 self.ax2.get_lines()[-1].remove()


			if self.fncs[0] is not None:
				xfnc = self.fncs[0]
			else:
				xfnc = lambda x: x

			if self.fncs[0] is None:
				self.fncs[0] = lambda x: x

			if self.fncs[1] is None:
				self.fncs[1] = lambda x: x

			if self.fncs[3] is None:
				self.fncs[2] = lambda x: x

			if self.fncs[3] is None:
				self.fncs[3] = lambda x: x

			if c is None:
				p = xrange(len(self.pData))
				offset = 0
			elif isinstance(c,list):
				p = xrange(c[0],c[-1]+1,1)
				offset = 0
			else:
				p = xrange(1)
				offset = c

			if pOlist is None:
				pOlist = []
				for i in p:
					pOlist.append(self.pData[i+offset])

			for i in xrange(len(pOlist)):

				if trange is not None:
					r1 = getindex_hi_sort(pOlist[i].x,trange[0],0,len(pOlist[i].x)-1)
					r2 = getindex_hi_sort(pOlist[i].x,trange[1],0,len(pOlist[i].x)-1)
				else:
					r1 = 0
					r2 = len(pOlist[i].x)

				self.set_colors()
				color_ = pOlist[i].color

				if pOlist[i].style <> '-':
					linestyle_ = pOlist[i].style
				else:
					linestyle_ = '-'

				if pOlist[i].marker <> '':
					markerstyle_ = pOlist[i].marker
				else:
					markerstyle_ = ''


				if pOlist[i].axis == 1:
					self.ax1.grid(True)
					pOlist[i].ax = self.ax1


					if pOlist[i].plot == 'semilogx':
						if pOlist[i].do_plot:
							pOlist[i].line = self.ax1.semilogx(self.fncs[0](pOlist[i].x[r1:r2]), self.fncs[1](pOlist[i].y[r1:r2]), color=color_,linestyle=linestyle_, marker=markerstyle_)[0]


					elif pOlist[i].plot == 'semilogy':
						if pOlist[i].do_plot:
							pOlist[i].line = self.ax1.semilogy(self.fncs[0](pOlist[i].x[r1:r2]), self.fncs[1](pOlist[i].y[r1:r2]), color=color_,linestyle=linestyle_, marker=markerstyle_)[0]


					elif pOlist[i].plot == 'loglog':
						if pOlist[i].do_plot:
							pOlist[i].line = self.ax1.loglog(self.fncs[0](pOlist[i].x[r1:r2]), self.fncs[1](pOlist[i].y[r1:r2]), color=color_,linestyle=linestyle_, marker=markerstyle_)[0]


					elif pOlist[i].plot == 'vline':
						if pOlist[i].do_plot:
							pOlist[i].line = self.ax1.axvline(x=pOlist[i].x, linewidth=1.5, color=color_,linestyle=linestyle_, marker=markerstyle_)


					else:
						if pOlist[i].do_plot:
							pOlist[i].line = self.ax1.plot(self.fncs[0](pOlist[i].x[r1:r2]), self.fncs[1](pOlist[i].y[r1:r2]), color=color_,linestyle=linestyle_, marker=markerstyle_)[0]


				elif pOlist[i].axis == 2:
					self.ax2 = pylab.twinx()
					self.ax2.yaxis.tick_right()
					self.ax2.grid(False)
					pOlist[i].ax = self.ax2

					if pOlist[i].plot == 'semilogx':
						if pOlist[i].do_plot:
							pOlist[i].line = self.ax2.semilogx(self.fncs[0](pOlist[i].x[r1:r2]), self.fncs[1](pOlist[i].y[r1:r2]), color=color_,linestyle=linestyle_, marker=markerstyle_)[0]


					elif pOlist[i].plot == 'semilogy':
						if pOlist[i].do_plot:
							pOlist[i].line = self.ax2.semilogy(self.fncs[0](pOlist[i].x[r1:r2]), self.fncs[1](pOlist[i].y[r1:r2]), color=color_,linestyle=linestyle_, marker=markerstyle_)[0]


					elif pOlist[i].plot == 'loglog':
						if pOlist[i].do_plot:
							pOlist[i].line = self.ax2.loglog(self.fncs[0](pOlist[i].x[r1:r2]), self.fncs[1](pOlist[i].y[r1:r2]), color=color_,linestyle=linestyle_, marker=markerstyle_)[0]


					else:
						if pOlist[i].do_plot:
							pOlist[i].line = self.ax2.plot(self.fncs[0](pOlist[i].x[r1:r2]), self.fncs[1](pOlist[i].y[r1:r2]), color=color_,linestyle=linestyle_, marker=markerstyle_)[0]

				pOlist[i].plotted = True

			fig.subplots_adjust(bottom=0.17, left=0.165, top=0.94, right=0.94)
			if self.pData[0].plot <> 'semilogy' and self.pData[0].plot <> 'loglog':
				from matplotlib.ticker import MultipleLocator
				a_y = self.ax1.get_yaxis()
				ymax = abs(self.ax1.get_ylim()[1] - self.ax1.get_ylim()[0])
				stepsize = ymax/len(a_y.get_major_ticks())
				stepsize = stepsize/5.
				a_y.set_minor_locator(MultipleLocator(stepsize))

			self.lines = None
			self.lines = []
			for pD in self.pData:
				if pD.set_legend and pD.do_plot and pD.param is not None:
					self.lines.append(pD.line)

			if self.ltexts != [] and self.ltexts is not None:
				self.set_legend()

			self.set_label()



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


class m_plotData(LP_data.m_Data):
		def __init__(self, x = None, y = None, axes = None, color = '#ffffff', style = '-', marker = '', param = None, plot = 'plot', axis = 1, ax = None, line = None, derived_class=LP_data.m_Data):
			derived_class.__init__(self, x, y, axes)

			self.param = param
			self.color = color
			self.style = style
			self.linesize = 2
			self.marker = marker
			self.markersize = 8
			self.plot = plot
			self.axis = axis
			self.ax = ax
			self.line = line
			self.do_plot = True
			self.set_legend = True
			self.plotted = False

		def append(self, x, y, axes, color = None, style = None, marker = None, param = None, plot = 'plot', axis = 1, line = None):
			self.add('x',x)
			self.add('y',y)
			self.add('axes', axes)
			self.add('color',color)
			self.add('style',style)
			self.add('marker',marker)
			self.add('text',text)
			self.add('plot',plot)
			self.add('axis',axis)

		def add(self, name, value):
			tmp = []
			tmp.append(self.__getattribute__(str(name)))
			tmp.append(value)
			self.__setattr__(str(name),tmp)

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

		def clear(self):
			self.color = '#ffffff'
			self.style = '-'
			self.linesize = 2
			self.marker = ''
			self.markersize = 5
			self.axis = 1
			self.ax = None
			self.line = None
			self.do_plot = True
			self.set_legend = True
			self.plotted = False

		def __call__(self):
				return  pylab.array(self.x, self.y)
