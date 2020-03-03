#!/usr/bin/python
# -*- coding: iso-8859-1 -*-


def main(argv=None):
	# use TkAgg instead of default GTK because of threading stuff
	if argv is None:
		import matplotlib
		matplotlib.use('TkAgg')


	from optparse import OptionParser
	import LPlot, os
	from numpy import where, array, flipud, linspace, exp, shape, isnan
	from string import split, digits
	import sys
	import scipy.signal

	# Read from the commandline
	usage = "usage: %prog [options] file1 file2 ..."
	parser = OptionParser(usage=usage)
	parser.add_option('-S', '--start', action='store', type='int',
						help="first file no. to plot", dest='start', default=-1)
	parser.add_option('-E', '--end', action='store', type='int',
						help="last file no. to plot", dest='end', default=-1)
	parser.add_option('-b', '--base', action='store', type='string',
						help='constant letters in filename. * will be replaced by filenumber in range between [-s, -e]'
						, dest='base', default='m*-h.dat')
	parser.add_option('-d', '--directory', action='store', type='string',
						help="directory to look for files", metavar="PATH", dest='directory')
	parser.add_option('-a', '--allinone', action='store_true', dest='allinone',
						help="plot a bunch of data in one graph?", default=False)
	parser.add_option('-s', '--Save', action='store_true', dest='save',
						help="save the plots?", default=False)
	parser.add_option('-p', '--plot', action='append', help='''data to be plotted versus each other "y" vs. "x".
														"scale" gives the axes scaling [plot, semilogx, semilogy, loglog]
														"linestyle" defines the style of the plotted lines
														[ - | -- | -. | : ]

														"marker" defines the style of the plotted markers
														[ + | * | , | . | 1 | 2 | 3 | 4
														| < | > | D | H | ^ | _ | d
														| h | o | p | s | v | x | |
														| TICKUP | TICKDOWN | TICKLEFT | TICKRIGHT ]

														"markersize" defines the size of the plotted marker
														"param1" is the constant Parameter for plotting single measurment values,
														e.g. mobility vs. carrier density, param1 could be the temperature
														"param2" see param1. If not given param2 is param1
														''',
						metavar='y:x[:scale:linestyle:marker:markersize:param1[:param2]]',dest='plots')
	parser.add_option('-f', '--filetype', action='store', type='string',
						help="define the type of data (simulation [sim], experiment [exp])", dest='ftype', default='sim')
	parser.add_option('-i', '--include', action='store', type='string',
						help='filters the plotted data by the param and value given. Only data for which "param" has the value "value"'
						,metavar='param:value', dest='inc')
	parser.add_option('-e', '--exclude', action='store', type='string',
						help='filters the plotted data by the param and value given. Only data for which "param" is not equal "value"'
						,metavar='param:value', dest='exc')
	parser.add_option('-l', '--legend', action='store_true',
						help="plot legend?",dest='legend', default=False)



	if len(sys.argv) == 1:
		parser.print_help()
		exit(2)

	if argv is None:
		(options, args) = parser.parse_args()
	else:
		(options, args) = parser.parse_args(argv)

	if options.directory is not None:
		options.directory = os.path.abspath(options.directory)
		os.chdir(options.directory)
	else:
		options.directory = os.path.abspath(os.curdir)

	if options.plots is None:
		options.plots = ['x:t:semilogx', 'N:t:semilogx', 'j1:t:loglog']

	if options.inc is not None:
		options.inc  = split(options.inc,':')

	if options.exc is not None:
		options.exc  = split(options.exc,':')

	# generate list with files to plot
	if len(args) == 0:				# if no files are given on commandline
		files = os.listdir('.')		# generate list of all files in directory (options.directory)
	else:							# else files are given on commandline
		files = (args)

	sp = split(options.base,'*')	# base is the constant part of the filenames
	if sp[0] != '':
		len_sp0 = len(sp[0])
	else:
		len_sp0 = 0
	if sp[1] != '':
		len_sp1 = len(sp[1])
	else:
		len_sp1 = 0

	# remove all files which do not fit the base
	if len_sp0 > 0:
		files = array([where(t_f[:len_sp0] == sp[0],t_f, '') for t_f in files]).tolist()
		while files.count('') > 0:
			files.remove('')
	else:
		files = array([where(t_f[0] in digits,t_f, '') for t_f in files]).tolist()
		while files.count('') > 0:
			files.remove('')

	if len_sp1 > 0:
		files = array([where(t_f[-len_sp1:] == sp[1],t_f, '') for t_f in files]).tolist()
		while files.count('') > 0:
			files.remove('')

	# remove all files which are not in between the counter range (options.start, options.end)
	if len_sp1 > 0:
		if options.start != -1:
			files = array([where(int(t_f[len_sp0:-len_sp1]) >= options.start, t_f, '') for t_f in files]).tolist()
			while files.count('') > 0:
				files.remove('')

		if options.end != -1:
			files = array([where(int(t_f[len_sp0:-len_sp1]) <= options.end, t_f, '') for t_f in files]).tolist()
			while files.count('') > 0:
				files.remove('')

		# sort the files by their file counter number
		files.sort(lambda x,y: int(x[len_sp0:-len_sp1]) - int(y[len_sp0:-len_sp1]))
	else:
		if options.start != -1:
			files = array([where(int(t_f[len_sp0:]) >= options.start, t_f, '') for t_f in files]).tolist()
			while files.count('') > 0:
				files.remove('')

		if options.end != -1:
			files = array([where(int(t_f[len_sp0:]) <= options.end, t_f, '') for t_f in files]).tolist()
			while files.count('') > 0:
				files.remove('')

		# sort the files by their file counter number
		files.sort(lambda x,y: int(x[len_sp0:]) - int(y[len_sp0:]))


	if len(files) > 0:
		# generate list with data to plot
		tmp = []
		for plot in options.plots:
			if len(split(plot,':')) == 2:
				t = split(plot,':')
				t.append('semilogx')
				t.append('-')
				t.append('')
				t.append(5)
				t.append('name')
				t.append('name')
				tmp.append(t)
			else:
				t = split(plot,':')
				if len(t) == 7:
					t.append(t[-1])
				tmp.append(t)

			if len(t) == 7:
				tmp.append(tmp[-1])
			options.plots = tmp


		plots = []
		# add files to a LPlot project and plot stuff
		if options.allinone == True:
			p = LPlot.m_Project('','')									# generate project
			if options.ftype == 'sim':
				p.load_files(options.directory,
					filetype = LPlot.LP_files.m_simFile, filelist=files)	# add the files
			else:
				p.load_files(options.directory,
					filetype = LPlot.LP_files.m_File, filelist=files)		# add the files

			p.fileHandler.load_files_data()								# load files content
			axeslist = p.fileHandler.fileObjects[0].get('d').keys()		# get a list with available axes
			headerlist = p.fileHandler.fileObjects[0].get('h').keys()		# get a list with available axes

			for plot in options.plots:									# run over the different plots
				doplot = False
				print plot
				if (plot[0] in axeslist and plot[1] in axeslist):		# check if the requested data is available
					# extract data from the content and smooth it
					if options.inc is not None:
						dh = p.fileHandler.get_dataSlice(plot[1], plot[0], [plot[6], plot[7]], inc=options.inc[0], value = options.inc[1])
					elif options.exc is not None:
						dh = p.fileHandler.get_dataSlice(plot[1], plot[0], [plot[6], plot[7]], exc=options.exc[0], value = options.exc[1])
					else:
						dh = p.fileHandler.get_dataSlice(plot[1], plot[0], [plot[6], plot[7]])
					for d in dh.dataObject:
						y = scipy.signal.wiener(d.y,35)
						if not any(isnan(y)):
							d.y = LPlot.LP_func.smooth(y,15)
						else:
							d.y = LPlot.LP_func.smooth(d.y,35)
							d.y = scipy.signal.wiener(d.y,35,1)
					doplot = True


				elif (plot[0] in headerlist and plot[1] in headerlist):	# check if the requested data is available
					dh = p.fileHandler.sort_Data(plot[1], plot[0], plot[6], plot[7])
					doplot = True

				if doplot and len(dh.dataObject) != 0:
					po = LPlot.LP_plot.m_Plot(dh, plot[2])			# create a plotobject
					po.plot_data()										# plot the data
					po.set_style('all',linestyle=plot[3],marker=plot[4],markersize=int(plot[5]))
					po.muster = '%s, %s' %('%s', '%s')
					po.ax1.set_xlabel(plot[1])
					po.ax1.set_ylabel(plot[0])
					if options.legend:
						po.set_legend(loc=0)
					fig = po.ax1.get_figure()							# get figure
					fig.subplots_adjust(0.125,0.14)						# and resize it
					if options.save:
						po.save_fig('plot_%s_%s' %(plot[1],plot[0]),'pdf')

					plots.append(po)
		else:
			for file in files:
				p = None
				p = LPlot.m_Project('','')									# generate project
				p.load_files(options.directory, [file,])						# add the files
				p.fileHandler.load_files_data()								# load files content
				axeslist = p.fileHandler.fileObjects[0].get('d').keys()		# get a list with available axes
				for plot in options.plots:									# run over the different plots
					if plot[0] in axeslist and plot[1] in axeslist:			# check if the requested data is available
						# extract data from the content and smooth it
						dh = p.fileHandler.get_dataSlice(plot[1], plot[0],['name'])
						for d in dh.dataObject:
							d.y = LPlot.LPFunc.smooth(d.y,15)

						po = LPlot.LP_plot.m_Plot(dh, plot[2])			# create a plotobject
						po.plot_data()										# plot the data
						po.muster = '%s, %s' %('%s', '%s')
						po.ax1.set_xlabel(plot[1])
						po.ax1.set_ylabel(plot[0])
						if options.legend:
							po.set_legend(loc=3)
						fig = po.ax1.get_figure()							# get figure
						fig.subplots_adjust(0.125,0.14)						# and resize it
						if options.save:
							po.save_fig('plot_%s_%s' %(plot[1],plot[0]),'pdf')
					plots.append(po)
		# at the end show all the plotted stuff
		po.show()
		return plots
#~



if __name__ == '__main__':
	main()


