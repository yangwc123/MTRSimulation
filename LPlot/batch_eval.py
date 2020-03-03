import numpy, LPlot, os

p1 = None
p1 = LPlot.m_Project('double_transients','')
p1.load_files('data')
p1.fileHandler.sort('F')
p1.fileHandler.sort('AHOMO')
p1.fileHandler.sort('Dmol_L')
p1.fileHandler.sort('ratio')


def plot_config(pnames,pvalues,pparams, anames, aparams):
	try:
		fh2 = LPlot.m_fileHandler(p1.fileHandler.get_fileObjects(pnames,pvalues))
		if len(fh2.fileObjects) != 0:
			fh2.load_files_data()
			dh2 = fh2.get_dataSlice(aparams[0],aparams[1],[pparams[0],pparams[1]])
			for d in dh2.dataObject:
				d.y = LPlot.smooth(d.y,15)
			po2 = LPlot.m_Plot(dh2, 'semilogx')
			#po2.set_fnc(LPlot.smooth_data,'x',2)
			po2.plot_data()
			#po2.set_style(c='all',linestyle = '', marker = 'o',markersize=4)
			po2.muster = '%s = %s' %(pparams[0],'%s')
			po2.set_text(r'%s = %.2g' %(pnames[0],pvalues[0]),0.65,0.7)
			po2.set_text(r'%s = %.2g' %(pnames[1],pvalues[1]),0.65,0.8)
			po2.resize_fonts('texts',18)
			#po2.set_text(r'L = 25 meV',0.6,0.3)
			po2.ax1.set_xlabel(anames[0])
			po2.ax1.set_ylabel(anames[1])
			#po2.set_range(1e-7,2e-2,2.7e-4,5e2)ratio
			fig = po2.ax1.get_figure()
			fig.subplots_adjust(0.125,0.14)
			po2.set_legend(loc=3)
			po2.show()
			#po2.save_fig('r_%.2f-dE_%.2f-l_%d_+_F_jvst' %(r,dE,l),'svg')
		else:
			try:
				print 'No Data for %s: %.2f, %s: %.2f, %s: %d' %(pnames[0],pvalues[0],pnames[1],pvalues[1],pnames[2], pvalues[2])
			except:
				print 'No Data for %s: %.2f, %s: %.2f' %(pnames[0],pvalues[0],pnames[1],pvalues[1])
	except:
		try:
			print 'No Data for %s: %.2f, %s: %.2f, %s: %d' %(pnames[0],pvalues[0],pnames[1],pvalues[1],pnames[2], pvalues[2])
		except:
			print 'No Data for %s: %.2f, %s: %.2f' %(pnames[0],pvalues[0],pnames[1],pvalues[1])



ratio = p1.fileHandler.get_param_list('ratio')
length = p1.fileHandler.get_param_list('Dmol_L')
deltaE = p1.fileHandler.get_param_list('AHOMO')
field = p1.fileHandler.get_param_list('F')
clustersize = p1.fileHandler.get_param_list('clustersize')

## Param is clustersize
for l in length:
	for r in ratio:
			plot_config(['Dmol_L', 'ratio', 'AHOMO','carriersort'], [l, r, -0.7, 'h'], ['F', 'T'], [r't [s]', r'j [$A/m^2$]'], ['t', 'j2'])


