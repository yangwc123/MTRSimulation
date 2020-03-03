import LPlot, pylab

p1 = None
p1 = LPlot.m_Project('Transport-E','Simulation of transport energy without interaction and variing Field')
#~ p1.load_files('/media/StorePlay/diplom/data')
#~ p1.fileHandler.remove_fileObjects(['carriersort'],['e'])
#~ p1.fileHandler.load_files_data()
#p1.fileHandler.load_distribution_data('E')
p1.load('data11.db')
p1.add_Slice(p1.fileHandler.get_fileObjects('F',0))
#~ p1.add_Slice(p1.fileHandler.get_fileObjects('F',0),1)
wd = p1.slices[-1].fileHandler
wd.sort('F')

dh2 = wd.sort_Data('T','t_E','Dsigma','ax')
po2 = LPlot.LPPlot.m_Plot(dh2, 'plot')
po2.hide_legend()
po2.plot_data()
#po2.set_style(c='all',linestyle = '-', marker = 'o',markersize=4)
po2.muster = '\sigma = %s'
#po2.set_legend()

p1.add_Slice(dh2,0)
p1.add_Slice(po2,0)
p1.save('data11.db')
dh2 = wd.sort_Data('T','t_upE','Dsigma','ax')
po2.append(dh2)

dh2 = wd.get_dataSlice('t','E',['F','ax'])
po2 = LPlot.LPPlot.m_Plot(dh2, 'semilogx')
po2.hide_legend()
po2.plot_data()
po2.muster = '\sigma = %s'
po2.set_legend()
po2.plot_data()
po2.set_style(c='all',linestyle = '-', marker = 'o',markersize=4)
po2.muster = 'L = %s nm'
#po2.set_text(r'L = $1\cdot 10^7 \frac{V}{m}$',0.6,0.2)
#po2.set_text(r'L = 25 meV',0.6,0.3)
po2.ax1.set_xlabel(r'1/$T^2$ [1/$K^2$]')
po2.ax1.set_ylabel(r'$mu_{mean}$ [$m^2/Vs$]')
po2.set_legend()

po2.append(dh3,['plot'],['vline'])
po2.plot_data([len(po2.pData)/2,len(po2.pData)])

fH.remove_fileObjects(['carriersort','interaction'],['e',1.])
fH.load_files_data()

dH = fH.sort_Data('T','mu_mean','no_of_carriers','F')

pp = LPlot.m_Plot(dH,plot='semilogy')

p1 = LPlot.m_Project('test','Das ist ein TestProjekt')

p1.add([fH,dH,pp])

pp.set_colors()
pp.plot_data()
pp.show()

p1.save('test.jl')


#reload(LPlot)
#reload(LP_files)
#b = None
#b = LPlot.m_Project('aa','this is aa project')
#b.load_files('/media/StorePlay/diplom/20081126_double_donor_molecule/qq')
#b.fileHandler.load_files_data()
#b.save('aa.h5')
