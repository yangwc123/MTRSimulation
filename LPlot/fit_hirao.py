############################
#
# Fit a Set of Data
#
############################
def fit_linlin(t_transit, t, j, step):

	index_t12 = getindex_low_sort(t,t_transit,0,len(t)-1)
	try:
		nt = pylab.arange(t[0],t[-1],1e-7)
		dj = derive(t, j, step)
		
		n = getindex_low_sort(t,t_transit/10.,0,len(t)-1)
		n1 = getindex_low_sort(t,t_transit/5.,0,len(t)-1)
		
		index_vor = getindex_hi_sort(dj,0,n,n)
		index_vor = index_vor+(dj[index_vor:index_t12]).argmax()
		index_nach = index_vor+(dj[index_vor:index_t12]).argmin()
		print index_vor, index_nach
		tangente_vor = min( (dj[:index_vor+15])[index_vor], 0)
		tangente_nach = (dj[index_nach-15:])[15]
		
		n_index_nach = getindex_low_sort(nt,t[index_nach], 0, len(nt)-1)
		
		fit_vor = (tangente_vor*(nt-t[index_vor]) + j[index_vor])
		fit_nach = (tangente_nach*(nt-t[index_nach]) + j[index_nach])
		
		t_cal_0 = -1
		for i in xrange(len(fit_nach)):
			t_cal_0 += 1
			if fit_nach[i] <= fit_vor[i]:
				break
		
		fit_vor = (tangente_vor*(t-t[index_vor]) + j[index_vor])
		fit_nach = (tangente_nach*(t-t[index_nach]) + j[index_nach])
			
		return  (fit_vor, fit_nach, nt[t_cal_0], t_cal_0, n_index_nach)
	except:
		print 'Fitting lin-lin impossible ...'
		return  ([0], [0], t[index_t12])
	
	
def fit_loglog(t_transit, t, j,step):
	
	index_t12 = getindex_low_sort(t,t_transit,0,len(t)-1)
	try:
		nt = pylab.arange(t[0],t[-1],1e-7)
		logt = pylab.log(t)
		logj = pylab.log(j)
		dj = derive(logt, logj, step)
		
		n = getindex_low_sort(t,t_transit/10.,0,len(t)-1)
		index_vor = getindex_hi_sort(dj,0,n,index_t12)
		index_vor = index_vor+1+(dj[index_vor+1:index_t12]).argmax()
		
		index_nach = index_vor+(dj[index_vor:index_t12]).argmin()
		print index_vor, index_nach
		tangente_vor = min( (dj[:index_vor+15])[index_vor], 0)
		tangente_nach = (dj[index_nach-15:])[15]
		
		n_index_nach = getindex_low_sort(nt,t[index_nach], 0, len(nt)-1)
				
		fit_vor = pylab.nan_to_num((nt/(t[index_vor]))**tangente_vor * j[index_vor])
		fit_nach = pylab.nan_to_num((nt/(t[index_nach]))**tangente_nach * j[index_nach])
		
		t_cal_0 = -1
		for i in xrange(len(fit_nach)):
			t_cal_0 += 1
			if fit_nach[i] <= fit_vor[i]:
				break
			
		fit_vor = pylab.nan_to_num((t/(t[index_vor]))**tangente_vor * j[index_vor])
		fit_nach = pylab.nan_to_num((t/(t[index_nach]))**tangente_nach * j[index_nach])
		return  (fit_vor, fit_nach, nt[t_cal_0], t_cal_0, n_index_nach)
	except:
		print 'Fitting log-log impossible ...'
		return  ([0], [0], t[index_t12])

