def transpose(lists):
    if not lists: return []
    return map(lambda *row: list(row), *lists)

def plot2D(ma, xs, ys, kind='imshow', fnc_x=lambda x:x, fnc_y=lambda y:y, fnc_z=lambda z:z , axis_label = None, appto = None, **kwargs):

        import pylab
        from matplotlib.image import NonUniformImage
        from numpy import linspace
        from scipy.interpolate import splrep, splev
        from mpl_toolkits.mplot3d import Axes3D

        if appto is None or kind not in ['imshow', 'contour', 'contourf']:
            f=pylab.figure()
            if kind in ['imshow', 'contour', 'contourf']:
                ax=f.add_subplot(111)
            else:
                ax = f.gca(projection='3d')
        else:
            ax = appto

        CS=None


        try:
            pylab.array(xs)
            x_min = max(transpose(xs)[0]); x_max = min(transpose(xs)[-1])
            x_axis = linspace(x_min, x_max,400)
            for i in xrange(len(xs)):
                ma[i] = splev(x_axis, splrep(xs[i], ma[i], k=3))
                xs[i] = x_axis
        except:
            x_axis = fnc_x(xs)

        y_axis = fnc_y(ys)

        if kind == 'imshow':
            im = NonUniformImage(ax, interpolation='bilinear' )
            im.set_cmap(kwargs.get('cmap',None))
            im.set_data( x_axis, y_axis, fnc_z(ma) )
            if kwargs.has_key('vmin'):
                im.set_clim(vmin=kwargs['vmin'])
            if kwargs.has_key('vmax'):
                im.set_clim(vmax=kwargs['vmax'])
            ax.images.append( im )
            #~ xlabel( r'Wavelength [nm]' )
            #~ ylabel( r'Delay [ps]' )
            pylab.show()
            if kwargs.has_key('bar'):
                bar = kwargs['bar']
                kwargs.pop('bar')
            else:
                bar = True
            if bar:
                ax.get_figure().colorbar(im)
            ax.set_xlim(x_axis[0],x_axis[-1])
            ax.set_ylim(y_axis[0],y_axis[-1])
            pylab.draw()
        elif kind == 'contour':
            N=kwargs.get('N', 8)
            X, Y = pylab.meshgrid(x_axis, y_axis)
            CS=ax.contour(X, Y, fnc_z(ma), N, **kwargs)
            if kwargs.has_key('labels'):
                labels = kwargs['labels']
                kwargs.pop('labels')
                fmt = {}
                for l, s in zip( CS.levels, labels ):
                    fmt[l] = s
            elif kwargs.has_key('fmt'):
                fmt = kwargs('fmt')
            else:
                fmt = '%1.2f'
            if kwargs.has_key('fontsize'):
                fontsize = kwargs['fontsize']
            else:
                fontsize = 12
            ax.clabel(CS, CS.levels, inline=1, fmt = fmt, fontsize = fontsize)
            pylab.show()
            ax.set_xlim(x_axis[0],x_axis[-1])
            ax.set_ylim(y_axis[0],y_axis[-1])
            pylab.draw()
        elif kind == 'contourf':
            N=kwargs.get('N', 8)
            X, Y = pylab.meshgrid(x_axis, y_axis)
            CS=ax.contourf(X, Y, fnc_z(ma), N, **kwargs)
            ax.get_figure().colorbar(CS)
            pylab.show()
            ax.set_xlim(x_axis[0],x_axis[-1])
            ax.set_ylim(y_axis[0],y_axis[-1])
            pylab.draw()
        elif kind == 'surf':
            X, Y = pylab.meshgrid(x_axis, y_axis)
            CS=ax.plot_surface(X, Y, fnc_z(pylab.array(ma)), **kwargs)
            #ax.get_figure().colorbar(CS, shrink=0.5, aspect=5)
            pylab.show()
            ax.set_xlim(x_axis[0],x_axis[-1])
            ax.set_ylim(y_axis[0],y_axis[-1])
            pylab.draw()
        elif kind == 'contour3d':
            N=kwargs.get('N', 8)
            X, Y = pylab.meshgrid(x_axis, y_axis)
            CS=ax.contourf(X, Y, fnc_z(ma), N, **kwargs)
            if kwargs.has_key('labels'):
                labels = kwargs['labels']
                kwargs.pop('labels')
                fmt = {}
                for l, s in zip( CS.levels, labels ):
                    fmt[l] = s
            elif kwargs.has_key('fmt'):
                fmt = kwargs('fmt')
            else:
                fmt = '%1.2f'
            if kwargs.has_key('fontsize'):
                fontsize = kwargs['fontsize']
            else:
                fontsize = 12
            ax.clabel(CS, CS.levels, inline=1, fmt = fmt, fontsize = fontsize)
            pylab.show()
            ax.set_xlim(x_axis[0],x_axis[-1])
            ax.set_ylim(y_axis[0],y_axis[-1])
            pylab.draw()

        if axis_label is not None:
            ax.set_xlabel(axis_label[0])
            ax.set_ylabel(axis_labe[1])
        pylab.show()

        return ax, CS

def test():
    import numpy as np
    import pylab
    from matplotlib import cm
    X = np.linspace(0,2,200)*np.pi
    Y = np.linspace(2,2.5,100)*np.pi
    Z = [[np.sin(Xi)+np.cos(Yi**2) for Xi in X] for Yi in Y]

    ax,CS=plot2D(Z,X,Y,'surf',cmap=cm.jet,
                              rstride=1, cstride=1, linewidth=0,
                              antialiased=True)
    #ax,CS=plot2D(Z,X,Y,'contour',appto=ax, colors='w', levels=[-1.75,-1,0,1,1.75])
    # So kannst du die labels der Konturlinien veraendern
    # pylab.clabel(CS, levels[1::2],  # label every second level
    #       inline=1,
    #       fmt='%1.1f',
    #       fontsize=14)


