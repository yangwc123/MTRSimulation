############################################
#
#   This class delivers a nice list of
#   rainbow colors for plots
#
############################################

from numpy import linspace, array, mod
from grapefruit import Color
from pylab import setp, gca, draw
from matplotlib.axes import Subplot

class rainbow_colors():
    class it():
        def __init__(self, size, c ,kind):
            self.c = c
            self.size = size
            self.n_iter = -1
            self.kind = kind

        def next(self):
            self.n_iter += 1
            return self.c[self.kind][self.n_iter % self.size]

        def last(self):
            self.n_iter -= 1
            return self.c[self.kind][self.n_iter % self.size]

        def reset(self):
            self.n_iter = -1
            return self.c[self.kind][self.n_iter % self.size]

    def __init__(self, n, start=0,end=270):
        self.size = n
        h_all = linspace(start,end,n)
        v_all = [0.75, 0.80, 0.85, 0.9, 0.95, 1]
        s_all = [.8]
        self.c = {}
        hsv = []
        rgb = []
        for i in xrange(n):
            s = s_all[0]
            v = v_all[5-mod(i,3)]
            hsv.append([h_all[i]+240,s,v])
            rgb.append(Color.HsvToRgb(h_all[i],s,v))
        self.c['rgb'] = array(rgb)
        self.c['hsv'] = array(hsv)
        self.iterator = {}
        self.iterator['rgb'] = self.it(n, self.c, 'rgb')
        self.iterator['hsv'] = self.it(n, self.c, 'hsv')

    def iter(self, kind = 'rgb'):
        tmp = self.it(self.size, self.c, kind)
        return tmp

    def next(self, kind = 'rgb'):
        return self.iterator[kind].next()

    def last(self, kind = 'rgb'):
        return self.iterator[kind].last()

    def reset(self, kind = 'rgb'):
        return self.iterator[kind].reset()

def map(ax, start=None, stop=-1, step=1,c=1, **kwargs):

    if ax != Subplot:
        ax=gca()
    if start == 'h' or start == 'H':
        start = len(ax.get_lines())/2
    if stop == 'h' or stop == 'H':
        stop = len(ax.get_lines())/2

    list=__getObjRange(ax.get_lines(), start, stop, step)
    if isinstance(c, (str, unicode)):
        for l in list:
            l.set_color(c)
            l.set_markerfacecolor(c)

    if c == 1:
      rb = rainbow_colors(len(list))
      for l in list:
          color = rb.next()
          l.set_color(color)
          l.set_markerfacecolor(color)

    elif c == 2:
        for l in list:
          l.set_color('k')
          l.set_markerfacecolor('k')

    elif c == 3:
        rb = rainbow_colors(len(list))
        rb.next()
        for l in list:
          color = rb.last()
          l.set_color(color)
          l.set_markerfacecolor(color)

    for k,v in kwargs.items():
        for l in list:
            try:
                int(v)
                exec( 'setp(l, %s=%d)'%(k,v))
            except:
                exec( 'setp(l, %s="%s")'%(k,v))

    draw()

def map_markers(ax, start=None, stop=-1, step=1,  mfc=1, mec=0, me=0, **kwargs):#
                #ms=8,  mes=1,
                #ls = '--',
    if ax != Subplot:
        ax=gca()
    if start == 'h' or start == 'H':
        start = len(ax.get_lines())/2
    if stop == 'h' or stop == 'H':
        stop = len(ax.get_lines())/2

    list=__getObjRange(ax.get_lines(), start, stop, step)
    n_sides = [0,  4, 4, 3,  3, 5,   5,  3,  3, 2,  2]
    angles =  [0, 45, 0, 0, 60, 0, 108, 30, 90, 0, 90]
    strides = [3,  0, 0, 0,  0, 0,   0,  0,  0, 1,  1]

    i = -1
    rb = rainbow_colors(len(list))
    if mfc == -1: rb.next()
    for l in list:
        i+=1
        if mfc in [1,-1]:
            if mfc == 1:
                color = rb.next()
            elif mfc == -1:
                color = rb.last()

            m_color = color

            if mec == 0:
                e_color = color
            else:
                e_color = mec
        else:
            color = rb.next()
            if mec == 0:
                e_color = color
            else:
                e_color = mec

            m_color = mfc

        if me != 0:
            l.set_markevery(me)

        l.set_color(color)
        l.set_markerfacecolor(m_color)
        l.set_markeredgecolor(e_color)
#        l.set_markersize(ms)
#        l.set_markeredgewidth(mes)
#        l.set_linestyle(ls)
        l.set_marker((n_sides[i%11], strides[i%11], angles[i%11]))

        for k,v in kwargs.items():
            for l in list:
                try:
                    int(v)
                    exec( 'setp(l, %s=%d)'%(k,v))
                except:
                    exec( 'setp(l, %s="%s")'%(k,v))

    draw()



def __getObjRange(obj, start=None, stop=-1, step=1):
    from numpy import iterable
    #
    # check if obj is iterable
    #
    def _getPos(n):
        if n < 0:
            return max(len(obj)+n+1, 0)
        else:
            return min(n,len(obj))

    if not iterable(obj):
        print('ERROR: Object "%s" is not iterable' %type(obj))
        return 1

    if start is not None:
        start=_getPos(start)
        stop=_getPos(stop)
        return [obj[i] for i in xrange(start, stop, step)]
    else:
        return obj
