#!/usr/bin/env python
#
#       LP_progress.py
#
#       Copyright 2010 Jens Lorrmann <jens@E07-Jens>
#
#       This program is free software; you can redistribute it and/or modify
#       it under the terms of the GNU General Public License as published by
#       the Free Software Foundation; either version 2 of the License, or
#       (at your option) any later version.
#
#       This program is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#       GNU General Public License for more details.
#
#       You should have received a copy of the GNU General Public License
#       along with this program; if not, write to the Free Software
#       Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#       MA 02110-1301, USA.



import sys, time
from array import array
try:
    from fcntl import ioctl
    import termios
except ImportError:
    pass
import signal

try:
    from IPython.core.display import clear_output
    have_ipython = True
except ImportError:
    have_ipython = False

class m_progressBarWidget(object):
    """This is an element of m_progressBar formatting.

    The m_progressBar object will call it's update value when an update
    is needed. It's size may change between call, but the results will
    not be good if the size changes drastically and repeatedly.
    """
    def update(self, pbar):
        """Returns the string representing the widget.

        The parameter pbar is a reference to the calling m_progressBar,
        where one can access attributes of the class for knowing how
        the update must be made.

        At least this function must be overriden."""
        pass

class m_progressBarWidgetHFill(object):
    """This is a variable width element of m_progressBar formatting.

    The m_progressBar object will call it's update value, informing the
    width this object must the made. This is like TeX \\hfill, it will
    expand to fill the line. You can use more than one in the same
    line, and they will all have the same width, and together will
    fill the line.
    """
    def update(self, pbar, width):
        """Returns the string representing the widget.

        The parameter pbar is a reference to the calling m_progressBar,
        where one can access attributes of the class for knowing how
        the update must be made. The parameter width is the total
        horizontal width the widget must have.

        At least this function must be overriden."""
        pass


class ETA(m_progressBarWidget):
    "Widget for the Estimated Time of Arrival"
    def format_time(self, seconds):
        return time.strftime('%H:%M:%S', time.gmtime(seconds))
    def update(self, pbar):
        if pbar.currval == 0:
            return 'ETA:  --:--:--'
        elif pbar.finished:
            return 'Time: %s' % self.format_time(pbar.seconds_elapsed)
        else:
            elapsed = pbar.seconds_elapsed
            eta = elapsed * pbar.maxval / pbar.currval - elapsed
            return 'ETA:  %s' % self.format_time(eta)

class FileTransferSpeed(m_progressBarWidget):
    "Widget for showing the transfer speed (useful for file transfers)."
    def __init__(self):
        self.fmt = '%6.2f %s'
        self.units = ['B','K','M','G','T','P']
    def update(self, pbar):
        if pbar.seconds_elapsed < 2e-6:#== 0:
            bps = 0.0
        else:
            bps = float(pbar.currval) / pbar.seconds_elapsed
        spd = bps
        for u in self.units:
            if spd < 1000:
                break
            spd /= 1000
        return self.fmt % (spd, u+'/s')

class RotatingMarker(m_progressBarWidget):
    "A rotating marker for filling the bar of progress."
    def __init__(self, markers='|/-\\'):
        self.markers = markers
        self.curmark = -1
    def update(self, pbar):
        if pbar.finished:
            return self.markers[0]
        self.curmark = (self.curmark + 1)%len(self.markers)
        return self.markers[self.curmark]

class Percentage(m_progressBarWidget):
    "Just the percentage done."
    def update(self, pbar):
        return '%3d%%' % pbar.percentage()

class Bar(m_progressBarWidgetHFill):
    "The bar of progress. It will strech to fill the line."
    def __init__(self, marker='#', left='|', right='|'):
        self.marker = marker
        self.left = left
        self.right = right
    def _format_marker(self, pbar):
        if isinstance(self.marker, (str, unicode)):
            return self.marker
        else:
            return self.marker.update(pbar)
    def update(self, pbar, width):
        percent = pbar.percentage()
        cwidth = width - len(self.left) - len(self.right)
        marked_width = int(percent * cwidth / 100)
        m = self._format_marker(pbar)
        bar = (self.left + (m*marked_width).ljust(cwidth) + self.right)
        return bar

class ReverseBar(Bar):
    "The reverse bar of progress, or bar of regress. :)"
    def update(self, pbar, width):
        percent = pbar.percentage()
        cwidth = width - len(self.left) - len(self.right)
        marked_width = int(percent * cwidth / 100)
        m = self._format_marker(pbar)
        bar = (self.left + (m*marked_width).rjust(cwidth) + self.right)
        return bar

default_widgets = [Percentage(), ' ', Bar('>'), ' ', 'Test...', ' ',
                   ReverseBar('<'),' ', ETA()]

class m_progressBar(object):
    """This is the m_progressBar class, it updates and prints the bar.

    The term_width parameter may be an integer. Or None, in which case
    it will try to guess it, if it fails it will default to 80 columns.

    The simple use is like this:
    >>> pbar = m_progressBar().start()
    >>> for i in xrange(100):
    ...    # do something
    ...    pbar.update(i+1)
    ...
    >>> pbar.finish()

    But anything you want to do is possible (well, almost anything).
    You can supply different widgets of any type in any order. And you
    can even write your own widgets! There are many widgets already
    shipped and you should experiment with them.

    When implementing a widget update method you may access any
    attribute or function of the m_progressBar object calling the
    widget's update method. The most important attributes you would
    like to access are:
    - currval: current value of the progress, 0 <= currval <= maxval
    - maxval: maximum (and final) value of the progress
    - finished: True if the bar is have finished (reached 100%), False o/w
    - start_time: first time update() method of m_progressBar was called
    - seconds_elapsed: seconds elapsed since start_time
    - percentage(): percentage of the progress (this is a method)
    """
    def __init__(self, maxval=100, title = None, widgets=default_widgets,
                 term_width=None, fd=sys.stdout):
        assert maxval > 0
        self.maxval = maxval
        self.widgets = widgets
        if title is not None:
            self.widgets[4] = title
        self.fd = fd
        self.signal_set = False
        if term_width is None:
            try:
                self.handle_resize(None,None)
                signal.signal(signal.SIGWINCH, self.handle_resize)
                self.signal_set = True
            except:
                self.term_width = 79
        else:
            self.term_width = term_width

        self.currval = 0
        self.finished = False
        self.prev_percentage = -1
        self.start_time = None
        self.seconds_elapsed = 0
        if have_ipython:
            self.clear = self.__clear
        else:
            self.clear = lambda : None

        self.start()

    def __clear(self):
        try:
            clear_output()
        except Exception:
            "# terminal IPython has no clear_output"
            pass

    def handle_resize(self, signum, frame):
        h,w=array('h', ioctl(self.fd,termios.TIOCGWINSZ,'\0'*8))[:2]
        self.term_width = w#int(w/1.2)

    def percentage(self):
        "Returns the percentage of the progress."
        return self.currval*100.0 / self.maxval

    def _format_widgets(self):
        r = []
        hfill_inds = []
        num_hfill = 0
        currwidth = 0
        for i, w in enumerate(self.widgets):
            if isinstance(w, m_progressBarWidgetHFill):
                r.append(w)
                hfill_inds.append(i)
                num_hfill += 1
            elif isinstance(w, (str, unicode)):
                r.append(w)
                currwidth += len(w)
            else:
                weval = w.update(self)
                currwidth += len(weval)
                r.append(weval)
        for iw in hfill_inds:
            r[iw] = r[iw].update(self, (self.term_width-currwidth)/num_hfill)
        return r

    def _format_line(self):
        return ''.join(self._format_widgets()).ljust(self.term_width)

    def _need_update(self):
        return int(self.percentage()) != int(self.prev_percentage)

    def update(self, value=1):
        "Updates the progress bar to a new value."
        value = min(value, self.maxval)
        self.currval += value
        if  self.finished:
            self.finish()
            return
        if not self._need_update():
            return
        if not self.start_time:
            self.start_time = time.time()
        self.seconds_elapsed = time.time() - self.start_time
        self.prev_percentage = self.percentage()

        self.clear()

        if self.currval < self.maxval:
            self.fd.write(self._format_line() + '\r')
            self.fd.flush()
        else:
            self.finished = True
            now = time.time()
            # Caclulate times
            elapsed = now - float(self.start_time)
            elapsed_s = '%s'%('%.1fs'%((elapsed % 60)))
            elapsed_m = '%s'%('%dm:'%(int(elapsed % 3600)/60) if int(elapsed % 3600)/60>0 else '')
            elapsed_h = '%s'%('%dh:'%(int(elapsed / 3600)) if int(elapsed / 3600)>0 else '')
            self.widgets[4] =' Total time: [%s%s%s] ' %(elapsed_h, elapsed_m, elapsed_s)

            self.fd.write(self._format_line() + '\n')
            self.fd.flush()
            self.finish()

    def start(self):
        """Start measuring time, and prints the bar at 0%.

        It returns self so you can use it like this:
        >>> pbar = m_progressBar().start()
        >>> for i in xrange(100):
        ...    # do something
        ...    pbar.update(i+1)
        ...
        >>> pbar.finish()
        """
        self.clear()
        self.fd.write('\n\n')
        self.fd.flush()
        self.update(0)
        return self

    def finish(self):
        """Used to tell the progress is finished."""
        self.clear()
        if not self.finished:
            self.update(self.maxval)

        if self.signal_set:
            signal.signal(signal.SIGWINCH, signal.SIG_DFL)
        self.fd.flush()



def main():
    import os
    from time import sleep
    def example1():
        widgets = ['Test: ', Percentage(), ' ', Bar(marker=RotatingMarker()),
                   ' ', ETA(), ' ', FileTransferSpeed()]
        pbar = m_progressBar(widgets=widgets, maxval=10000000).start()
        for i in range(1000000):
            # do something
            pbar.update(10*i+1)

        pbar.finish()
        print

    def example2():
        class CrazyFileTransferSpeed(FileTransferSpeed):
            "It's bigger between 45 and 80 percent"
            def update(self, pbar):
                if 45 < pbar.percentage() < 80:
                    return 'Bigger Now ' + FileTransferSpeed.update(self,pbar)
                else:
                    return FileTransferSpeed.update(self,pbar)

        widgets = [CrazyFileTransferSpeed(),' <<<', Bar(), '>>> ', Percentage(),' ', ETA()]
        pbar = m_progressBar(widgets=widgets, maxval=10000000)
        # maybe do something
        pbar.start()
        for i in range(2000000):
            # do something
            pbar.update(5*i+1)
            #sleep(0.01)
        pbar.finish()
        print

    def example3():
        widgets = [Bar('>'), ' ', ETA(), ' ', ReverseBar('<')]
        pbar = m_progressBar(widgets=widgets, maxval=10000000).start()
        for i in range(1000000):
            # do something
            pbar.update(10*i+1)
            #sleep(0.01)
        pbar.finish()
        print

    def example4():
        widgets = ['Test: ', Percentage(), ' ',
                   Bar(marker='0',left='[',right=']'),
                   ' ', ETA(), ' ', FileTransferSpeed()]
        pbar = m_progressBar(widgets=widgets, maxval=500)
        pbar.start()
        for i in range(100,500+1,50):
            time.sleep(0.2)
            pbar.update(i)
            #sleep(0.01)
        pbar.finish()
        print


    example1()
    example2()
    example3()
    example4()

if __name__=='__main__':
    main()
