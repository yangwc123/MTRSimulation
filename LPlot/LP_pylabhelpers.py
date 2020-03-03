"""
This provides several classes used for blocking interaction with figure windows:

:class:`BlockingInput`
    creates a callable object to retrieve events in a blocking way for interactive sessions

:class:`BlockingKeyMouseInput`
    creates a callable object to retrieve key or mouse clicks in a blocking way for interactive sessions.
    Note: Subclass of BlockingInput. Used by waitforbuttonpress

:class:`BlockingMouseInput`
    creates a callable object to retrieve mouse clicks in a blocking way for interactive sessions.
    Note: Subclass of BlockingInput.  Used by ginput

:class:`BlockingContourLabeler`
    creates a callable object to retrieve mouse clicks in a blocking way that will then be used to place labels on a ContourSet
    Note: Subclass of BlockingMouseInput.  Used by clabel
"""

from matplotlib import verbose
import matplotlib.lines as mlines
from matplotlib.pylab import gcf
from matplotlib.widgets import SpanSelector

class m_BlockingInput(object):
    """
    Class that creates a callable object to retrieve events in a
    blocking way.
    """
    def __init__(self, fig, eventslist=()):
        self.fig = fig
        #~ assert is_sequence_of_strings(eventslist), "Requires a sequence of event name strings"
        self.eventslist = eventslist

    def on_event(self, event):
        """
        Event handler that will be passed to the current figure to
        retrieve events.

        """
        # Add a new event to list - using a separate function is
        # overkill for the base class, but this is consistent with
        # subclasses
        self.add_event(event)

        verbose.report("Event %i" % len(self.events))

        # This will extract info from events
        self.post_event()

        # Check if we have enough events already
        if len(self.events) >= self.n and self.n > 0:
            self.fig.canvas.stop_event_loop()

    def post_event(self):
        """For baseclass, do nothing but collect events"""
        pass

    def cleanup(self):
        """Disconnect all callbacks"""
        for cb in self.callbacks:
            self.fig.canvas.mpl_disconnect(cb)

        self.callbacks=[]

    def add_event(self,event):
        """For base class, this just appends an event to events."""
        self.events.append(event)

    def pop_event(self,index=-1):
        """
        This removes an event from the event list.  Defaults to
        removing last event, but an index can be supplied.  Note that
        this does not check that there are events, much like the
        normal pop method.  If not events exist, this will throw an
        exception.
        """
        self.events.pop(index)

    def pop(self,index=-1):
        self.pop_event(index)
    pop.__doc__=pop_event.__doc__

    def __call__(self, n=1, timeout=30 ):
        """
        Blocking call to retrieve n events
        """

        assert isinstance(n, int), "Requires an integer argument"
        self.n = n

        self.events = []
        self.callbacks = []

        # Ensure that the figure is shown
        try:
            self.fig.show()
        except:
            pass

        # connect the events to the on_event function call
        for n in self.eventslist:
            self.callbacks.append( self.fig.canvas.mpl_connect(n, self.on_event) )

        try:
            # Start event loop
            self.fig.canvas.start_event_loop(timeout=timeout)
        finally: # Run even on exception like ctrl-c
            # Disconnect the callbacks
            self.cleanup()

        # Return the events in this case
        return self.events

class m_BlockingMouseInput(m_BlockingInput):
    """
    Class that creates a callable object to retrieve mouse clicks in a
    blocking way.

    This class will also retrieve keyboard clicks and treat them like
    appropriate mouse clicks (delete and backspace are like mouse button 3,
    enter is like mouse button 2 and all others are like mouse button 1).
    """

    button_add    = 1
    button_pop    = 3
    button_stop   = 2

    def __init__(self, fig, mouse_add=1, mouse_pop=3, mouse_stop=2, marker='o',
                 markersize=8, markerfacecolor='None', markeredgewidth=1,
                 markeredgecolor='r'):
        m_BlockingInput.__init__(self, fig=fig,
                               eventslist=('button_press_event',
                                           'key_press_event') )
        self.button_add = mouse_add
        self.button_pop = mouse_pop
        self.button_stop= mouse_stop
        self.m = marker
        self.ms = markersize
        self.mfc = markerfacecolor
        self.mew = markeredgewidth
        self.mec = markeredgecolor

    def post_event(self):
        """
        This will be called to process events
        """
        assert len(self.events)>0, "No events yet"

        if self.events[-1].name == 'key_press_event':
            self.key_event()
        else:
            self.mouse_event()

    def mouse_event(self):
        '''Process a mouse click event'''

        event = self.events[-1]
        button = event.button

        if button == self.button_pop:
            self.mouse_event_pop(event)
        elif button == self.button_stop:
            self.mouse_event_stop(event)
        else:
            self.mouse_event_add(event)

    def key_event(self):
        '''
        Process a key click event.  This maps certain keys to appropriate
        mouse click events.
        '''

        event = self.events[-1]
        if event.key is None:
            # at least in mac os X gtk backend some key returns None.
            return

        key = event.key.lower()

        if key in ['backspace', 'delete']:
            self.mouse_event_pop(event)
        elif key in ['escape', 'enter']: # on windows XP and wxAgg, the enter key doesn't seem to register
            self.mouse_event_stop(event)
        else:
            self.mouse_event_add(event)

    def mouse_event_add( self, event ):
        """
        Will be called for any event involving a button other than
        button 2 or 3.  This will add a click if it is inside axes.
        """
        if event.inaxes:
            self.add_click(event)
        else: # If not a valid click, remove from event list
            m_BlockingInput.pop(self,-1)

    def mouse_event_stop( self, event ):
        """
        Will be called for any event involving button 2.
        Button 2 ends blocking input.
        """

        # Remove last event just for cleanliness
        m_BlockingInput.pop(self,-1)

        # This will exit even if not in infinite mode.  This is
        # consistent with MATLAB and sometimes quite useful, but will
        # require the user to test how many points were actually
        # returned before using data.
        self.fig.canvas.stop_event_loop()

    def mouse_event_pop( self, event ):
        """
        Will be called for any event involving button 3.
        Button 3 removes the last click.
        """
        # Remove this last event
        m_BlockingInput.pop(self,-1)

        # Now remove any existing clicks if possible
        if len(self.events)>0:
            self.pop(event,-1)

    def add_click(self,event):
        """
        This add the coordinates of an event to the list of clicks
        """
        self.clicks.append((event.xdata,event.ydata))

        verbose.report("input %i: %f,%f" %
                       (len(self.clicks),event.xdata, event.ydata))

        # If desired plot up click
        if self.show_clicks:
            line = mlines.Line2D([event.xdata], [event.ydata],
                                 marker=self.m, markersize=self.ms,
                                 markerfacecolor=self.mfc,
                                 markeredgewidth=self.mew,
                                 markeredgecolor=self.mec)

            event.inaxes.add_line(line)
            self.marks.append(line)
            self.fig.canvas.draw()



    def pop_click(self,event,index=-1):
        """
        This removes a click from the list of clicks.  Defaults to
        removing the last click.
        """
        self.clicks.pop(index)

        if self.show_clicks:

            mark = self.marks.pop(index)
            mark.remove()

            self.fig.canvas.draw()
            # NOTE: I do NOT understand why the above 3 lines does not work
            # for the keyboard backspace event on windows XP wxAgg.
            # maybe event.inaxes here is a COPY of the actual axes?


    def pop(self,event,index=-1):
        """
        This removes a click and the associated event from the object.
        Defaults to removing the last click, but any index can be
        supplied.
        """
        self.pop_click(event,index)
        m_BlockingInput.pop(self,index)

    def cleanup(self,event=None):
        # clean the figure
        if self.show_clicks:

            for mark in self.marks:
                mark.remove()
            self.marks = []

            self.fig.canvas.draw()

        # Call base class to remove callbacks
        m_BlockingInput.cleanup(self)

    def __call__(self, n=1, timeout=30, show_clicks=True):
        """
        Blocking call to retrieve n coordinate pairs through mouse
        clicks.
        """
        self.show_clicks = show_clicks
        self.clicks      = []
        self.marks       = []
        m_BlockingInput.__call__(self,n=n,timeout=timeout)

        return self.clicks

class m_BlockingKeyMouseInput(m_BlockingInput):
    """
    Class that creates a callable object to retrieve a single mouse or
    keyboard click
    """
    def __init__(self, fig):
        m_BlockingInput.__init__(self, fig=fig, eventslist=('button_press_event','key_press_event') )

    def post_event(self):
        """
        Determines if it is a key event
        """
        assert len(self.events)>0, "No events yet"

        self.keyormouse = self.events[-1].name == 'key_press_event'

    def __call__(self, timeout=30):
        """
        Blocking call to retrieve a single mouse or key click
        Returns True if key click, False if mouse, or None if timeout
        """
        self.keyormouse = None
        m_BlockingInput.__call__(self,n=1,timeout=timeout)

        return self.keyormouse

def ginput(figure=None, n=1, timeout=30, show_clicks=True, mouse_add=1, mouse_pop=3, mouse_stop=2, **kwargs):
    """
    call signature::

      ginput(figure, n=1, timeout=30, show_clicks=True,
             mouse_add=1, mouse_pop=3, mouse_stop=2)

    Blocking call to interact with the figure.

    This will wait for *n* clicks from the user and return a list of the
    coordinates of each click.

    If *timeout* is zero or negative, does not timeout.

    If *n* is zero or negative, accumulate clicks until a middle click
    (or potentially both mouse buttons at once) terminates the input.

    Right clicking cancels last input.

    The buttons used for the various actions (adding points, removing
    points, terminating the inputs) can be overriden via the
    arguments *mouse_add*, *mouse_pop* and *mouse_stop*, that give
    the associated mouse button: 1 for left, 2 for middle, 3 for
    right.

    The keyboard can also be used to select points in case your mouse
    does not have one or more of the buttons.  The delete and backspace
    keys act like right clicking (i.e., remove last point), the enter key
    terminates input and any other key (not already used by the window
    manager) selects a point.
    """

    if figure is None:
        figure=gcf()
    blocking_mouse_input = m_BlockingMouseInput(figure, mouse_add =mouse_add,
                                                 mouse_pop =mouse_pop,
                                                 mouse_stop=mouse_stop,
                                                 **kwargs)
    return blocking_mouse_input(n=n, timeout=timeout,
                                show_clicks=show_clicks)


class selector(SpanSelector):
    def __init__(self, ax, **kwargs):
        SpanSelector.__init__(self, ax, lambda x,y:x+y, 'horizontal',useblit=False,
                    rectprops=dict(alpha=0.5, facecolor='red'), **kwargs)

        self.xmin = 0.0
        self.xmax = 0.0

        self.blocking = m_BlockingMouseInput(ax.get_figure(), mouse_add=1,mouse_pop =2,mouse_stop=3)
        self.blocking(n=2, timeout=30, show_clicks=False)

    def release(self, event):
        'on button release event'
        if self.pressv is None or (self.ignore(event) and not self.buttonDown): return
        self.buttonDown = False
        self.event = event
        self.rect.set_visible(False)
        self.canvas.draw()
        self.vmin = self.pressv
        if self.direction == 'horizontal':
            self.vmax = event.xdata or self.prev[0]
        else:
            self.vmax = event.ydata or self.prev[1]

        if self.vmin>self.vmax: self.vmin, self.vmax = self.vmax, self.vmin
        self.span = self.vmax - self.vmin
        self.xmin = self.rect.get_x()
        self.xmax = self.xmin + self.rect.get_width()

        self.pressv = None
        self.blocking.mouse_event_stop(event)
        return False
        