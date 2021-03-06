#!/usr/bin/python
# -*- coding: iso-8859-1 -*-
#  Januar 2009 (JL)

 # =====================================================================================
 #
 #       Filename:  Llegend.py
 #
 #    Description:
 #
 #        Version:  1.0
 #        Created:  25.02.2009
 #       Revision:  none
 #
 #         Author:  Jens Lorrmann (Copyright, 2009)
 #
 # =====================================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.transforms as mtransforms
import matplotlib.text as mtext


class LegendLine(lines.Line2D):

   def __init__(self, *args, **kwargs):
      # we'll update the position when the line data is set
      self.text = mtext.Text(0, 0, '')
      self.trans = [ 8, -10 ]
      if kwargs.has_key('trans'):
          self.trans = kwargs['trans']
          kwargs.pop('trans')
      lines.Line2D.__init__(self, *args, **kwargs)

      # we can't access the label attr until *after* the line is
      # inited
      self.text.set_text(self.get_label())

   def set_figure(self, figure):
      self.text.set_figure(figure)
      lines.Line2D.set_figure(self, figure)

   def set_axes(self, axes):
      self.text.set_axes(axes)
      lines.Line2D.set_axes(self, axes)

   def set_transform(self, transform):
      # 2 pixel offset
      texttrans = transform + mtransforms.Affine2D().translate(self.trans[0], self.trans[1])
      self.text.set_transform(texttrans)
      lines.Line2D.set_transform(self, transform)


   def set_data(self, x, y):
      if len(x):
         self.text.set_position((x[-1], y[-1]))

      lines.Line2D.set_data(self, x, y)

   def draw(self, renderer):
      # draw my label at the end of the line with 2 pixel offset
      lines.Line2D.draw(self, renderer)
      self.text.draw(renderer)
