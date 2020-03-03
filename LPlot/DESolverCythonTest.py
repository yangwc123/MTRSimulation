#!/usr/bin/env python
#
#       DESolverCythonTest.py
#       
#       Copyright 2011 Jens Lorrmann <jens@E07-Jens>
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

import pyximport; pyximport.install()
import DESolver
import LP_fitting

def ff(p,t,c):
    return p[0]*sin(p[1]*t)
    
    
x=linspace(0,10,1e5)
y=80*sin(10*x)+randn(1e5)*5
data=LP_fitting.m_Data(x,y,1.,kind='linear')
model=LP_fitting.m_Model(ff,[{'p0':[1,False,0,1000]},{'p1':[5,False,1e-7,1e5]}])

solver = DESolver.DESolver(data, model,150,100, DESolver.DE_BEST_1, scale=[0.5,1.0], 
                            crossover_prob=0.8, goal_error=1e-7, polish=True, 
                            verbose=False, parallel=False)
                            
%time solver.Solve()

