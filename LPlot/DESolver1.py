#emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the DESolver package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#
#

# required modules
import numpy
import sys, os
from copy import copy
import warnings

# optional modules

# scipy for optimization
try:
    HAS_SCIPY = True
    import scipy.optimize
except ImportError:
    HAS_SCIPY = False

# http;//www.parallelpython.com -
# can be single CPU, multi-core SMP, or cluster parallelization
try:
    import threading
    HAS_PP = True
except ImportError:
    HAS_PP = False
    
# # Import Psyco if available
# try:
#     import psyco
#     psyco.full()
# except ImportError:
#     pass #print "psyco not loaded"

# set up the enumerated DE method types
DE_RAND_1 = 0
DE_BEST_1 = 1
DE_BEST_2 = 2
DE_BEST_1_JITTER = 3
DE_LOCAL_TO_BEST_1 = 4

if HAS_PP:
    class DESolverThread(threading.Thread):
        count=0
        
        """
        Genetic minimization based on Differential Evolution.

        See http://www.icsi.berkeley.edu/~storn/code.html

        """
    
        def __init__(self, deSolver, population, population_size, crossover_prob):
            """
            Initialize and solve the minimization problem.

            """
            self.deSolver = copy(deSolver)
            self.deSolver.crossover_prob = crossover_prob
            self.deSolver.population = population
            self.deSolver.population_size = population_size
            self.deSolver.rot_ind = numpy.arange(population_size)
            self.deSolver.population_errors = numpy.empty(population_size)
            threading.Thread.__init__(self)

        def run(self):
            # try/finally block is to ensure remote worker processes are
            # killed if they were started
            warnings.filterwarnings("ignore")

            self.deSolver._solve(self.deSolver._error_func)
            
                

class DESolver:
    """
    Genetic minimization based on Differential Evolution.

    See http://www.icsi.berkeley.edu/~storn/code.html

    """
    
    def __init__(self, data, model, population_size, max_generations,
                 method = DE_LOCAL_TO_BEST_1, seed=None, scale=[0.5,1.0], 
                 crossover_prob=0.9, goal_error=1e-8, polish=True, 
                 parallel=True, n_proc=4, verbose=False):
        """
        Initialize and solve the minimization problem.

        
        """        
        # transform model vars to deSolver vars
        self.model = model
        self.data = data
        self.parallel = parallel
        self.param_ranges = [[min,max] for min, max in zip(model.mins, model.maxs)]
        self.params = model.params
        self.num_params = len(self.params)
        self.param_names = model.names
        self.param_froozen = model.froozen

        self.population_size = population_size
        self.max_generations = max_generations
        self.method = method
        self.scale = scale
        self.crossover_prob = crossover_prob
        self.goal_error = goal_error
        if polish and not HAS_SCIPY:
            print "WARNING: SciPy was not found on your system, "\
                  "so no polishing will be performed."
            polish = False
        self.polish = polish
        self.verbose = verbose

        # set helper vars
        self.rot_ind = numpy.arange(self.population_size)

        # set status vars
        self.generation = 0

        # set the seed (must be 2D)
        self.seed = seed
        
        self.best_error=1e120

        # set up the population
        if self.seed is None:
            # we'll be creating an entirely new population
            num_to_gen = self.population_size
        else:
            self.seed = numpy.atleast_2d(self.seed)
            num_to_gen = self.population_size - len(self.seed)

        # generate random population
        # eventually we can allow for unbounded min/max values with None
        self.population=[]
        for i in xrange(self.num_params):
            if self.param_froozen[i]:
                self.population.append(numpy.ones(shape=[num_to_gen,1])*self.params[i])
            else:
                tp_max=numpy.log10(abs(self.param_ranges[i][1])) 
                if tp_max == -numpy.inf:
                    tp_max=0.0 
                tp_min=numpy.log10(abs(self.param_ranges[i][0]))
                if tp_min == -numpy.inf:
                    tp_min=0.0 
                    
                if abs(tp_max-tp_min)<=3:
                    self.population.append(numpy.random.normal(self.params[i], 
                                               abs(self.param_ranges[i][1]-\
                                               self.param_ranges[i][0])*0.2, 
                                               size=[num_to_gen,1]))                    
                else:
                    self.population.append(10**(numpy.random.normal(
                                                numpy.log10(self.params[i]), 
                                                abs(tp_max -tp_min)*0.2, 
                                                size=[num_to_gen,1])))
                                                
                                                
        self.population = numpy.hstack(self.population)
        # apply the boundary constraints
        for p in xrange(self.num_params):
            # get min and max
            min_val = self.param_ranges[p][0]
            max_val = self.param_ranges[p][1]

            # find where exceeded max
            ind = self.population[:,p] > max_val
            if ind.sum() > 0:
                # bounce back
                self.population[ind,p] = max_val + \
                                    numpy.random.rand(ind.sum())*\
                                    (max_val-self.population[ind,p])

            # find where below min
            ind = self.population[:,p] < min_val
            if ind.sum() > 0:
                # bounce back
                self.population[ind,p] = min_val + \
                                    numpy.random.rand(ind.sum())*\
                                    (min_val-self.population[ind,p])
        
        # add in the seed if necessary
        if not self.seed is None:
            self.population = numpy.vstack([self.population,seed])

        self.population_errors = numpy.empty(self.population_size)

        # save the best error from each generation
        self.best_gen_errors = numpy.zeros(max_generations)*numpy.nan
        self.best_gen_indv = numpy.zeros((max_generations,self.num_params))*numpy.nan
            
        # set the internal vars
        #self.error_func = error_func # don't keep b/c can't pickle it
        if parallel:
            self.jobs=[]
            population_size = len(self.population)/n_proc
            for i in xrange(n_proc):
                if i == 0:
                    cr=crossover_prob
                else:
                    cr=numpy.random.uniform(0.2,crossover_prob)
                proc=DESolverThread(self, self.population[
                                          population_size*i:\
                                          population_size*(i+1)],
                                    population_size,
                                    crossover_prob=cr )
                self.jobs.append(proc)        
        else:
            pass

                
    def _error_func(self, indiv):
        #print self.model.fcn(indiv, self.data.x, self.model.const), self.data.y
        error = (self.model.fcn(indiv, self.data.x, self.model.const) -\
                 self.data.y)**2 * self.data.we
                 
        return numpy.sum(error*error)

    def _indv_to_dictstr(self,indv):
        return '{' + \
            ', '.join(["'%s': %f" % (name,val) \
                           for name,val \
                           in zip(self.param_names,indv)]) + '}'

    def _report_best(self):
        print "Current generation: %g" % (self.generation)
        print "Current Best Error: %g" % (self.best_gen_errors[self.generation])
        print "Current Best Indiv: " + \
            self._indv_to_dictstr(self.best_gen_indv[self.generation,:])
        print "Overall Best generation: %g" % (self.best_generation)
        print "Overall Best Error: %g" % (self.best_error)
        #print "Best Indiv: " + str(self.best_individual)
        print "Overall Best Indiv: " + self._indv_to_dictstr(self.best_individual)
        print

    def get_scale(self):
        # generate random scale in range if desired
        if isinstance(self.scale,list):
            # return range
            return numpy.random.uniform(self.scale[0],self.scale[1])
        else:
            return self.scale


    def _eval_population(self):
        """
        Evals the provided population, returning the errors from the
        function.
        """
        # see if use job_server
        if self.verbose:
            print "Generation: %d (%d)" % (self.generation,self.max_generations)
            sys.stdout.write('Evaluating population (%d): ' % (self.population_size))

        # eval the function for the initial population
        for i in xrange(self.population_size):
            if self.verbose:
                sys.stdout.write('%d ' % (i))
                sys.stdout.flush()
            self.population_errors[i] = self._error_func(self.population[i,:])
        

        if self.verbose:
            sys.stdout.write('\n')
            sys.stdout.flush()

    def _evolve_population(self):
        """
        Evolove to new generation of population.
        """
        # save the old population
        self.old_population = self.population.copy()
        self.old_population_errors = self.population_errors.copy()

        # index pointers
        rind = numpy.random.permutation(4)+1

        # shuffle the locations of the individuals
        ind1 = numpy.random.permutation(self.population_size)
        pop1 = self.old_population[ind1,:]
        
        # rotate for remaining indices
        rot = numpy.remainder(self.rot_ind + rind[0], self.population_size)
        ind2 = ind1[rot,:]
        pop2 = self.old_population[ind2,:]
        #~ print len(ind1),len(ind2)

        rot = numpy.remainder(self.rot_ind + rind[1], self.population_size)
        ind3 = ind2[rot,:]
        pop3 = self.old_population[ind3,:]

        rot = numpy.remainder(self.rot_ind + rind[2], self.population_size)
        ind4 = ind3[rot,:]
        pop4 = self.old_population[ind4,:]

        rot = numpy.remainder(self.rot_ind + rind[3], self.population_size)
        ind5 = ind4[rot,:]
        pop5 = self.old_population[ind5,:]
        
        # population filled with best individual
        best_population = self.best_individual[numpy.newaxis,:].repeat(self.population_size,axis=0)

        # figure out the crossover ind
        xold_ind = numpy.random.rand(self.population_size,self.num_params) >= \
            self.crossover_prob

        # get new population based on desired strategy
        # DE/rand/1
        if self.method == DE_RAND_1:
            population = pop3 + self.get_scale()*(pop1 - pop2)
            population_orig = pop3
        # DE/BEST/1
        if self.method == DE_BEST_1:
            population = best_population + self.get_scale()*(pop1 - pop2)
            population_orig = best_population
        # DE/best/2
        elif self.method == DE_BEST_2:
            population = best_population + self.get_scale() * \
                         (pop1 + pop2 - pop3 - pop4)
            population_orig = best_population
        # DE/BEST/1/JITTER
        elif self.method == DE_BEST_1_JITTER:
            population = best_population + (pop1 - pop2) * \
                         ((1.0-0.9999) * \
                          numpy.random.rand(self.population_size,self.num_params) + \
                          self.get_scale())
            population_orig = best_population
        # DE/LOCAL_TO_BEST/1
        elif self.method == DE_LOCAL_TO_BEST_1:
            population = self.old_population + \
                         self.get_scale()*(best_population - self.old_population) + \
                         self.get_scale()*(pop1 - pop2)
            population_orig = self.old_population
            
        # crossover
        population[xold_ind] = self.old_population[xold_ind]

        # apply the boundary constraints
        for p in xrange(self.num_params):
            # get min and max
            min_val = self.param_ranges[p][0]
            max_val = self.param_ranges[p][1]

            # find where exceeded max
            ind = population[:,p] > max_val
            if ind.sum() > 0:
                # bounce back
                population[ind,p] = max_val + \
                                    numpy.random.rand(ind.sum())*\
                                    (max_val-population_orig[ind,p])

            # find where below min
            ind = population[:,p] < min_val
            if ind.sum() > 0:
                # bounce back
                population[ind,p] = min_val + \
                                    numpy.random.rand(ind.sum())*\
                                    (min_val-population_orig[ind,p])

        # set the class members
        self.population = population
        self.population_orig = population

    
    def _solve(self):
        """
        Optimize the parameters of the function.
        """
        import os
        #warnings.filterwarnings("ignore")
        #~ sys.stdout = os.devnull
        #~ sys.stderr = os.devnull
        # eval the initial population to fill errors
        self._eval_population()

        # set the index of the best individual
        best_ind = self.population_errors.argmin()
        self.best_error = self.population_errors[best_ind]
        self.best_individual = numpy.copy(self.population[best_ind,:])
        self.best_generation = self.generation

        # save the best for that gen
        self.best_gen_errors[0] = self.population_errors[best_ind]
        self.best_gen_indv[0,:] = self.population[best_ind,:]

        if self.verbose:
            self._report_best()

        # loop over generations
        for g in xrange(1,self.max_generations):
            # set the generation
            self.generation = g

            # update the population
            self._evolve_population()
            
            # evaluate the population
            self._eval_population()

            # set the index of the best individual
            best_ind = self.population_errors.argmin()

            # update what is best
            if self.population_errors[best_ind] < self.best_error:
                self.best_error = self.population_errors[best_ind]
                self.best_individual = numpy.copy(self.population[best_ind,:])
                self.best_generation = self.generation

            # save the best indv for that generation
            self.best_gen_errors[g] = self.population_errors[best_ind]
            self.best_gen_indv[g,:] = self.population[best_ind,:]

            if self.verbose:
                self._report_best()

            # see if done
            if self.best_error < self.goal_error:
                break

            # decide what stays 
            # (don't advance individuals that did not improve)
            ind = self.population_errors > self.old_population_errors
            self.population[ind,:] = self.old_population[ind,:]
            self.population_errors[ind] = self.old_population_errors[ind]

        # see if polish with fmin search after the last generation
        if self.polish:
            if self.verbose:
                print "Polishing best result: %g" % (self.best_error)
                iprint = 1
            else:
                iprint = -1
            # polish with bounded min search
            try:
                polished_individual, polished_error, details = \
                                     scipy.optimize.fmin_l_bfgs_b(self._error_func,
                                                                  self.best_individual,
                                                                  bounds=self.param_ranges,
                                                                  approx_grad=True,
                                                                  iprint=iprint)
                if self.verbose:
                    print "Polished Result: %g" % (polished_error)
                    print "Polished Indiv: " + str(polished_individual)
                if polished_error < self.best_error:
                    # it's better, so keep it
                    # update what is best
                    self.best_error = polished_error
                    self.best_individual = polished_individual
                    self.best_generation = -1
            except:
                pass
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

            
    def Solve(self):
        # try/finally block is to ensure remote worker processes are
        # killed if they were started
        
        try:
            if self.parallel:
                for job in self.jobs:
                    job.start()
                
                for job in self.jobs:
                    job.join()
            else:
                #~ # init solving and solve
                self._solve()
        finally:
            # collect results from the best job
            if self.parallel:
                self.best_error = 1e120
                best_job = self.jobs[0]
               
                for job in self.jobs:
                    if job.deSolver.best_error < self.best_error:
                        self.best_error = job.deSolver.best_error
                        self.best_individual = numpy.copy(job.deSolver.best_individual)
                        self.best_generation = job.deSolver.best_generation
                        best_job = job
                
                if best_job is None:
                    if self.verbose:
                        print '\n!!!WARNING: Fitting failed. Check start parameters\n'
                        print self.params
                    
                else:
                    self.generation = best_job.deSolver.generation
                    self.best_gen_errors = best_job.deSolver.best_gen_errors
                    self.best_gen_indv = best_job.deSolver.best_gen_indv
                    
        warnings.filterwarnings("default")
