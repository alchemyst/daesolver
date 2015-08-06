#!/usr/bin/env python

# Author: Carl Sandrock 

import numpy
from scipy.optimize import fsolve
from scipy.integrate import odeint


def daesolver(sys, x0, xdot0, tspan, 
              diffeq_indexes=None, algeq_indexes=None, 
              statevar_indexes=None, algvar_indexes=None):
    """ Solve DAEs of the following form:

       f(x, xdot, t) = 0

    The strategy followed is to identify which equations are
    differential equations and which are algebraics, then to use
    scipy.integrate.odeint to integrate the differential part while
    solving the algebraic part at every timestep using
    scipy.optimize.fsolve.

    An attempt is made to figure out which equations contain
    differentials and which variables can be seen as continuous
    states. If this process is not working, you can pass them
    explicitly using the various _indexes input arguments.
    
    """

    def asys(*args):
        return numpy.asarray(sys(*args))
    
    # Make sure the input arguments are arrays
    x0, xdot0, tspan = map(numpy.asarray, [x0, xdot0, tspan])

    # TODO: This should be a seperate routine
    if diffeq_indexes is None:
        # find out which equations contain derivatives by setting the derivatives equal to nan
        # and then seeing how the nan propagates
        testxdot = numpy.zeros_like(xdot0) + numpy.nan
        test = asys(x0, testxdot, 0)

        diffeq_indexes = numpy.flatnonzero(numpy.isnan(test))
        algeq_indexes = numpy.array([i for i in range(len(test)) if i not in diffeq_indexes])

    Neqs = len(diffeq_indexes) + len(algeq_indexes)
    Nvars = len(x0)
    assert Nvars == Neqs, "There must be the same number of variables as equations"

    if statevar_indexes is None:
        # find out which variables are the state variables
        statevar_indexes = []
        algvar_indexes = []
        for i in range(Nvars):
            testxdot = xdot0.copy()
            # Tag one of the variables
            testxdot[i] = numpy.nan
            # See if it affects the equations
            test = asys(x0, testxdot, 0)
            if numpy.any(numpy.isnan(test)):
                statevar_indexes.append(i)
            else:
                algvar_indexes.append(i)

    statevar_indexes = numpy.array(statevar_indexes)
    algvar_indexes = numpy.array(algvar_indexes)
    derDOF = len(diffeq_indexes) - len(statevar_indexes)
    algDOF = len(algeq_indexes) - len(algvar_indexes)
    
    assert derDOF==0, 'The derivative problem must have zero DOF. Eqs - Vars = {}'.format(derDOF)
    assert algDOF==0, 'The algebraic problem must have zero DOF. Eqs - Vars = {}'.format(algDOF)

    def buildvec(der, alg):
        r = numpy.empty(len(statevar_indexes) + len(algvar_indexes))
        r[statevar_indexes] = der
        r[algvar_indexes] = alg
        return r

    alg0 = x0[algvar_indexes]
    
    # This function is for the ODE version of the equations
    def ode(statevar_values, t):
        """ derivatives of state vars """
        # Step 1. Calculate the value of the algebraic values by
        # knowing the state values
        zerodx = buildvec(0, 0)
        def algeqs(algvar_values):
            """ residual on the algebraic equations given the state vars """
            return asys(buildvec(statevar_values, algvar_values), zerodx, t)[algeq_indexes]
        
        algvar_values = fsolve(algeqs, alg0)

        # Step 2. Calculate the values of the derivatives knowing the full current state
        full_state = buildvec(statevar_values, algvar_values)
        def diffeqs(stateder_values):
            """ Residual of the differential equations """
            return asys(full_state, buildvec(stateder_values, 0), t)[diffeq_indexes]

        der0 = xdot0[statevar_indexes]
        stateder_values = fsolve(diffeqs, der0)
        
        return stateder_values

    # integrate the ODE version of the equations    
    y = odeint(ode, x0[statevar_indexes], tspan)

    return y

