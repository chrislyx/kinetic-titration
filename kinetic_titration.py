import numpy as np
from scipy import stats
from scipy import integrate
from lmfit import minimize, Parameters

def remove_RU_outliers( Rt, zcut=3.):
    '''Detect outliers in the response curve, and remove them from the data.

    The outliers are detected by |z-score| > zcut.

    Args:

    Rt:  2-dimensional array; Rt[:,0] gives the time points, Rt[:,1] gives
    the response units.

    Returns:

    Rt: 2-dimensional array; Rt with the outliers removed.

    outliers: 2-dimensional array; the subset of Rt that are removed
    as outliers.  
    '''
    outidx = np.arange( len(Rt))[ np.abs( stats.zscore( Rt[:,1])) > zcut]
    outliers = Rt[outidx]
    return np.delete( Rt, outidx, axis=0), outliers

def remove_RU_wash_spikes( Rt, tCinject, twash):
    '''
    '''
    pass

def rates_two_compartments( Csurf, Bound, Cliquid, Rmax, ka, kd, kM):
    '''The rates of change for the free and the bound analyte
    concentrations at the chip surface, according to the
    two-compartment model for surface binding:

    C(liquid) = C(surface)  (diffusion into the surface layer)
    C(surface) + R = C.R    (binding reaction with the immobilized receptors)

    Args:

    Csurf, Bound: instantaneous concentrations of free and
    receptor-bound analytes in the surface layer.

    Cliquid: the analyte concentration in the flow liquid phase.

    Rmax: maximum response, determined by the receptor density on the surface.

    ka, kd: the association and dissociation rate constants.

    kM: exchange rate constant of the analyte between the flow liquid phase and
    the surface layer.

    Returns:

    dCsurfdt, dBounddt: the rate of changes of the free and the bound
    analyte concentrations in the surface layer.

    '''
    
    dCsurfdt = -ka*Csurf*(Rmax - Bound) + kd*Bound + kM*(Cliquid - Csurf)
    dBounddt = ka*Csurf*(Rmax - Bound) - kd*Bound
    # print(Cliquid, Csurf, Bound, dCsurfdt, dBounddt)
    return dCsurfdt, dBounddt

def jac_rates_two_compartments( Csurf, Bound, Cliquid, Rmax, ka, kd, kM):
    '''Compute the Jacobian of the rates of change for Csurf and Bound
with respect to Csurf and Bound.
    
    Args:

    Same as in rates_two_compartments().

    Returns:

    jac: 2x2 matrix, 
    ( dCsurfdt/dCsurf, dCsurfdt/dBound )
    ( dBounddt/dCsurf, dBounddt/dBound )

    '''
    jac = np.array( [
        [ -ka*(Rmax - Bound) - kM, ka*Csurf + kd ],
        [ ka*(Rmax - Bound), -ka*Csurf - kd ] ])
    return jac

def titration_rapid_mixing( tCinject, tinject, Rmax, ka, kd):
    '''
    Compute the titration curve assuming rapid mixing (kM >> 1).
    '''
    # Each injection consists of two events at which the concentration
    # changes: the injection of analytes followed by injection of
    # blank after tinject.
    nevents = 2*len(tCinject)
    tevents = np.zeros( nevents)
    tevents[::2] = tCinject[:,0]
    tevents[1::2] = tCinject[:,0] + tinject
    Ct = np.zeros( nevents)
    
    # The analyte concentration at each event time point.
    Ct[::2] = tCinject[:,1]
    Ct[1::2] = 0.

    # Compute the equilibrium Bound concentration between events i and i+1.
    #
    # Beq[i] = ka Ct Rmax/(ka Ct + kd)
    #
    # where Ct is the concentration between events i and i+1, equal to
    # the concetration at time t_i.
    #
    # Also compute the Bound concentration at each event time point.
    # B[i+1] = Beq[i] + (B[i] - Beq[i]) exp( - (ka Ct[i] + kd) (t_{i+1} - t_i))
    # where t is the time between events i and i+1, t_i is the time of event
    # i.
    Beq = np.zeros( nevents)
    Bt = np.zeros( nevents)
    kr = np.zeros( nevents)

    for i, t in enumerate( tevents):
        C = Ct[i]
        Beq[i] = ka*C*Rmax/(ka*C + kd)
        kr[i] = ka*C + kd
        if i>0:
            dt = tevents[i] - tevents[i-1]
            Bt[i] = Beq[i-1] + (Bt[i-1] - Beq[i-1])*np.exp( -kr[i-1]*dt)

    return dict( tevents=tevents,
                 Ct = Ct,
                 Beq = Beq,
                 kr = kr,
                 Bt = Bt)
                 
def CliquidAtTime( time, tCinject, tinject):
    '''
    Return the analyte concentration in the flow liquid phase at time t.
    '''
    idx = np.searchsorted( tCinject[:,0], time, side='right')
    if idx==0: return 0.
    # the analyte concentration drops to zero after tinject time of each
    # injection.
    if time > tCinject[idx-1, 0] + tinject: return 0.
    return tCinject[idx-1, 1]

def calc_response_curve( times, tCinject, tinject, Rmax, ka, kd, kM):
    '''Solve the ODE for the reactions and compute the response vs time curve.

    Args:

    times: array; the time points at which the responses are to be calculated.

    tCinject: two dimensional array; tCinject[:,0] gives the time
    points of analyte injections and tCinject[:,1] gives the
    corresponding analyte concentration.

    tinject: float; the duration of each injection, after which the
    analyte concentraton in the flow liquid phase drops to zero.

    Rmax: maximum response, determined by the receptor density on the surface.

    ka, kd: the association and dissociation rate constants.

    kM: exchange rate constant of the analyte between the flow liquid phase and
    the surface layer.

    Returns:

    Rt: array; Rt[i] gives the calculated response unit at time times[i].

    '''

    def CliquidAtT( t):
        return CliquidAtTime( t, tCinject, tinject)
    
    def rates( t, y, Rmax, ka, kd, kM):
        Cliquid = CliquidAtT( t)
        return np.array( rates_two_compartments( y[0], y[1], Cliquid,
                                                 Rmax, ka, kd, kM))

    def jac_rates( t, y, Rmax, ka, kd, kM):
        Cliquid = CliquidAtT( t)
        return jac_rates_two_compartments( y[0], y[1], Cliquid,
                                           Rmax, ka, kd, kM)


    # At time 0, the analyte concentration on the surface is 0.
    y0 = np.array( [0., 0.])
    tmin = np.min(times)
    tmax = np.max(times)
    # Set the max_step such that the integration step will not exceed
    # 0.2 times of the minimum time intervals between injections, in order
    # to prevent the integrator from overstepping the concentration changes. 
    dtinject = tCinject[1:,0] - tCinject[:-1,0]
    max_step = 0.2*np.min( dtinject)
    #import pdb; pdb.set_trace()
    results = integrate.solve_ivp( rates, (tmin, tmax), y0,
                                   method='LSODA',
                                   t_eval=times,
                                   args = (Rmax, ka, kd, kM),
                                   jac = jac_rates,
                                   max_step = max_step)

    return results.t, results.y[1]

def calc_response_curve_rapid_mixing( times, tCinject, tinject, Rmax, ka, kd):
    '''Solve the ODE for the reactions and compute the response vs time
    curve, under the rapid mixing regime, i.e. kM >> 1.

    Args:

    times: array; the time points at which the responses are to be calculated.

    tCinject: two dimensional array; tCinject[:,0] gives the time
    points of analyte injections and tCinject[:,1] gives the
    corresponding analyte concentration.

    tinject: float; the duration of each injection, after which the
    analyte concentraton in the flow liquid phase drops to zero.

    Rmax: maximum response, determined by the receptor density on the surface.

    ka, kd: the association and dissociation rate constants.

    kM: exchange rate constant of the analyte between the flow liquid phase and
    the surface layer.

    Returns:

    Rt: array; Rt[i] gives the calculated response unit at time times[i].

    '''
    events = titration_rapid_mixing( tCinject, tinject, Rmax, ka, kd)
    # index of the events.
    eidx = -1
    tevents = events['tevents']
    Rt = np.zeros( len(times))
    t0 = 0.
    Beq = 0.
    B0 = 0.
    Ct = 0.
    kr = 0.
    for j, t in enumerate(times):
        if eidx < len(tevents)-1 and t >= tevents[eidx+1]:
            eidx += 1
            t0 = tevents[eidx]
            Beq = events['Beq'][eidx]
            B0 = events['Bt'][eidx]
            Ct = events['Ct'][eidx]
            kr = events['kr'][eidx]
        Rt[j] = Beq + (B0-Beq)*np.exp(-kr*(t - t0))

    return times, Rt

def fit_SPR( Rt, tCinject, tinject, Rmax=None, ka=1., kd=1., kM=1.):
    '''Fit kinetic parameters to the SPR data.

    Args:

    Rt: two dimensional array; Rt[:,0] gives the time points and
    Rt[:,1] gives the corresponding response units.

    tCinject: two dimensional array; tCinject[:,0] gives the time of
    injections and tCinject[:,1] gives the corresponding injection
    concentration.

    tinject: float; the duration of each injection, after which the
    analyte concentraton in the flow liquid phase drops to zero.

    Rmax, ka, kd, kM: floats; initial guesses for the values of the fitting
    parameters.

    Returns:

    results: fitting results of the kinetic parameters.

    results.params contains the fitted parameters for Rmax, ka, kd, and kM.
    '''
    def residual( params, Rt, tCinject, tinject):
        Rmax = params['Rmax']
        ka = params['ka']
        kd = params['kd']
        kM = params['kM']
        # ka, kd, kM = np.exp(lnka), np.exp(lnkd), np.exp(lnkM)
        #import pdb; pdb.set_trace()
        times, Rtmodel = calc_response_curve( Rt[:,0], tCinject, tinject,
                                              Rmax, ka, kd, kM)
        # print( 'Rmax=%g, ka=%g, kd=%g, kM=%g, residual=%g' % (Rmax, ka, kd, kM, np.mean( np.square( Rt[:,1] - Rtmodel))))
        return Rt[:,1] - Rtmodel

    params = Parameters()

    if Rmax is None: Rmax = np.max( Rt[:,1])
    params.add( 'Rmax', value = Rmax, min = np.max( Rt[:,1]))
    params.add( 'ka', value = ka, min=0.)
    params.add( 'kd', value = kd, min=0.)
    params.add( 'kM', value = kM, min=0.)
    result = minimize( residual, params, args=(Rt, tCinject, tinject),
                       method='least_squares', nan_policy='omit')

    return result

def fit_SPR_rapid_mixing( Rt, tCinject, tinject, Rmax=None, ka=1., kd=1.):
    '''Fit kinetic parameters to the SPR data, under the rapid mixing
    regime, i.e. kM >> 1.

    Args:

    Rt: two dimensional array; Rt[:,0] gives the time points and
    Rt[:,1] gives the corresponding response units.

    tCinject: two dimensional array; tCinject[:,0] gives the time of
    injections and tCinject[:,1] gives the corresponding injection
    concentration.

    tinject: float; the duration of each injection, after which the
    analyte concentraton in the flow liquid phase drops to zero.

    Rmax, ka, kd: floats; initial guesses for the values of the fitting
    parameters.

    Returns:

    results: fitting results of the kinetic parameters.

    results.params contains the fitted parameters for Rmax, ka, and kd.

    '''
    def residual( params, Rt, tCinject, tinject):
        Rmax = params['Rmax']
        ka = params['ka']
        kd = params['kd']
        # ka, kd = np.exp(lnka), np.exp(lnkd)
        times, Rtmodel = calc_response_curve_rapid_mixing( Rt[:,0], tCinject, tinject, Rmax, ka, kd)
        # print( 'Rmax=%g, ka=%g, kd=%g, residual=%g' % (Rmax, ka, kd, np.mean( np.square( Rt[:,1] - Rtmodel))))
        return Rt[:,1] - Rtmodel

    params = Parameters()

    if Rmax is None: Rmax = np.max( Rt[:,1])
    params.add( 'Rmax', value=Rmax, min=np.max( Rt[:,1]))
    params.add( 'ka', value=ka, min=0.)
    params.add( 'kd', value=kd, min=0.)

    result = minimize( residual, params, args=(Rt, tCinject, tinject),
                       method='least_squares', nan_policy='omit')

    return result

def loadSPR( RU, inject):
    '''Load SPR data, including 1) the response vs time from RU file and
    2) the injection time points and the corresponding analyte
    concentrations from the inject file.

    Returns:

    Rt: two dimensional array; Rt[:,0] gives the time points of the
    SPR readings, Rt[:,1] gives the corresponding response units.

    tCinject: two dimensional array; tCinject[:,0] gives the time
    points of analyte injections and tCinject[:,1] gives the
    corresponding analyte concentration.

    '''
    nHeaderRows = 16
    Rt = np.loadtxt( RU, skiprows = nHeaderRows)
    tCinject = np.loadtxt( inject, skiprows = 1)
    return Rt, tCinject

def main():
    Rt, tCinject = loadSPR( "data/ASPIRE_Her1short_Her1aza_50mMNaCl_04132021 Full Assay.txt", "data/ASPIRE_Her1short_Her1aza_50mMNaCl_04132021 concs.txt")

    Rt, outliers = remove_RU_outliers( Rt)
    
    tinject = 120

    calc_response_curve_rapid_mixing( Rt[:,0], tCinject, tinject, 100., 1.e-3, 1.e-4)
    
    resultsRM = fit_SPR_rapid_mixing( Rt, tCinject, tinject)
    
    results = fit_SPR( Rt, tCinject, tinject,
                       Rmax = resultsRM.params['Rmax'].value,
                       ka = 0.1*resultsRM.params['ka'].value,
                       kd = resultsRM.params['kd'].value,
                       kM = 1.e-3)

    return resultsRM, results

if __name__ == '__main__':
    main()
    
