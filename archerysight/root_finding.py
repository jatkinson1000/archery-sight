"""Routines for root finding"""

from typing import Callable, ParamSpec
import numpy as np

from archerysight.constants import GRAV, M2INCH, GN2KG
import archerysight.trajectories as traj
import archerysight.arrow as arrow


def f_root_projectile(v_est, x_t, m_s, x_pa, x_ps, m_per_inch):
    # Take first measurement and v_estimate to get datum
    theta_a = traj.get_theta_a_projectile(v_est, x_t)
    x_s = get_x_s(x_t, theta_a, x_pa, x_ps)
    m_datum = get_sight_datum(x_s[0], m_s[0], m_per_inch)
    # Use datum to get mark for second measurment
    m_est = x_s_to_sight_mark(x_s[1], m_datum, m_per_inch)
    # work out if we over or under estimated v
    return m_est - m_s[1]


def root_find(
    f_root: Callable,
    x_t: Tuple[float, float],
    m_s: Tuple[float, float],
    x_pa: float,
    x_ps: float,
    m_per_inch: float,
    *args: 
) -> float:
    """
    Calculate the arrow velocity given sight marks using root-finding.

    Parameters
    ----------
    x_t : Tuple[float, float]
        distance to target [m]
    m_s : Tuple[float, float]
        Sight mark corresponding to x_t [-]
    x_pa : float
        peep-to-arrow vertical distance [m]
    x_ps : float
        peep-to-sight distance [m]
    m_per_inch : float
        number of marks per inch on the sight [1/inch]

    Returns
    -------
    v_est: float
        estimate of the arrow velocity [m/s]

    References
    ----------
    Brent's Method for Root Finding in Scipy
    - https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html
    - https://github.com/scipy/scipy/blob/dde39b7cc7dc231cec6bf5d882c8a8b5f40e73ad/
      scipy/optimize/Zeros/brentq.c
    """

    x = [30.0, 150.0]

    f = [
        f_root(x[0], x_t, m_s, x_pa, x_ps, m_per_inch),
        f_root(x[1], x_t, m_s, x_pa, x_ps, m_per_inch),
    ]
    xtol = 1.0e-12
    rtol = 0.00
    xblk = 0.0
    fblk = 0.0
    scur = 0.0
    spre = 0.0
    dpre = 0.0
    dblk = 0.0
    stry = 0.0

    if abs(f[1]) <= f[0]:
        xcur = x[1]
        xpre = x[0]
        fcur = f[1]
        fpre = f[0]
    else:
        xpre = x[1]
        xcur = x[0]
        fpre = f[1]
        fcur = f[0]

    for _ in range(75):
        if (fpre != 0.0) and (fcur != 0.0) and (np.sign(fpre) != np.sign(fcur)):
            xblk = xpre
            fblk = fpre
            spre = xcur - xpre
            scur = xcur - xpre
        if abs(fblk) < abs(fcur):
            # xpre = xcur
            # xcur = xblk
            # xblk = xpre
            xpre, xcur, xblk = xcur, xblk, xcur

            # fpre = fcur
            # fcur = fblk
            # fblk = fpre
            fpre, fcur, fblk = fcur, fblk, fcur

        delta = (xtol + rtol * abs(xcur)) / 2.0
        sbis = (xblk - xcur) / 2.0

        if (fcur == 0.0) or (abs(sbis) < delta):
            v_est = xcur
            break

        if (abs(spre) > delta) and (abs(fcur) < abs(fpre)):
            if xpre == xblk:
                stry = -fcur * (xcur - xpre) / (fcur - xpre)
            else:
                dpre = (fpre - fcur) / (xpre - xcur)
                dblk = (fblk - fcur) / (xblk - xcur)
                stry = -fcur * (fblk - fpre) / (fblk * dpre - fpre * dblk)

            if 2 * abs(stry) < min(abs(spre), 3 * abs(sbis) - delta):
                # accept step
                spre = scur
                scur = stry
            else:
                # bisect
                spre = sbis
                scur = sbis
        else:
            # bisect
            spre = sbis
            scur = sbis
        xpre = xcur
        fpre = fcur
        if abs(scur) > delta:
            xcur += scur
        else:
            if sbis > 0:
                xcur += delta
            else:
                xcur -= delta

        fcur = f_root(xcur, x_t, m_s, x_pa, x_ps, m_per_inch)
        v_est = xcur
    return v_est

