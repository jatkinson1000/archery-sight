"""Routines for sight mark calculations"""

from typing import Tuple
import numpy as np

# gravity, g, 9.81 m/s^2
GRAV = 9.81
M2INCH = 100.0 / 2.54


def get_theta_a_projectile(
    v_0: float,
    x_t: float,
) -> float:
    """
    Get launch angle required for given speed and distance.

    Parameters
    ----------
    v_0 : float
        Arrow velocity [m/s]
    x_t : float
        distance to target [m]

    Returns
    -------
    theta_a : float
        Angle to shoot at to hit target [rad]
    """
    arg = GRAV * x_t / v_0**2
    if arg.any() > 1.0:
        raise ValueError("A bow of speed {v_0} m/s cannot reach {x_t} m.")
    theta_a = 0.5 * np.arcsin(arg)
    return theta_a


def get_x_s(
    x_t: float,
    theta_a: float,
    x_pa: float,
    x_ps: float,
) -> float:
    """
    Get launch angle required for given speed and distance.

    Parameters
    ----------
    x_t : float
        distance to target [m]
    theta_a : float
        Angle to shoot at to hit target [rad]
    x_pa : float
        peep-to-arrow vertical distance [m]
    x_ps : float
        peep-to-sight distance [m]

    Returns
    -------
    x_s : float
        sight measurement below peep height [m]
    """
    theta_pt = np.arctan(x_pa * np.cos(theta_a) / x_t)
    theta_s = theta_a + theta_pt
    x_s = x_ps * np.tan(theta_s)
    return x_s


def get_sight_datum(
    x_s: float,
    m_s: float,
    m_per_inch: float,
) -> float:
    """
    Get datum on sight (mark where sight parallel to peep)

    Parameters
    ----------
    x_s : float
        sight measurement below peep height [m]
    m_s : float
        Sight mark reading for x_s [-]
    m_per_inch : float
        number of marks per inch on the sight [1/inch]

    Returns
    -------
    m_datum : float
        Sight mark corresponding to x_s = 0 [-]
    """
    # convert x_s to inch and then marks for marks below zero
    d_m = (x_s * M2INCH) * m_per_inch
    # subtract from measured to get datum (m_s // peep)
    m_datum = m_s - d_m
    return m_datum


def x_s_to_sight_mark(
    x_s: float,
    m_datum: float,
    m_per_inch: float,
) -> float:
    """
    Convert sight distance into sight mark

    Parameters
    ----------
    x_s : float
        sight measurement below peep height [m]
    m_datum : float
        Sight mark corresponding to x_s = 0 [-]
    m_per_inch : float
        number of marks per inch on the sight [1/inch]

    Returns
    -------
    m_s : float
        Sight mark reading for x_s [-]
    """
    # convert x_s to inch and then marks for marks below zero
    d_m = (x_s * M2INCH) * m_per_inch
    # add to datum to get sight mark
    m_s = m_datum + d_m
    return m_s


def get_sight_mark(
    v_0: float,
    x_t: float,
    m_datum: float,
    x_pa: float,
    x_ps: float,
    m_per_inch: float,
) -> float:
    """
    Get sight mark for distance given known v_0 and m_datum

    Parameters
    ----------
    v_0 : float
        Arrow velocity [m/s]
    x_t : float
        distance to target [m]
    m_datum : float
        Sight mark corresponding to x_s = 0 [-]
    x_pa : float
        peep-to-arrow vertical distance [m]
    x_ps : float
        peep-to-sight distance [m]
    m_per_inch : float
        number of marks per inch on the sight [1/inch]

    Returns
    -------
    m_s : float
        Sight mark reading for x_t [-]
    """
    theta_a = get_theta_a_projectile(v_0, x_t)
    x_s = get_x_s(x_t, theta_a, x_pa, x_ps)
    m_s = x_s_to_sight_mark(x_s, m_datum, m_per_inch)
    return m_s


def root_find(
    x_t: Tuple[float, float],
    m_s: Tuple[float, float],
    x_pa: float,
    x_ps: float,
    m_per_inch: float,
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

    def f_root(v_est, x_t, m_s, x_pa, x_ps, m_per_inch):
        # Take first measurement and v_estimate to get datum
        theta_a = get_theta_a_projectile(v_est, x_t)
        x_s = get_x_s(x_t, theta_a, x_pa, x_ps)
        m_datum = get_sight_datum(x_s[0], m_s[0], m_per_inch)
        # Use datum to get mark for second measurment
        m_est = x_s_to_sight_mark(x_s[1], m_datum, m_per_inch)
        # work out if we over or under estimated v
        return m_est - m_s[1]

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


if __name__ == "__main__":
    vel = root_find(
        x_t=np.array([20.0, 60.0]),
        m_s=np.array([25.0, 53.8]),
        x_pa=98.0e-3,
        x_ps=0.80,
        m_per_inch=24.0,
    )

    print(f"bow speed is {vel * 3.28083} feet per second.")

    theta_a = get_theta_a_projectile(
        v_0=vel,
        x_t=20.0,
    )

    x_s = get_x_s(
        x_t=20.0,
        theta_a=theta_a,
        x_pa=98.0e-3,
        x_ps=0.80,
    )

    m_datum = get_sight_datum(
        x_s=x_s,
        m_s=25.0,
        m_per_inch=24.0,
    )

    print(f"sight datum is {m_datum}")

    m_s = get_sight_mark(
        v_0=vel,
        x_t=np.arange(5.0, 91.0, 5.0),
        m_datum=m_datum,
        x_pa=98.0e-3,
        x_ps=0.80,
        m_per_inch=24.0,
    )
    for i, ival in enumerate(np.arange(5.0, 91.0, 5.0)):
        print(f"{ival:3.0f} - {m_s[i]:3.2f}")

    import matplotlib.pyplot as plt

    plt.plot(np.arange(5.0, 92.0, 5.0), m_s)
    plt.savefig("marks.png")
