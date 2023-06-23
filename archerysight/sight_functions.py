"""Routines for sight mark calculations"""

from typing import Tuple
from dataclasses import dataclass
import numpy as np

from archerysight.constants import GRAV, M2INCH, GN2KG
import archerysight.trajectories as traj
import archerysight.arrow as arrow


@dataclass
class SightParams:
    """
    Dataclass to hold information about a sight.
    """
    n_click = 20  # Number of clicks per turn
    m_per_inch = 24.0
    x_pa = 0.0
    x_ps = 0.0


def get_x_s(
    x_t: float,
    theta_a: float,
    x_pa: float,
    x_ps: float,
) -> float:
    """
    Get sight height below peep for given launch angle and target.

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
    theta_a = traj.get_theta_a_projectile(v_0, x_t)
    x_s = get_x_s(x_t, theta_a, x_pa, x_ps)
    m_s = x_s_to_sight_mark(x_s, m_datum, m_per_inch)
    return m_s



if __name__ == "__main__":
    vel = root_find(
        x_t=np.array([20.0, 60.0]),
        m_s=np.array([25.0, 53.8]),
        x_pa=98.0e-3,
        x_ps=0.80,
        m_per_inch=24.0,
    )

    print(f"bow speed is {vel * 3.28083} feet per second.")

    theta_a = traj.get_theta_a_projectile(
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

    # plt.plot(np.arange(5.0, 92.0, 5.0), m_s)
    # plt.savefig("marks.png")



    x_t = 70.0
    v_0 = 330 / 3.28083

    theta_a = traj.get_theta_a_projectile(
        v_0=v_0,
        x_t=x_t,
    )
    print(f"Hitting {x_t} m with a {v_0*3.28083} fps requires an angle of {theta_a}.")

    

    x_rk, y_rk, _, _, _, _ = traj.integrate_RK(v_0, theta_a, 70.0, 1.0, arrow.ArrowParams, 0.0, 1.0, dt=None, imax=None)

    print(f"RK with angle of {theta_a} and velocity of {v_0*3.28083} fps goes {x_rk[-1]} m to land at {y_rk[-1]} m.")

    print(np.amax(y_rk))

    import matplotlib.pyplot as plt
    plt.plot(x_rk, y_rk, "r.")
    plt.plot(x_rk, np.tan(theta_a)*x_rk*(1.0-x_rk/70.0), "bo")
    plt.savefig("comparison.png")
