"""Functions to calculate arrow trajectories."""

import numpy as np
from archerysight.constants import GRAV, RHO_AIR


def get_theta_a_projectile(
    v_0: float,
    x_t: float,
) -> float:
    """
    Get launch angle for given speed and x_t according to projectile motion.

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

    if type(x_t) is not "np.ndarray":
        x_t = np.asarray(x_t)

    arg = GRAV * x_t / v_0**2
    if arg.any() > 1.0:
        raise ValueError("A bow of speed {v_0} m/s cannot reach {x_t} m.")
    theta_a = 0.5 * np.arcsin(arg)
    return theta_a


def get_theta_a_RK(
    v_0: float,
    x_t: float,
    arrow_data: arrow.ArrowParams
) -> float:
    """
    Get launch angle for given speed and x_t according to Runge-Kutta integration.

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

    # Use projectile motion as initial guess
    theta_a = get_theta_a_projectile(v_0, x_t)

    # Run calculation and iterate until found
    x_rk, y_rk, _, _, _, _ = traj.integrate_RK(v_0, theta_a, x_t, 1.0, arrow_data, 0.0, 1.0, dt=None, imax=None)
    
    return theta_a


def integrate_RK(v_0, alpha_0, x_t, rho, arw_dat, h_arr, h_tar, dt=None, imax=None):
    """
    Itegrate along a trajectory until a fixed x_t is reached.
    
    Parameters
    ----------

    Returns
    -------

    """
    # Initialise
    if dt is None:
        dt = 0.0005
    if imax is None:
        imax = 400000
    # print('dt = ', dt)
    x = np.zeros(1)
    y = np.full(1, h_arr)
    u = np.full(1, v_0 * np.cos(alpha_0))
    v = np.full(1, v_0 * np.sin(alpha_0))
    theta = np.full(1, alpha_0)
    KE = np.full(1, 0.5 * arw_dat.arw_wt * (u[0] ** 2 + v[0] ** 2))

    i = 1
    while (x[i - 1] <= x_t) & (i <= imax):
        # Integrate until x_t reached, or max number of iterations
        un, vn, xn, yn = integrate_step(
            u[i - 1], v[i - 1], x[i - 1], y[i - 1], rho, arw_dat, dt
        )
        u = np.append(u, un)
        v = np.append(v, vn)
        x = np.append(x, xn)
        y = np.append(y, yn)

        theta = np.append(theta, np.arctan(v[i] / u[i]))
        KE = np.append(KE, 0.5 * arw_dat.arw_wt * (u[i] ** 2 + v[i] ** 2))

        i += 1
    if i >= imax:
        sys.exit(
            "\nError Encountered:\nMaximum iterations exceeded in integrate_RK.\nTerminating program."
        )

    return (x, y, u, v, theta, KE)


def integrate_step(um1, vm1, xm1, ym1, rho, arw_dat, dt):
    # performs one step of the RK routine
    # 4th order RK for u,v
    du1, dv1 = f_ballistic(rho, arw_dat, um1, vm1)
    du2, dv2 = f_ballistic(rho, arw_dat, um1 + dt * du1 / 2.0, vm1 + dt * dv1 / 2.0)
    du3, dv3 = f_ballistic(rho, arw_dat, um1 + dt * du2 / 2.0, vm1 + dt * dv2 / 2.0)
    du4, dv4 = f_ballistic(rho, arw_dat, um1 + dt * du3, vm1 + dt * dv3)

    # du1, dv1 = f_projectile()
    # du2, dv2 = f_projectile()
    # du3, dv3 = f_projectile()
    # du4, dv4 = f_projectile()

    un = um1 + (du1 + 2 * du2 + 2 * du3 + du4) * (dt / 6)
    vn = vm1 + (dv1 + 2 * dv2 + 2 * dv3 + dv4) * (dt / 6)

    # 4th order RK integration for x,y: dx/dt = u
    dx1 = um1
    dy1 = vm1
    dx2 = um1 + dx1 * dt / 2.0
    dy2 = vm1 + dy1 * dt / 2.0
    dx3 = um1 + dx2 * dt / 2.0
    dy3 = vm1 + dy2 * dt / 2.0
    dx4 = um1 + dx3 * dt
    dy4 = vm1 + dy3 * dt

    xn = xm1 + (dx1 + 2 * dx2 + 2 * dx3 + dx4) * (dt / 6)
    yn = ym1 + (dy1 + 2 * dy2 + 2 * dy3 + dy4) * (dt / 6)

    return un, vn, xn, yn


def f_projectile():
    """
    Acceleration experienced due to projectile motion.
    i.e. gravity and no drag
    """
    du = 0.0
    dv = -GRAV

    return du, dv


def f_ballistic(rho, arw_dat, ui, vi):
    """
    Acceleration experienced due to ballistic motion.
    i.e. with drag.
    """
    Drag = (
        0.5
        * rho
        * arw_dat.arw_Cd
        * (ui**2 + vi**2)
        * (arw_dat.arw_D**2)
        / arw_dat.arw_wt
    )
    ang = np.arctan(vi / ui)
    du = -Drag * np.cos(ang)
    dv = -GRAV - Drag * np.sin(ang)

    return du, dv
