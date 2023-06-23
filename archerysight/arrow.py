"""Routines for arrow calculations"""

from dataclasses import dataclass
import numpy as np
from archerysight.constants import GRAV, M2INCH, GN2KG


@dataclass
class ArrowParams:
    # References:
    # - Drag Coefficient of arrows
    #   http://dx.doi.org/10.1007/s12283-012-0102-y
    #
    arw_wt_gn = 350.0
    arw_wt = 350.0 * GN2KG
    arw_D_in = 23.0 / 64.0
    arw_D = (23.0 / 64.0) / M2INCH
    arw_Cd = np.pi * 2.65 / 4.0
