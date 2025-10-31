""" """

from pylhe import *
import pylhe
import os
import numpy as np
import awkward as ak
from typing import List, Dict, Union, Optional, Tuple
def Jpsi_filter(event: ak.Array, pt_min: float = 4.0, y_max: float = 2.5) -> bool:
    """Filter function to select events with at least one J/psi satisfying the pt and y cuts.

    Args:
        event (ak.Array): The event data containing particle information.
        pt_min (float): Minimum transverse momentum cut for J/psi.
        y_max (float): Maximum rapidity cut for J/psi.

    Returns:
        bool: True if the event contains at least one J/psi passing the cuts, False otherwise.
    """
    # Select J/psi particles (PDG ID 443)
    jpsi_particles = event.particles[event.particles.id == 443]

    # Apply pt and y cuts
    passing_jpsi = jpsi_particles[
        (np.sqrt(jpsi_particles.vector.x**2 + jpsi_particles.vector.y**2) > pt_min**2)
        & (np.abs(jpsi_particles.vector.y) < y_max)
    ]

    # Return True if at least one J/psi passes the cuts
    return ak.num(passing_jpsi) > 0
