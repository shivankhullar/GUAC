import sys
sys.path.append('/src/generic_utils/')
from fire_utils import *
from overall_cloud_props import *
from meshoid import Meshoid


class CloudBox:
    """
    A class to store the parameters of the box around the cloud

    Attributes:
        x_max (float): The maximum x-coordinate of the box
        x_min (float): The minimum x-coordinate of the box
        y_max (float): The maximum y-coordinate of the box
        y_min (float): The minimum y-coordinate of the box
        z_max (float): The maximum z-coordinate of the box
        z_min (float): The minimum z-coordinate of the box
    """
    def __init__(self):
        self.x_max = 0
        self.x_min = 0
        self.y_max = 0
        self.y_min = 0
        self.z_max = 0
        self.z_min = 0
    





########################################################################################
########################### Visualizing clouds   #######################################
########################################################################################


def get_surf_dens(coords, masses, hsml, box_size):
    """
    Get the surface density of the cloud

    Parameters
    ----------
    coords : array_like
        The coordinates of the particles
    masses : array_like
        The masses of the particles
    hsml : array_like
        The smoothing lengths of the particles
    """
    # Use meshoid to get the surface density
    M = Meshoid(coords, masses, hsml)
    image_box_size = box_size 


    return surf_dens
