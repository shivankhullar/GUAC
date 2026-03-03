"""
This file contains some utility functions for getting scale bar sizes.

Author: Shivan Khullar
Date: March 2026
"""

import numpy as np
from generic_utils.constants import *


def get_scale_bar_size(image_box_size):
    """
    Parameters
    ----------
    image_box_size : float
        Size of the image box in kpc.
    Returns
    -------
    scale_bar_size : float
        Size of the scale bar in kpc.
    scale_bar_label : str
        Label for the scale bar (e.g. "10 pc", "1 kpc", etc.)
    """
    scale_bar_sizes_kpc = np.array([30, 20, 15, 10, 8, 5, 2, 1])
    compare_box_size_max = scale_bar_sizes_kpc*5
    compare_box_size_min = scale_bar_sizes_kpc*4

    try:
        multiplier = "kpc"
        for i in range(len(scale_bar_sizes_kpc)):
            #print (f"Comparing image box size {image_box_size} to range {compare_box_size_min[i]} to {compare_box_size_max[i]}")
            if image_box_size <= compare_box_size_max[i] and image_box_size >= compare_box_size_min[i]:
                #print (f"Image box size {image_box_size} in range for scale bar size {scale_bar_sizes_kpc[i]}")
                scale_bar_value = scale_bar_sizes_kpc[i]
                break
            if image_box_size <= compare_box_size_min[i] and image_box_size > compare_box_size_max[i+1]:
                #print (f"Image box size {image_box_size} less than range for scale bar size {scale_bar_sizes_kpc[i]}")
                scale_bar_value = scale_bar_sizes_kpc[i]
                break        

        print (f"Selected scale bar size: {scale_bar_value} kpc for image box size: {image_box_size} kpc")

    except:
        multiplier = "pc"
        try:
            scale_bar_sizes_pc = np.array([500, 300, 200, 100, 75, 50, 25, 20, 15, 10, 5, 2, 1, 0.5, 0.2, 0.1])
            compare_box_size_max = scale_bar_sizes_pc*5
            compare_box_size_min = scale_bar_sizes_pc*4

            #image_box_sizes = np.linspace(100, 4, 10)
            image_box_size = image_box_size*kpc/pc
            for i in range(len(scale_bar_sizes_pc)):
                #print (f"Comparing image box size {image_box_size} to range {compare_box_size_min[i]} to {compare_box_size_max[i]}")
                if image_box_size <= compare_box_size_max[i] and image_box_size >= compare_box_size_min[i]:
                    #print (f"Image box size {image_box_size} in range for scale bar size {scale_bar_sizes_pc[i]}")
                    scale_bar_value = scale_bar_sizes_pc[i]
                    break
                elif image_box_size <= compare_box_size_min[i] and image_box_size > compare_box_size_max[i+1]:
                    #print (f"Image box size {image_box_size} less than range for scale bar size {scale_bar_sizes_pc[i]}")
                    scale_bar_value = scale_bar_sizes_pc[i]
                    break        
                else:
                    scale_bar_value = scale_bar_sizes_pc[0]
                    #print (f"Image box size {image_box_size} not in range for scale bar size {scale_bar_sizes_pc[i]}")
                #    scale_bar_value = scale_bar_sizes_pc[i]
                    #break

            print (f"Selected scale bar size: {scale_bar_value} pc for image box size: {image_box_size} pc")
        except:
            multiplier = "au"
            try:
                scale_bar_sizes_au = np.array([10000, 5000, 2000, 1000, 500, 200, 100, 50, 20, 10, 5, 2, 1])
                compare_box_size_max = scale_bar_sizes_au*5
                compare_box_size_min = scale_bar_sizes_au*4

                #image_box_sizes = np.linspace(100, 4, 10)
                
                image_box_size = image_box_size*pc/AU
                for i in range(len(scale_bar_sizes_au)):
                    #print (f"Comparing image box size {image_box_size} to range {compare_box_size_min[i]} to {compare_box_size_max[i]}")
                    if image_box_size <= compare_box_size_max[i] and image_box_size >= compare_box_size_min[i]:
                        #print (f"Image box size {image_box_size} in range for scale bar size {scale_bar_sizes_au[i]}")
                        scale_bar_value = scale_bar_sizes_au[i]
                        break
                    elif image_box_size <= compare_box_size_min[i] and image_box_size > compare_box_size_max[i+1]:
                        #print (f"Image box size {image_box_size} less than range for scale bar size {scale_bar_sizes_au[i]}")
                        scale_bar_value = scale_bar_sizes_au[i]
                        break        
                    else:
                        scale_bar_value = scale_bar_sizes_au[0]
                        #print (f"Image box size {image_box_size} not in range for scale bar size {scale_bar_sizes_au[i]}")
                    #    scale_bar_value = scale_bar_sizes_au[i]
                        #break

                print (f"Selected scale bar size: {scale_bar_value} au for image box size: {image_box_size} au")
            except:
                scale_bar_value = scale_bar_sizes_au[-1]
                print (f"Could not determine scale bar size automatically, using previous value of {scale_bar_value} AU.")        
                #scale_bar_value

    if multiplier == "pc":
        if scale_bar_value < 1:
            return scale_bar_value * pc / kpc, f"{scale_bar_value:.1f} pc"
        else:
            return scale_bar_value * pc / kpc, f"{int(scale_bar_value)} pc"
    elif multiplier == "au":
        return scale_bar_value * AU / kpc, f"{int(scale_bar_value)} AU"
    elif multiplier == "kpc":
        return scale_bar_value, f"{int(scale_bar_value)} kpc"
    else:
        return None

