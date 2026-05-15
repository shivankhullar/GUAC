"""
Scale bar utility for movie frames.

Author: Shivan Khullar
Date: March 2026
"""

import numpy as np
from generic_utils.constants import *   # kpc, pc, AU (CGS cm values)


# 1 pc in kpc  = pc/kpc (exact)
# 1 AU in kpc  = AU/kpc (exact, using CGS constants)
_PC_IN_KPC = pc / kpc      # ≈ 1e-3
_AU_IN_KPC = AU / kpc      # ≈ 4.848e-9


def get_scale_bar_size(image_box_size_kpc):
    """Return (scale_bar_size_kpc, label) for a given image box size in kpc.

    Picks the largest "nice" size that is <= image_box_size / 5, so the bar
    occupies roughly 1/5 of the image width.  Works from Mpc scales down to
    single-AU scales without relying on exception-based tier cascades.

    Parameters
    ----------
    image_box_size_kpc : float
        Full width of the rendered image box, in kpc.

    Returns
    -------
    scale_bar_size_kpc : float
    label : str   e.g. "100 pc", "1 kpc", "500 AU"
    """
    target_kpc = image_box_size_kpc / 5.0

    # Build candidate list (large → small) as (size_in_kpc, label_string).
    # Every tier is represented, so there are no gaps.
    candidates = []

    # kpc / Mpc tier
    for v in [3000, 1000, 500, 300, 100, 50, 30, 20, 15, 10, 8, 5, 3, 2, 1]:
        if v >= 1000:
            label = f"{v // 1000} Mpc"
        else:
            label = f"{v} kpc"
        candidates.append((float(v), label))

    # pc tier (converted to kpc)
    for v in [500, 300, 200, 100, 75, 50, 25, 20, 15, 10, 5, 2, 1,
              0.5, 0.2, 0.1, 0.05, 0.02, 0.01]:
        v_kpc = v * _PC_IN_KPC
        label = f"{v:.2f} pc".rstrip('0').rstrip('.') + " pc" if v < 1 else f"{int(v) if v == int(v) else v} pc"
        # simpler label building:
        if v >= 1:
            label = f"{int(v)} pc"
        else:
            # remove trailing zeros: 0.50 → "0.5 pc", 0.10 → "0.1 pc"
            label = f"{v:.2f}".rstrip('0').rstrip('.') + " pc"
        candidates.append((v_kpc, label))

    # AU tier (converted to kpc)
    for v in [10000, 5000, 2000, 1000, 500, 200, 100, 50, 20, 10, 5, 2, 1]:
        v_kpc = v * _AU_IN_KPC
        candidates.append((v_kpc, f"{v} AU"))

    # Pick the largest candidate that fits within target_kpc
    for size_kpc, label in candidates:
        if size_kpc <= target_kpc:
            print(f"Scale bar: {label}  (box={image_box_size_kpc:.3e} kpc, "
                  f"target={target_kpc:.3e} kpc, bar={size_kpc:.3e} kpc)")
            return size_kpc, label

    # Fallback: smallest candidate (single AU)
    size_kpc, label = candidates[-1]
    print(f"Scale bar fallback: {label}  (box={image_box_size_kpc:.3e} kpc)")
    return size_kpc, label
