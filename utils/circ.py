import numpy as np

def create_circle(SLM_width: int,
                  SLM_height: int,
                  R: float,
                  shiftx: float = 0.0,
                  shifty: float = 0.0) -> np.ndarray:
    """
    Create a binary circular mask of radius R on an SLM of given width/height,
    centered at (SLM_width/2 - shiftx, SLM_height/2 - shifty).

    Parameters
    ----------
    SLM_width : int
        Number of columns (pixels) of the SLM.
    SLM_height : int
        Number of rows (pixels) of the SLM.
    R : float
        Radius of the circle (in pixels).
    shiftx : float, optional
        Horizontal shift of the circle center (in pixels), by default 0.
    shifty : float, optional
        Vertical shift of the circle center (in pixels), by default 0.

    Returns
    -------
    circle_mask : (SLM_height, SLM_width) uint8 array
        Binary mask: 1 inside the circle, 0 outside.
    """
    # Create 1-based coordinate grids to match MATLAB’s meshgrid(1:W,1:H)
    x = np.arange(1, SLM_width + 1)
    y = np.arange(1, SLM_height + 1)
    X, Y = np.meshgrid(x, y)  # shapes (SLM_height, SLM_width)

    # Compute shifted center
    center_x = SLM_width / 2.0 - shiftx
    center_y = SLM_height / 2.0 - shifty

    # Compute distance from center
    distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)

    # Build binary mask
    circle_mask = np.zeros((SLM_height, SLM_width), dtype=np.uint8)
    circle_mask[distance <= R] = 1

    return circle_mask



