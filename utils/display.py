import matplotlib.pyplot as plt
from screeninfo import get_monitors
import numpy as np
from utils.circ import *
from utils.boldoc import *

def slm_disp1(M: np.ndarray, monitor: int = 2):
    """
    Display an image on a specific monitor by creating or reusing
    figure 1, then setting the figure window to match that monitor.

    Parameters
    ----------
    M : (H, W[, C]) array_like
        Image matrix to display (grayscale or RGB).
    monitor : int, optional
        1-based index of the monitor (per screeninfo.get_monitors()),
        by default 1.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Handle to the figure.
    ax  : matplotlib.axes.Axes
        Handle to the axes containing the image.
    """
    # --- Query all monitors ---
    mons = get_monitors()
    if monitor < 1 or monitor > len(mons):
        raise ValueError(f"Monitor must be between 1 and {len(mons)}")
    mon = mons[monitor - 1]

    # --- Create or reuse figure 1 ---
    fig = plt.figure(1)
    fig.clf()

    # --- Display the image ---
    ax = fig.add_subplot(111)
    ax.imshow(M, aspect='auto', interpolation='nearest')
    ax.axis('off')

    # --- Position & resize the window to cover the monitor ---
    mng = plt.get_current_fig_manager()
    try:
        # Qt backend (e.g. Qt5Agg)
        mng.window.setGeometry(mon.x, mon.y, mon.width, mon.height)
    except AttributeError:
        try:
            # TkAgg backend
            mng.window.wm_geometry(f"{mon.width}x{mon.height}+{mon.x}+{mon.y}")
        except Exception:
            pass

    # --- Force draw ---
    plt.ion()
    plt.show()
    #plt.pause(0.5)
    #plt.waitforbuttonpress()

    return fig, ax




