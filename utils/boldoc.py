import numpy as np


def arcsinc2(amp):
    """
    Approximation function:
      f = a*exp(b*x) + c*exp(d*x) + a1*x^10 + a2*x^11 + g*sin(G*x)
    where x = amp.

    Parameters
    ----------
    amp : array_like or scalar
        Input amplitude value(s).

    Returns
    -------
    f : ndarray or scalar
        Output of the approximation.
    """
    x = np.asarray(amp, dtype=float)

    G  = -0.06909
    a  = -0.2979
    a1 = -1.675
    a2 =  2.246
    b  = -1.214
    c  = -2.82
    d  =  0.004894
    g  = -32.29

    f = (a  * np.exp(  b * x) +
         c  * np.exp(  d * x) +
         a1 * np.power(x, 10) +
         a2 * np.power(x, 11) +
         g  * np.sin(  G * x))
    return f


def slm_bolduc_create_mask(
        phi: np.ndarray,
        amp: np.ndarray,
        grating_period: float = 3,
        slm_max: float = 255,
        slm_min: float = 0,
        grating_phase_on: float = 1,
        grating_dir: bool = False,
        slm_width: int = 1920,
        slm_height: int = 1080,
        normalize_phi: bool = True
) -> np.ndarray:
    """
    Generate an 8-bit mask for a phase/amplitude SLM (Bolduc method).

    Parameters
    ----------
    phi : (H, W) array
        Phase map in radians.
    amp : (H, W) array
        Desired amplitude map (will be normalized to [0,1]).
    grating_period : float, optional
        Period of the blazed grating, by default 8.
    slm_max : float, optional
        Maximum SLM gray­level, by default 255.
    slm_min : float, optional
        Minimum SLM gray­level, by default 0.
    grating_phase_on : float, optional
        Grating phase toggle (1 = on), by default 1.
    grating_dir : bool, optional
        Grating direction flag, by default True.
    slm_width : int, optional
        Number of horizontal pixels, by default 1920.
    slm_height : int, optional
        Number of vertical pixels, by default 1080.
    normalize_phi : bool, optional
        If True, scale phi to ±π, by default True.

    Returns
    -------
    mask : (H, W) uint8 array
        8-bit hologram mask to upload to the SLM.
    """
    # --- Create XY grid matching MATLAB’s meshgrid(1:W,1:H) ---
    x = np.arange(1, slm_width + 1)
    y = np.arange(1, slm_height + 1)
    X, Y = np.meshgrid(x, y)  # shapes (H, W)

    # --- Normalize amplitude to [0,1] ---
    amp = amp / np.max(np.abs(amp))

    # --- Optionally normalize phi to ±π ---
    if np.max(phi) != 0 and normalize_phi:
        phi = phi / np.max(np.abs(phi)) * np.pi

    # --- Compute M and F per Bolduc’s formulas ---
    M = 1.0 + arcsinc2(amp) / np.pi
    F = phi - np.pi * M

    # --- Apply grating modulation ---
    if grating_dir:
        T = M * np.mod(F + grating_phase_on * 2 * np.pi * X / grating_period,
                       2 * np.pi)
    else:
        T = M * np.mod(F + 2 * np.pi * (1 - grating_phase_on * X / grating_period),
                       2 * np.pi)

    # --- Scale to [slm_min, slm_max] and convert to uint8 ---
    mask = ((slm_max - slm_min) * T / (2 * np.pi) + slm_min).astype(np.uint8)
    return mask



