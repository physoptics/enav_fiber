import sys, clr, os
sys.path.append(r"C:\Users\enav\PycharmProjects\buldoc\fiber_computing_platform\MatlabWork\Optical computing\2LayerSHG\enav\camera_enav")
clr.AddReference("uc480DotNet")

from uc480 import Camera, Defines
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def ok(s):
    try:
        return s == Defines.Status.SUCCESS or int(s) == 0
    except Exception:
        return "SUCCESS" in str(s).upper()


def camera_setup():
    cam = Camera()
    ret = cam.Init()
    if not ok(ret):
        raise RuntimeError(f"Init failed: {ret}")

    cam.Display.Mode.Set(Defines.DisplayMode.DiB)
    # cam.PixelFormat.Set(Defines.ColorMode.Mono8)
    cam.PixelFormat.Set(Defines.ColorMode.RGBA8Packed)
    cam.Gain.Hardware.Boost.SetEnable(False)
    cam.Timing.Exposure.Set(10)
    cam.Gamma.Software.Set(0)
    pos_x, pos_y = 756, 580
    aoi_width, aoi_height = 144, 144
    # Set AOI
    ret = cam.Size.AOI.Set(pos_x, pos_y, aoi_width, aoi_height)
    # Or: ret = cam.Size.AOI.Set(Rectangle(pos_x, pos_y, aoi_width, aoi_height))

    if not ok(ret):
        cam.Exit()
        raise RuntimeError(f"AOI.Set failed: {ret}")
    return cam

def camera_capture(cam):
    ret = cam.Memory.Allocate(True)
    if not ok(ret):
        cam.Exit()
        raise RuntimeError(f"Allocate failed: {ret}")
    ret, rect = cam.Size.AOI.Get()
    if not ok(ret):
        cam.Exit()
        raise RuntimeError(f"AOI.Get failed: {ret}")
    w, h = rect.Width, rect.Height

    ret = cam.Acquisition.Freeze(Defines.DeviceParameter.Wait)
    if not ok(ret):
        cam.Exit()
        raise RuntimeError(f"Freeze failed: {ret}")

    ret, arr = cam.Memory.CopyToArray(True)
    if not ok(ret):
        cam.Exit()
        raise RuntimeError(f"CopyToArray failed: {ret}")


    data = np.frombuffer(arr, dtype=np.uint32)
    rgba = data.reshape(h, w,4)
    print(rgba.max())
    return rgba[:,:,2]






