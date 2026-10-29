# HIKROBOT MV-CS120-10UC

## Install MVS

Use the official [HIKROBOT download center](https://www.hikrobotics.com/cn/machinevision/service/download/?module=0). On this Ubuntu 24.04 x86_64 computer, select **机器视觉工业相机客户端 MVS V5.1.0 (Linux) (x86_64)**. The installer includes the MVS viewer, Python SDK, examples, and USB driver.

After downloading the official package, follow its included Linux installation instructions. The camera source looks for the SDK under `/opt/MVS`, including `Samples/64/Python/MvImport`.

## First image

Connect the camera directly to a USB 3.x port, attach a lens, and close MVS viewer so the SDK can open the camera exclusively. From the project root:

```bash
.venv/bin/python tools/camera_preview.py --save-frame out/mv_cs120_first.png
```

The preview opens a resizable window; press `q` or Esc to close it. `--no-display` checks and saves one frame without opening a window. `--serial SERIAL` selects a camera by its device serial number when more than one camera is connected. `--width` and `--height` can request a smaller sensor image for a USB 2.0 link; use `--width 640 --height 480` for a basic connection check. A full-resolution capture should use USB 3.x.

Use `--no-display --frames 120` to verify continuous acquisition. The live window includes an increasing frame counter. Standard 8-bit Bayer, monochrome, RGB and BGR images are converted with OpenCV; this avoids a dependency on MVS's optional `libFormatConversion.so` for the MV-CS120-10UC's BayerRG8 output. Other pixel formats use the MVS converter and require the SDK library environment at process startup.

If the image is dark, test manual brightness controls, which are restored on exit:

```bash
.venv/bin/python tools/camera_preview.py --exposure-us 10000 --gain-db 23
```

These values match EGA's camera configuration and produced visible indoor details on this camera. Adjust them for the scene; bright lights may saturate and high gain adds noise. With the original 5000 microsecond exposure and 0 dB gain, indoor details were nearly black. The built-in test pattern and a real ceiling image were both captured successfully, confirming image delivery and a response to light in those tests.

If acquisition returns only a few frames and then `0x80000007`, check the USB connection as well as the capture settings. On this computer the SDK's native C test also stalled. A disconnect briefly enumerated at USB2 speed; after returning to USB3, the native test stalled after three frames with one USB error frame. Unplugging both ends for ten seconds and reconnecting firmly to the USB3 port then restored 20 native frames with no errors and 300 project frames at 28.7 fps, all at 4024×3036. These results establish recovery in those tests; they do not establish long-term stability or identify a single cause among the cable, connectors, host and camera transport stack.

On 2026-10-10, a later GPU preview again ended with `0x80000007`. Reconnecting
the camera produced `0x80000203` when opening it, even with USB3 enumeration,
correct device permissions and no other process holding the device. A USB reset
of this camera alone restored 60 consecutive frames at 26.3 fps. This is a
temporary transport recovery, not evidence that the recurring stability issue is
resolved. Reset only a positively identified camera while no capture client is
using it; do not reset an entire USB bus.

In MVS or the project source, use continuous acquisition and set **Trigger Mode = Off** for free-running capture. The source leaves the camera's frame-rate setting at its device default, returns copied BGR8 images and monotonic capture timestamps, and restores the camera's original width and height when it closes. It does not read or change the intrinsic/extrinsic calibration.

## Use in JGRadar

Set `io.frame_source: camera` in the run configuration. Optional `io.camera_index` (zero-based, default 0) and `io.camera_serial` fields select a camera. `io.camera_width` and `io.camera_height` can request a smaller capture size; both default to 0, which keeps the sensor default. Configure actual intrinsic/extrinsic values and the lookup map before using camera coordinates in deployment; the checked-in camera calibration remains a placeholder.
