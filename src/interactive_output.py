import cv2 as cv
import numpy as np

from src.ttetest import emulate_browser
from src.utils import fit_cam_to_linear_srgb

ALPHA, BETA = 0.20, 0.15
#ALPHA, BETA = 0.25, 0.20
STEPS = 100 # trackbar resolution

def user_gains(t: float, q: float, alpha: float, beta: float) -> np.ndarray:
    """RGB gain offsets based on user input, t and q are sliders centered at (0,0) which means no change"""
    r = np.exp(alpha * t + beta * q)
    g = 1.0
    b = np.exp(-alpha*t + beta * q)
    return np.array([r, g, b])

def interactive(samples):
    """
    samples: list of (label: str, base_gains: np.ndarray, frame_fn)
    render_fn(gains: np.ndarray) -> uint8 sRGB image
    Keys: a / LEFT prev sample, d / RIGHT next sample, r reset sliders, esc quit
    """
    win = "Personalized AWB"
    cv.namedWindow(win, cv.WINDOW_AUTOSIZE)
    idx = 0

    def to_param(val):
        return (val-STEPS) / STEPS

    def refresh(_=None):
        label, base_gains, render_fn = samples[idx]
        t = to_param(cv.getTrackbarPos('R/B (t)', win))
        q = to_param(cv.getTrackbarPos('R&B/G (q)', win))
        u = user_gains(t, q, ALPHA, BETA)

        place_id = label.split('_')
        print(f'data/sony_superraw/{place_id[0]}/{label}'.replace('.png', '.arw'))
        # M = fit_cam_to_linear_srgb(f'data/sony_superraw/{place_id[0]}/{label}'.replace('.png', '.arw'))
        # variant_rgb = cv.cvtColor(render_fn(base_gains), cv.COLOR_BGR2RGB)
        expected_rgb = cv.cvtColor(render_fn(base_gains * u), cv.COLOR_BGR2RGB)
        # emulated_rgb = emulate_browser(variant_rgb, M, u)

        # diagnose(expected_rgb, emulated_rgb, variant_rgb)

        # diff = np.abs(expected_rgb.astype(int) - emulated_rgb.astype(int))
        # print("mean:", diff.mean(), "| 99th pct:", np.percentile(diff, 99), "| max:", diff.max())

        cv.imshow(win, cv.cvtColor(expected_rgb, cv.COLOR_RGB2BGR))

    def switch(d):
        nonlocal idx
        idx = (idx + d) % len(samples)
        label = samples[idx][0]
        print(f"switching to {label}")
        cv.setWindowTitle(win, f"{win} - {label} [{idx + 1}/{len(samples)}]")
        cv.setTrackbarPos("R/B (t)", win, STEPS) # fires refresh on new sample
        cv.setTrackbarPos("R&B/G (q)", win, STEPS)
        refresh()  # make sure we refresh even if trackbar are already in initial position.

    cv.createTrackbar('R/B (t)', win, STEPS, 2*STEPS, refresh)
    cv.createTrackbar('R&B/G (q)', win, STEPS, 2*STEPS, refresh)
    refresh()

    while True:
        key = cv.waitKey(30) & 0xFF
        if key == 27: # esc: quit
            break
        if key == ord('r'):
            cv.setTrackbarPos("R/B (t)", win, STEPS)
            cv.setTrackbarPos("R&B/G (q)", win, STEPS)
        elif key in (ord('d'), 39): # d or ->
            switch(1)
        elif key in (ord('a'), 37): # a or <-
            switch(-1)

    cv.destroyAllWindows()

def diagnose(expected, emulated, variant):
    diff = emulated.astype(int).reshape(-1, 3) - expected.astype(int).reshape(-1, 3)
    pixels = variant.reshape(-1, 3)

    print("signed mean diff (R, G, B):", diff.mean(axis=0))

    # exclude highlights (already clipped in the variant) and deep shadows
    mid = (pixels.max(axis=1) < 200) & (pixels.min(axis=1) > 20)
    print("mean abs diff, mid-tones only:", np.abs(diff[mid]).mean())
    print("mean abs diff, highlights   :", np.abs(diff[~mid]).mean())