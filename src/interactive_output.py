import numpy as np
import cv2 as cv

ALPHA, BETA = 0.35, 0.25
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
        _, base_gains, render_fn = samples[idx]
        t = to_param(cv.getTrackbarPos('R/B (t)', win))
        q = to_param(cv.getTrackbarPos('R&B/G (q)', win))
        gains = base_gains * user_gains(t, q, ALPHA, BETA)
        cv.imshow(win, render_fn(gains))

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