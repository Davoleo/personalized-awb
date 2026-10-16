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

def interactive(render_fn, base_gains):
    """
    render_fn(gains: np.ndarray) -> uint8 sRGB image
    """
    win = "Personalized AWB"
    cv.namedWindow(win, cv.WINDOW_AUTOSIZE)

    def to_param(val):
        return (val-STEPS) / STEPS

    def refresh(_=None):
        t = to_param(cv.getTrackbarPos('R/B (t)', win))
        q = to_param(cv.getTrackbarPos('R&B/G (q)', win))
        gains = base_gains * user_gains(t, q, ALPHA, BETA)
        out = render_fn(gains)
        cv.imshow(win, out)

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

    cv.destroyAllWindows()