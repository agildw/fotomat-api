import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import tau
from scipy.integrate import quad_vec
from tqdm import tqdm
from matplotlib.animation import FuncAnimation,FFMpegWriter
# import matplotlib.animation as animation
from io import BytesIO
import tempfile

Mat = np.typing.NDArray[np.uint8]

def load_and_process_image(image: Mat):
    # img = cv2.imread(image_path)
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_contour = max(contours, key=cv2.contourArea)
    contour = max_contour.squeeze()
    x_list, y_list = contour[:, 0].astype(np.float64), -contour[:, 1].astype(np.float64)
    x_list -= np.mean(x_list)
    y_list -= np.mean(y_list)
    return x_list, y_list

def compute_fourier_coefficients(x_list, y_list, order=100):
    t_list = np.linspace(0, tau, len(x_list))
    def f(t): return np.interp(t, t_list, x_list + 1j * y_list)

    c = []
    with tqdm(total=2*order + 1, desc="Calculating Fourier Coefficients") as pbar:
        for n in range(-order, order + 1):
            result, _ = quad_vec(lambda t: f(t) * np.exp(-n * 1j * t), 0, tau)
            coef = result / tau
            c.append(coef)
            pbar.update(1)
    return np.array(c)

def animate_epicycle(x_list, y_list, c, xlim, ylim):
    fig, ax = plt.subplots()
    ax.plot(x_list, y_list, 'g--', linewidth=0.5, alpha=0.5)
    ax.set_xlim(xlim[0] - 300, xlim[1] + 300)
    ax.set_ylim(ylim[0] - 300, ylim[1] + 300)
    ax.set_aspect('equal')
    ax.set_axis_off()

    lines = [ax.plot([], [], 'r-')[0] for _ in range(2 * len(c) + 1)]
    circles = [plt.Circle((0, 0), 0, color='blue', fill=False, linewidth=2) for _ in range(len(c))]
    for circle in circles:
        ax.add_artist(circle)

    point, = ax.plot([], [], 'ro')
    drawing, = ax.plot([], [], 'b-', linewidth=2)

    path = []

    def init():
        for line in lines:
            line.set_data([], [])
        point.set_data([], [])
        drawing.set_data([], [])
        for circle in circles:
            circle.center = (0, 0)
            circle.radius = 0
        return lines + [point, drawing] + circles

    def update(frame):
        time = np.linspace(0, tau, num=600)
        sums = np.cumsum([c[k] * np.exp(1j * (k - len(c)//2) * time[frame]) for k in range(len(c))])
        lines_data = [[(0, 0)] + [(s.real, s.imag) for s in sums]]

        x_prev, y_prev = 0, 0
        for line, circle, s in zip(lines, circles, sums):
            x, y = s.real, s.imag
            line.set_data([x_prev, x], [y_prev, y])
            circle.center = (x_prev, y_prev)
            circle.radius = np.sqrt((x - x_prev)**2 + (y - y_prev)**2)
            x_prev, y_prev = x, y

        point.set_data(sums[-1].real, sums[-1].imag)
        path.append((sums[-1].real, sums[-1].imag))
        drawing.set_data(*zip(*path))
        return lines + [point, drawing] + circles

    
    ani = FuncAnimation(fig, update, frames=600, init_func=init, blit=False)
    # with tempfile.TemporaryFile() as buffer:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        writer = FFMpegWriter(fps=30)
        ani.save(temp_file.name, writer=writer)
        temp_file.seek(0)
        buf = temp_file.read()
        return buf
        
    # writer = FFMpegWriter(fps=30)
    # ani.save(buffer)
    # buffer.seek(0)
    # return buffer




