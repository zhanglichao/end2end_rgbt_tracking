import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2 as cv
import time
import os
import vot
from vot import Rectangle
import numpy as np

class BaseTracker:
    """Base class for all trackers."""

    def __init__(self, params):
        self.params = params

    def initialize(self, image, state, class_info=None):
        """Overload this function in your tracker. This should initialize the model."""
        raise NotImplementedError

    def track(self, image):
        """Overload this function in your tracker. This should track in the frame and update the model."""
        raise NotImplementedError

    def track_sequence(self, sequence):
        """Run tracker on a sequence."""

        # Initialize
        image = self._read_image(sequence.frames[0])

        times = []
        start_time = time.time()
        self.initialize(image, sequence.init_state)
        init_time = getattr(self, 'time', time.time() - start_time)
        times.append(init_time)

        if self.params.visualization:
            self.init_visualization()
            self.visualize(image, sequence.init_state)

        # Track
        tracked_bb = [sequence.init_state]
        for frame in sequence.frames[1:]:
            image = self._read_image(frame)

            start_time = time.time()
            state = self.track(image)
            times.append(time.time() - start_time)

            tracked_bb.append(state)

            if self.params.visualization:
                self.visualize(image, state)

        return tracked_bb, times
    def track_vot(self, imgtype):
        """Run tracker on a sequence."""
        
        handle = vot.VOT("rectangle", "rgbt")
        rect = list(handle.region())
        colorimage, thermalimage = handle.frame()
        startnum = 20
        if imgtype == 'rgb':
            image = self._read_image(colorimage[startnum:len(colorimage)-2])
        elif imgtype == 'ir':
            image = self._read_image(thermalimage[startnum:len(thermalimage)-2])
        self.initialize(image, rect)
        
        while True:
            colorimage, thermalimage = handle.frame()
            if not colorimage:
                break
            print(imgtype)
            if imgtype == 'rgb':
                image = self._read_image(colorimage[startnum:len(colorimage)-2])
            elif imgtype == 'ir':
                image = self._read_image(thermalimage[startnum:len(thermalimage)-2])
            state = self.track(image)
            region = np.array(state).astype(int)
            handle.report(Rectangle(region[0], region[1], region[2], region[3]))
    def track_vot2(self, imgtype):
        """Run tracker on a sequence."""
        
        handle = vot.VOT("rectangle", "rgbt")
        rect = list(handle.region())
        colorimage, thermalimage = handle.frame()
        startnum = 20        
        image_rgb = self._read_image(colorimage[startnum:len(colorimage)-2])
        image_ir = self._read_image(thermalimage[startnum:len(thermalimage)-2])
        if imgtype == 'rgb':
            self.initialize(image_ir, image_rgb, rect)
        elif imgtype == 'ir':
            self.initialize(image_rgb, image_ir, rect)
        while True:
            colorimage, thermalimage = handle.frame()
            if not colorimage:
                break
            image_rgb = self._read_image(colorimage[startnum:len(colorimage)-2])
            image_ir = self._read_image(thermalimage[startnum:len(thermalimage)-2])
            if imgtype == 'rgb':
                state = self.track(image_ir, image_rgb)
            elif imgtype == 'ir':
                state = self.track(image_rgb, image_ir)
            region = np.array(state).astype(int)
            handle.report(Rectangle(region[0], region[1], region[2], region[3]))

    
    def reset_tracker(self):
        pass

    def press(self, event):
        if event.key == 'p':
            self.pause_mode = not self.pause_mode
            print("Switching pause mode!")
        elif event.key == 'r':
            self.reset_tracker()
            print("Resetting target pos to gt!")

    def init_visualization(self):
        # plt.ion()
        self.pause_mode = False
        self.fig, self.ax = plt.subplots(1)
        self.fig.canvas.mpl_connect('key_press_event', self.press)
        plt.tight_layout()

    def visualize(self, image, state):
        self.ax.cla()
        self.ax.imshow(image)
        rect = patches.Rectangle((state[0], state[1]), state[2], state[3], linewidth=1, edgecolor='r', facecolor='none')
        self.ax.add_patch(rect)

        if hasattr(self, 'gt_state') and False:
            gt_state = self.gt_state
            rect = patches.Rectangle((gt_state[0], gt_state[1]), gt_state[2], gt_state[3], linewidth=1, edgecolor='g',
                                     facecolor='none')
            self.ax.add_patch(rect)
        self.ax.set_axis_off()
        self.ax.axis('equal')
        plt.draw()
        plt.pause(0.001)

        if self.pause_mode:
            plt.waitforbuttonpress()

    def _read_image(self, image_file: str):
        return cv.cvtColor(cv.imread(image_file), cv.COLOR_BGR2RGB)

