from neuron.gui2.utilities import _segment_3d_pts
import matplotlib.pyplot as plt
import numpy as np
from neuron import h
import math

MIN_CM = -80
MAX_CM = 40

def rotate_matrix (x, y, angle, x_shift=0, y_shift=0, units="DEGREES"):
    """
    Rotates a point in the xy-plane counterclockwise through an angle about the origin
    https://en.wikipedia.org/wiki/Rotation_matrix
    :param x: x coordinate
    :param y: y coordinate
    :param x_shift: x-axis shift from origin (0, 0)
    :param y_shift: y-axis shift from origin (0, 0)
    :param angle: The rotation angle in degrees
    :param units: DEGREES (default) or RADIANS
    :return: Tuple of rotated x and y
    """

    # Shift to origin (0,0)
    x = x - x_shift
    y = y - y_shift

    # Convert degrees to radians
    if units == "DEGREES":
        angle = math.radians(angle)

    # Rotation matrix multiplication to get rotated x & y
    xr = (x * math.cos(angle)) - (y * math.sin(angle)) + x_shift
    yr = (x * math.sin(angle)) + (y * math.cos(angle)) + y_shift

    return xr, yr

def rotate_vecs(x, y, rotation_deg):
    x_rot = []
    y_rot = []
    for cur_x, cur_y in zip(x,y):
        xr, yr = rotate_matrix(cur_x, cur_y, rotation_deg)
        x_rot.append(xr)
        y_rot.append(yr)
    return np.array(x_rot), np.array(y_rot)

class NeuronPlotter:
    def __init__(self, data, segments=None, cmap=None):
        self.data = data
        self.segments = segments
        self.segs_coords = {}
        self.__segs_to_3d()
        self.cmap = cmap
        self.soma_coords = {}
        self.__set_soma()

    def __segs_to_3d(self):
        h.define_shape()
        sections = list(self.data.all)
        for sec in sections:
            self.segs_coords.update(self.__section_pts(sec))

    def get_seg_coord(self, segment_id, x0=0, y0=0, x_scaling=1, y_scaling=1, rotation=0):
        if segment_id == -1:
            xs, ys = list(self.soma_coords.values())[0]
        else:
            segment = self.segments[segment_id]
            xs, ys = self.segs_coords[str(segment)]
        xs = np.array(xs)
        ys = np.array(ys)
        xs, ys = rotate_vecs(xs, ys, rotation)

        xs = x0 + xs * x_scaling
        ys = y0 + ys * y_scaling
        return xs, ys

    def __set_soma(self):
        for sec in self.data.soma:
            self.soma_coords.update(self.__section_pts(sec))

    def __section_pts(self, section):
        all_seg_pts = _segment_3d_pts(section)
        return {str(seg): (xs, ys) for seg, (xs, ys, _, _, _) in zip(section, all_seg_pts)}

    def mark(self, segment, ax=None, marker='o', **kwargs):
        """plot a marker on a segment

        Args:
            segment = the segment to mark
            marker = matplotlib marker
            **kwargs = passed to matplotlib's plot
        """

        x, y = self.segs_coords[str(segment)]
        ax = plt.gca() if ax is None else ax
        ax.plot([x], [y], marker=marker, **kwargs)
        return self

    def mark_point(self, segment, jitter=0, ax=None, marker='o', **kwargs):
        """plot a marker on a segment

        Args:
            segment = the segment to mark
            marker = matplotlib marker
            **kwargs = passed to matplotlib's plot
        """

        x, y = self.segs_coords[str(segment)]
        ax = plt.gca() if ax is None else ax
        x = np.mean(x) + np.random.uniform(-jitter, jitter)
        y = np.mean(y) + np.random.uniform(-jitter, jitter)
        ax.scatter(x, y, marker=marker, **kwargs)
        return self

    def plot_shape(self, allSegsV=None, somaV=None, ax=None, allSegsColor=None, somaColor=None,
                    x0=0, y0=0, x_scaling=1, y_scaling=1, rotation=0, realistic_width=True, soma_scatter=True, soma_diam_factor=10,
                     **kwargs):
        """
        Plots the morphology of the neuron. If allSegsV or somaV are give,
        colours the segments according to its voltage.
        :param allSegsV: a list of all the segments' voltage.
        :param somaV: the voltage in the soma[0](0.1)
        :param kwargs: any plt.plot arguments you want
        :return: None
        """
        ax = plt.gca() if ax is None else ax

        if allSegsV is not None:
            allSegsColor = []
            for segV in allSegsV:
                allSegsColor.append(self.cmap[int(segV) - MIN_CM])

        kwargs_color = kwargs.pop('color', 'grey')
        kwargs_linewidth = kwargs.pop('linewidth', 1)

        if allSegsColor is None:
            allSegsColor = np.array([kwargs_color]*len(self.segments))

        for seg, segColor in zip(self.segments, allSegsColor):
            xs, ys = self.segs_coords[str(seg)]
            xs = np.array(xs)
            ys = np.array(ys)
            xs, ys = rotate_vecs(xs, ys, rotation)
            linewidth = seg.diam * kwargs_linewidth if realistic_width else kwargs_linewidth
            ax.plot(x0 + xs*x_scaling, y0 + ys*y_scaling, linewidth=linewidth, color=segColor, **kwargs)

        if somaV is not None:
            somaColor = self.cmap[int(somaV) - MIN_CM]

        if somaColor is None:
            somaColor = kwargs_color

        for seg, (xs, ys) in self.soma_coords.items():
            xs = np.array(xs)
            ys = np.array(ys)
            xs, ys = rotate_vecs(xs, ys, rotation)
            # TODO: make this work
            # diam = seg.diam
            diam = self.data.soma[0](0.5).diam * soma_diam_factor
            linewidth = diam * kwargs_linewidth if realistic_width else kwargs_linewidth
            if soma_scatter:
                ax.scatter(x0 + xs.mean()*x_scaling, y0 + ys.mean()*y_scaling, s=linewidth, color=somaColor, **kwargs)
            else:
                ax.plot(x0 + xs*x_scaling, y0 + ys*y_scaling, linewidth=linewidth, **kwargs)
