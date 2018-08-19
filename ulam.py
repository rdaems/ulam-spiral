import numpy as np
import vispy.plot as vp


def main():
    points = np.random.randn(10, 2)

    fig = vp.Fig(show=False)
    fig[0, 0].plot(points, width=0, face_color=(.8, .4, .2, .5), edge_color=None, marker_size=100.)

    fig.show(run=True)


if __name__ == '__main__':
    main()
