#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vispy: gallery 2
""" Simple example plotting 2D points.
"""

from vispy import gloo
from vispy import app
import numpy as np
import vispy.io as io

VERT_SHADER = """
attribute vec2  a_position;
attribute vec3  a_color;
attribute float a_size;

varying vec4 v_fg_color;
varying vec4 v_bg_color;
varying float v_radius;
varying float v_linewidth;
varying float v_antialias;

void main (void) {
    v_radius = a_size;
    v_linewidth = 1.0;
    v_antialias = 1.0;
    v_fg_color  = vec4(0.0,0.0,0.0,0.0);
    v_bg_color  = vec4(a_color,    1.0);

    gl_Position = vec4(a_position, 0.0, 1.0);
    gl_PointSize = 2.0*(v_radius + v_linewidth + 1.5*v_antialias);
}
"""

FRAG_SHADER = """
#version 120

varying vec4 v_fg_color;
varying vec4 v_bg_color;
varying float v_radius;
varying float v_linewidth;
varying float v_antialias;
void main()
{
    float size = 2.0*(v_radius + v_linewidth + 1.5*v_antialias);
    float t = v_linewidth/2.0-v_antialias;
    float r = length((gl_PointCoord.xy - vec2(0.5,0.5))*size);
    float d = abs(r - v_radius) - t;
    if( d < 0.0 )
        gl_FragColor = v_fg_color;
    else
    {
        float alpha = d/v_antialias;
        alpha = exp(-alpha*alpha);
        if (r > v_radius)
            gl_FragColor = vec4(v_fg_color.rgb, alpha*v_fg_color.a);
        else
            gl_FragColor = mix(v_bg_color, v_fg_color, alpha);
    }
}
"""


class Canvas(app.Canvas):
    def __init__(self, prime_numbers):
        app.Canvas.__init__(self, keys='interactive')
        self.prime_numbers = prime_numbers

        # Create vertices
        v_position, v_color, v_size = self.get_vertices()

        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        # Set uniform and attribute
        self.program['a_color'] = gloo.VertexBuffer(v_color)
        self.program['a_position'] = gloo.VertexBuffer(v_position)
        self.program['a_size'] = gloo.VertexBuffer(v_size)
        gloo.set_state(clear_color='white', blend=True,
                       blend_func=('src_alpha', 'one_minus_src_alpha'))

        img = self.render()
        io.write_png('render.png', img)

        self.show()

    def get_vertices(self):
        n = self.prime_numbers.n
        no_prime_color = (1.0, 1.0, 1.0)
        prime_color = (0.0, 0.0, 0.0)
        ps = self.pixel_scale
        scale = 0.005

        # v_position = 0.25 * np.random.randn(n, 2).astype(np.float32)
        v_position = self.get_spiral_points(n) * scale

        v_color = np.array([prime_color if s else no_prime_color for s in self.prime_numbers.sieve]).astype(np.float32)

        # v_size = np.random.uniform(2*ps, 12*ps, (n, 1)).astype(np.float32)
        v_size = np.ones((n, 1), dtype=np.float32) * 1.5 * ps
        return v_position, v_color, v_size

    @staticmethod
    def get_spiral_points(n):
        i = np.arange(n).astype(np.float32)
        i_sqrt = np.sqrt(i)
        x = np.cos(i_sqrt * 2 * np.pi) * i_sqrt
        y = np.sin(i_sqrt * 2 * np.pi) * i_sqrt
        return np.array([x, y]).T

    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.physical_size)

    def on_draw(self, event):
        gloo.clear(color=True, depth=True)
        self.program.draw('points')


class PrimeNumbers:
    def __init__(self, n):
        self.n = n
        primes = self.rwh_primes1(self.n)
        self.sieve = np.zeros(n, dtype=bool)
        self.sieve[primes] = 1

    @staticmethod
    def rwh_primes1(n):
        # https://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n-in-python/3035188#3035188
        """ Returns  a list of primes < n """
        sieve = [True] * (n // 2)
        for i in range(3, int(n ** 0.5) + 1, 2):
            if sieve[i // 2]:
                sieve[i * i // 2::i] = [False] * ((n - i * i - 1) // (2 * i) + 1)
        return [2] + [2 * i + 1 for i in range(1, n // 2) if sieve[i]]


if __name__ == '__main__':
    n = 100000
    p = PrimeNumbers(n)
    c = Canvas(p)
    app.run()
