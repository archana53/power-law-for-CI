from manim import *
import numpy as np

class MovingAround(Scene):
    def construct(self):
        mnist_sample = np.load("example_18_lbl_6.npy")
        img = ImageMobject((255*mnist_sample))
        img.height = 3

        np.random.seed(10)
        permutation = np.random.permutation(28**2)

        row_major = mnist_sample.reshape(28**2)
        permuted = row_major[permutation]

        permuted_image = permuted.reshape((28,28))

        img2 = ImageMobject((255*permuted_image))
        img2.height = 3
        # img2.shift(5*RIGHT)

        square = Square(color=BLUE, fill_opacity=1)
        arrow = Arrow([-3, 0, 0], [2, 0, 0], buff=0)
        # arrow2 = Arrow(ORIGIN, [2, -2, 0], buff=0)
        b1 = Brace(arrow, direction=arrow.copy().rotate(PI / 2).get_unit_vector())
        b1text = b1.get_text("Apply Permutation")


        self.play(img.animate.shift(4*LEFT))

        self.play(img2.animate.shift(4*RIGHT))
        # self.add(img2)

        self.play(GrowArrow(arrow))
        self.add(b1, b1text)

        self.wait(2)