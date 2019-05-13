import numpy as np


class GrayscaleCanvasFiller:

    def __init__(self):
        self.cmap = 'gray'

    def build_canvas(self, images, fig_size, image_size):
        n_cols_images, n_rows_images = fig_size
        IMAGE_DIM_X, IMAGE_DIM_Y = image_size
        canvas = np.empty((IMAGE_DIM_Y*n_rows_images, IMAGE_DIM_X*n_cols_images))

        index = 0
        for j in range(n_rows_images):
            start_y = j*IMAGE_DIM_Y
            end_y = start_y + IMAGE_DIM_Y
            for i in range(n_cols_images):
                start_x = i*IMAGE_DIM_X
                end_x = start_x + IMAGE_DIM_X
                canvas[start_y:end_y, start_x:end_x] = images[index].reshape(IMAGE_DIM_X, IMAGE_DIM_Y)
                index += 1

        return canvas


class RgbCanvasFiller:

    def __init__(self):
        self.cmap = None

    def build_canvas(self, images, fig_size, image_size):
        n_cols_images, n_rows_images = fig_size
        IMAGE_DIM_X, IMAGE_DIM_Y, IMAGE_DIM_Z = image_size

        canvas = np.empty((IMAGE_DIM_Y*n_rows_images, IMAGE_DIM_X*n_cols_images, IMAGE_DIM_Z))

        index = 0
        for j in range(n_rows_images):
            start_y = j*IMAGE_DIM_Y
            end_y = start_y + IMAGE_DIM_Y
            for i in range(n_cols_images):
                start_x = i*IMAGE_DIM_X
                end_x = start_x + IMAGE_DIM_X
                canvas[start_y:end_y, start_x:end_x, :] = images[index]
                index += 1

        return canvas
