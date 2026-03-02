import numpy as np


class SimpleEncoder(): # Maps a scalar value to a contigous encoding
    def __init__(self, dims, encoding_width, num_buckets):
        self.dims = dims
        self.encoding_width = encoding_width
        self.num_buckets = num_buckets


        self.encoding = np.zeros(dims)


    def encode(self, value):
        flat_encoding = self.encoding.flatten()
        flat_dims = flat_encoding.shape[0]


        encoding_start = (value * flat_dims/self.num_buckets) % flat_dims

        flat_encoding[0:self.encoding_width] = 1
        flat_encoding = np.roll(flat_encoding, encoding_start)

        self.encoding = flat_encoding.reshape(self.dims)

        return self.encoding