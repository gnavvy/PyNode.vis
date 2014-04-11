__author__ = 'ywang'

import numpy as np
import pyopencl as cl
from pyopencl.array import vec
from sklearn.neighbors import NearestNeighbors

class Accelerator:

    def __init__(self):
        self.platform = cl.get_platforms()[0]
        self.device = self.platform.get_devices()[0]
        self.context = cl.Context([self.device])
        self.flags = cl.mem_flags

    @staticmethod
    def _get_exact_ranks(data):
        nn = NearestNeighbors(n_neighbors=7).fit(data)
        rank_matrix = nn.kneighbors(data, return_distance=False)
        return rank_matrix[:, 1:]  # remove first column -> self index

    def get_seed_values(self, seeds, embedding, selected):
        rank_matrices = []
        for s in seeds:
            embedding[selected] = s
            rank_matrices[s] = self._get_exact_ranks(embedding)
        rank_matrices = np.array(rank_matrices, dtype=cl.array.vec.float4)

        print(rank_matrices.shape)

        # print("getting seed data")
        # if selected is None:
        #     selected = np.argmin(self.acrm.evals)
        # self.seed_values = [self.acrm.update_data_low(selected, s) for s in seeds]
        # self.seed_values = self._normalize(np.array(self.seed_values))
        # return self.seed_values

    def dummy_calculation(self):
        vector = np.zeros((1, 1), cl.array.vec.float4)
        matrix = np.zeros((1, 4), cl.array.vec.float4)
        result = np.zeros(4, np.float32)  # result

        matrix[0, 0] = (1, 2, 4, 8)
        matrix[0, 1] = (16, 32, 64, 128)
        matrix[0, 2] = (3, 6, 9, 12)
        matrix[0, 3] = (5, 10, 15, 25)
        vector[0, 0] = (1, 2, 4, 8)

        program = cl.Program(self.context, """
            __kernel void matrix_dot_vector(__global const float4 *matrix,
                                            __global const float4 *vector,
                                            __global float *result) {
                int gid = get_global_id(0);
                result[gid] = dot(matrix[gid], vector[0]);
            }
            """).build()

        queue = cl.CommandQueue(self.context)
        # host -> device
        matrix_buf = cl.Buffer(self.context, self.flags.READ_ONLY | self.flags.COPY_HOST_PTR, hostbuf=matrix)
        vector_buf = cl.Buffer(self.context, self.flags.READ_ONLY | self.flags.COPY_HOST_PTR, hostbuf=vector)
        result_buf = cl.Buffer(self.context, self.flags.WRITE_ONLY, result.nbytes)

        program.matrix_dot_vector(queue, result.shape, None, matrix_buf, vector_buf, result_buf)
        # device <- host
        cl.enqueue_copy(queue, result, result_buf)

        print(result)

if __name__ == "__main__":
    acc = Accelerator()
    # acc.dummy_calculation()
    acc.get_seed_values()   # TODO