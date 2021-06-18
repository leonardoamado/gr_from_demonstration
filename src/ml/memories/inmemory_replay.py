import numpy as np

from .replay import Replay


class InMemoryReplay(Replay):
    def __init__(self, size, input_shape):
        shape = (size,) + input_shape
        # data structure: State x Action x Next State x Reward x Terminal
        self.s = np.zeros(shape, dtype='float32')
        self.s_p = np.zeros(shape, dtype='float32')
        self.a = np.zeros(size, dtype='int32')
        self.r = np.zeros(size, dtype='float32')
        self.t = np.zeros(size, dtype='float32')
        self.curr = 0
        self.max_size = size

    def add_transition(self, s, a, s_p, r, t):
        s = np.squeeze(s)
        s_p = np.squeeze(s_p)
        i = self.curr % self.max_size
        self.s[i] = s
        self.a[i] = a
        self.s_p[i] = s_p
        self.r[i] = r
        self.t[i] = t
        # self.curr = min(self.curr + 1, self.max_size)
        self.curr += 1
        if self.curr % 10000 == 0:
            print(f'Current memory size: {self.curr}')

    def get_batch(self, batch_size):
        if self.curr < batch_size:
            return None
        i = np.random.randint(0, min(self.curr, self.max_size), size=(batch_size,))
        return self.s[i], self.a[i], self.s_p[i], self.r[i], self.t[i]
