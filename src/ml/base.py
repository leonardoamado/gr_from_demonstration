from collections import deque


class BaseMethod:
    def __init__(self):
        self.curr_state = deque(maxlen=4)
        self.next_state = deque(maxlen=4)
