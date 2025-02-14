import numpy as np
from collections import deque

class Stack:
    def __init__(self, n=4):
        self.n = n
        self.q = deque(maxlen=n)
    
    def rst(self, f):
        self.q.clear()
        for _ in range(self.n):
            self.q.append(f)
        return np.stack(self.q, axis=0)
    
    def upd(self, f):
        self.q.append(f)
        return np.stack(self.q, axis=0)