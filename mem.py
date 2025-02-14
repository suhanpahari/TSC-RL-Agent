
import random
from collections import deque

class Mem:
    def __init__(self, c):
        self.q = deque(maxlen=c)
    
    def add(self, t):
        self.q.append(t)
    
    def smp(self, b):
        return random.sample(self.q, b)
    
    def __len__(self):
        return len(self.q)