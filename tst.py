import gym
import numpy as np
import cv2
import torch
import random
import torch.nn as nn
import torch.optim as optim
from collections import deque

def pre(f):
    f = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
    f = cv2.resize(f, (84, 84))
    return f / 255.0

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

class DQN(nn.Module):
    def __init__(self, d, a):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(d, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(nn.Linear(3136, 512), nn.ReLU(), nn.Linear(512, a))
    
    def forward(self, x):
        return self.fc(self.conv(x))

class Mem:
    def __init__(self, c):
        self.q = deque(maxlen=c)
    
    def add(self, t):
        self.q.append(t)
    
    def smp(self, b):
        return random.sample(self.q, b)
    
    def __len__(self):
        return len(self.q)

g, g_ = 0.99, 1e-4
eps, eps_, eps_d, b, cap, tgt_u, ep = 1.0, 0.1, 1000000, 32, 100000, 1000, 50000
env = gym.make("PongNoFrameskip-v4")
a = env.action_space.n
d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
q, q_t = DQN(4, a).to(d), DQN(4, a).to(d)
q_t.load_state_dict(q.state_dict())
q_t.eval()
opt, mem = optim.Adam(q.parameters(), lr=g_), Mem(cap)
step = 0

for e in range(ep):
    s = pre(env.reset()[0])
    stk = Stack()
    s = stk.rst(s)
    done, r_tot = False, 0
    while not done:
        step += 1
        eps = max(eps_, eps - step / eps_d)
        if random.random() < eps:
            a_ = env.action_space.sample()
        else:
            with torch.no_grad():
                a_ = q(torch.tensor(s, dtype=torch.float32, device=d).unsqueeze(0)).argmax().item()
        s_, r, done, _, _ = env.step(a_)
        s_ = pre(s_)
        s_ = stk.upd(s_)
        r_tot += r
        mem.add((s, a_, r, s_, done))
        s = s_
        if len(mem) > b:
            smp = mem.smp(b)
            ss, aa, rr, ss_, dd = zip(*smp)
            ss, aa, rr, ss_, dd = torch.tensor(ss, dtype=torch.float32, device=d), torch.tensor(aa, dtype=torch.long, device=d).unsqueeze(1), torch.tensor(rr, dtype=torch.float32, device=d), torch.tensor(ss_, dtype=torch.float32, device=d), torch.tensor(dd, dtype=torch.float32, device=d)
            q_vals = q(ss).gather(1, aa).squeeze()
            q_next = q_t(ss_).max(1)[0].detach()
            q_tgt = rr + g * q_next * (1 - dd)
            loss = nn.MSELoss()(q_vals, q_tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if step % tgt_u == 0:
            q_t.load_state_dict(q.state_dict())
    print(f"Ep {e}, R: {r_tot}, Eps: {eps:.4f}")
env.close()