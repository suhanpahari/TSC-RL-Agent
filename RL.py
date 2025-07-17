import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing as mp
import random
import cv2
from collections import deque

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

class Mem:
    def __init__(self, c):
        self.q = deque(maxlen=c)
    
    def add(self, t):
        self.q.append(t)
    
    def smp(self, b):
        return random.sample(self.q, b)
    
    def __len__(self):
        return len(self.q)

def train(policy, q, q_t, opt, mem, env, d, output_q, ep=50000):
    g, g_ = 0.99, 1e-4
    b, cap, tgt_u = 32, 100000, 1000
    eps, eps_, eps_d = 1.0, 0.1, 1000000
    temp, c = 1.0, 2.0  # Parameters for Boltzmann and UCB
    visit_counts = np.zeros(env.action_space.n)
    step = 0
    res = []
    for e in range(ep):
        s = pre(env.reset()[0])
        stk = Stack()
        s = stk.rst(s)
        done, r_tot = False, 0
        while not done:
            step += 1
            eps = max(eps_, eps - step / eps_d)
            if policy == 'eps_greedy':
                if random.random() < eps:
                    a_ = env.action_space.sample()
                else:
                    with torch.no_grad():
                        a_ = q(torch.tensor(s, dtype=torch.float32, device=d).unsqueeze(0)).argmax().item()
            elif policy == 'boltzmann':
                q_vals = q(torch.tensor(s, dtype=torch.float32, device=d).unsqueeze(0))
                prob = torch.softmax(q_vals / temp, dim=1).cpu().numpy().flatten()
                a_ = np.random.choice(len(prob), p=prob)
            elif policy == 'ucb':
                q_vals = q(torch.tensor(s, dtype=torch.float32, device=d).unsqueeze(0)).cpu().numpy().flatten()
                ucb_vals = q_vals + c * np.sqrt(np.log(step + 1) / (1 + visit_counts))
                a_ = np.argmax(ucb_vals)
                visit_counts[a_] += 1
            elif policy == 'thompson':
                q_vals = q(torch.tensor(s, dtype=torch.float32, device=d).unsqueeze(0)).cpu().detach().numpy().flatten()
                a_ = np.argmax(np.random.normal(q_vals, 1.0))
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
        res.append([e, r_tot, eps if policy == 'eps_greedy' else temp])
    output_q.put((policy, res))

def main():
    policies = ['eps_greedy', 'boltzmann', 'ucb', 'thompson']
    envs = [gym.make("PongNoFrameskip-v4") for _ in policies]
    d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_q = mp.Queue()
    procs = []
    for i, policy in enumerate(policies):
        q, q_t = DQN(4, envs[i].action_space.n).to(d), DQN(4, envs[i].action_space.n).to(d)
        q_t.load_state_dict(q.state_dict())
        q_t.eval()
        opt, mem = optim.Adam(q.parameters(), lr=1e-4), Mem(100000)
        p = mp.Process(target=train, args=(policy, q, q_t, opt, mem, envs[i], d, output_q))
        procs.append(p)
        p.start()
    for p in procs:
        p.join()
    results = {policy: output_q.get() for policy in policies}
    print("Epoch\t", "\t".join([f"{p} R\t{p} eps/T" for p in policies]))
    for i in range(len(results['eps_greedy'][1])):
        row = [str(i)] + [f"{results[p][1][i][1]:.2f}\t{results[p][1][i][2]:.4f}" for p in policies]
        print("\t".join(row))

if __name__ == "__main__":
    main()
#oky
