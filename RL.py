import gym
import torch
import torch.optim as optim
import multiprocessing as mp
from dnq import DQN
from train import train
from mem import Mem

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
