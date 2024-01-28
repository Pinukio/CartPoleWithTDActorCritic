# 바닥부터 배우는 강화 학습 P.240 TD Actor-Critic 구현

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# HyperParameters
learning_rate = 0.0002
# discount factor
gamma = 0.98
# TD 식으로 하되 10틱에 한 번 업데이트함
n_rollout = 10

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(4, 256)
        # Policy Network
        self.fc_pi = nn.Linear(256, 2)
        # Value Network
        self.fc_v = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    # Policy Network Forward Propagation
    # Probability Distirbution을 출력함
    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    # Value Network Forward Propagation
    # State Value를 출력함
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
    
    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_list, a_list, r_list, s_prime_list, done_list = [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, done = transition
            s_list.append(s)
            a_list.append([a])
            r_list.append([r/100.0])
            s_prime_list.append(s_prime)
            
            done_mask = 0.0 if done else 1.0
            done_list.append([done_mask])

        s_batch, a_batch, r_batch, s_prime_batch, done_batch = torch.tensor(s_list, dtype=torch.float), torch.tensor(a_list), torch.tensor(r_list, dtype=torch.float), torch.tensor(s_prime_list, dtype=torch.float), torch.tensor(done_list, dtype=torch.float)
        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch
    
    def train_net(self):
        s, a, r, s_prime, done = self.make_batch()
        td_target = r + gamma * self.v(s_prime) * done
        delta = td_target - self.v(s)

        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1,a)
        # Loss로 분기된 두 NN의 Loss를 더해주고 있음
        # detach는 상수 취급하기 위해 붙는 것이다. NN의 학습에 집중하기 위해서이다.
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())
        
        self.optimizer.zero_grad()
        # Gradient 계산
        loss.mean().backward()
        # 계산된 Gradient로 Gradient Descent(Policy Network는 -를 붙여 Gradient Ascent, Value Network는 Descent)
        self.optimizer.step()

def main():
    env = gym.make('CartPole-v1')
    model = ActorCritic()
    print_interval = 20
    score = 0.0

    for n_epi in range(10000):
        done = False
        s = env.reset()[0]
        while not done:
            for t in range(n_rollout):
                prob = model.pi(torch.from_numpy(s).float())
                # Categorical.sample()은 확률 분포에 의한 Sampling을 진행해 준다.
                # Policy에 의한 Action Sampling
                m = Categorical(prob)
                a = m.sample().item()
                
                s_prime, r, done, trun, _ = env.step(a)
                model.put_data((s,a,r,s_prime,done))

                s = s_prime
                score += r

                if done:
                    break

            model.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode: {}, avg score: {}".format(n_epi, score/print_interval))
            score = 0.0
            
        env.close()