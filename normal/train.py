import gym, random, pickle, os.path, math, glob
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from atari_wrappers import make_atari, wrap_deepmind, LazyFrames
from tensorboardX import SummaryWriter


# Hyper Parameters
taskname = 'BreakoutNoFrameskip-v4'
env = make_atari(taskname)
env = wrap_deepmind(env, frame_stack=True)#把彩色图像预处理成84*84的灰度图像
batch_size = 32
memory_size = 1000000
network_updata_fre = 10000
gamma = 0.99  #discount factor
initial_epsilon = 1
final_epsilon = 0.0001
eps_decay = 10000 #episilon衰减的帧数
frames = 100000000
learning_rate = 0.00025
print_interval = 1000
learning_start = 1000000
USE_CUDA = True
action_dim = env.action_space.n
state_channel = env.observation_space.shape[2]


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(state_channel, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, action_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.contiguous().view(x.size(0), -1)))
        return self.fc5(x)


class Memory_Buffer(object):
    def __init__(self):
        self.buffer = []
        self.next_idx = 0

    def push(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        if len(self.buffer) <= memory_size:
            self.buffer.append(data)
        else:
            self.buffer[self.next_idx] = data
        self.next_idx = (self.next_idx + 1) % memory_size

    def sample(self):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in range(batch_size):
            idx = random.randint(0, self.size() - 1)
            data = self.buffer[idx]
            state, action, reward, next_state, done = data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        return np.concatenate(states), actions, rewards, np.concatenate(next_states), dones

    def size(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self):
        self.memory_buffer = Memory_Buffer()
        self.DQN = DQN()
        self.DQN_target = DQN()
        self.DQN_target.load_state_dict(self.DQN.state_dict())

        self.USE_CUDA = USE_CUDA
        if USE_CUDA:
            self.DQN = self.DQN.cuda()
            self.DQN_target = self.DQN_target.cuda()
        self.optimizer = optim.RMSprop(self.DQN.parameters(), lr=learning_rate, eps=0.001, alpha=0.95)

    def observe(self, lazyframe):
        state = torch.from_numpy(lazyframe._force().transpose(2, 0, 1)[None] / 255).float()
        if self.USE_CUDA:
            state = state.cuda()
        return state

    def value(self, state):
        q_values = self.DQN(state)
        return q_values

    def act(self, state, epsilon):
        q_values = self.value(state).cpu().detach().numpy()
        if random.random() < epsilon:
            aciton = random.randrange(action_dim)
        else:
            aciton = q_values.argmax(1)[0]
        return aciton

    def compute_td_loss(self, states, actions, rewards, next_states, is_done):
        actions = torch.tensor(actions).long()
        rewards = torch.tensor(rewards, dtype=torch.float)
        is_done = torch.tensor(is_done, dtype=torch.uint8)

        if self.USE_CUDA:
            actions = actions.cuda()
            rewards = rewards.cuda()
            is_done = is_done.cuda()

        predicted_qvalues = self.DQN(states)
        predicted_qvalues_for_actions = predicted_qvalues[range(states.shape[0]), actions]
        predicted_next_qvalues = self.DQN_target(next_states)
        next_state_values = predicted_next_qvalues.max(-1)[0]
        target_qvalues_for_actions = rewards + gamma * next_state_values
        target_qvalues_for_actions = torch.where(is_done, rewards, target_qvalues_for_actions)
        loss = F.smooth_l1_loss(predicted_qvalues_for_actions, target_qvalues_for_actions.detach())
        return loss

    def sample_from_buffer(self):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in range(batch_size):
            idx = random.randint(0, self.memory_buffer.size() - 1)
            data = self.memory_buffer.buffer[idx]
            frame, action, reward, next_frame, done = data
            states.append(self.observe(frame))
            actions.append(action)
            rewards.append(reward)
            next_states.append(self.observe(next_frame))
            dones.append(done)
        return torch.cat(states), actions, rewards, torch.cat(next_states), dones

    def learn_from_experience(self):
        if self.memory_buffer.size() > batch_size:
            states, actions, rewards, next_states, dones = self.sample_from_buffer()
            td_loss = self.compute_td_loss(states, actions, rewards, next_states, dones)
            self.optimizer.zero_grad()
            td_loss.backward()
            for param in self.DQN.parameters():
                param.grad.data.clamp_(-1, 1)

            self.optimizer.step()
            return (td_loss.item())
        else:
            return (0)


epsilon_by_frame = lambda frame_idx: final_epsilon + (initial_epsilon - final_epsilon) * math.exp(-1. * frame_idx / eps_decay)


def main():
    agent = DQNAgent()
    frame = env.reset()
    episode_reward = 0
    all_rewards = []
    losses = []
    episode_num = 0
    summary_writer = SummaryWriter(log_dir="DQN_stackframe")
    # agent.DQN.load_state_dict(torch.load("trained model/DQN_dict7170.pth"))
    # agent.DQN_target.load_state_dict(torch.load("trained model/DQN_dict7170.pth"))

    for i in range(frames):
        env.render()  # 显示游戏画面
        epsilon = epsilon_by_frame(i)
        state_tensor = agent.observe(frame)
        action = agent.act(state_tensor, epsilon)
        next_frame, reward, done, _ = env.step(action)
        episode_reward += reward
        agent.memory_buffer.push(frame, action, reward, next_frame, done)
        frame = next_frame

        loss = 0
        if agent.memory_buffer.size() >= learning_start:
            loss = agent.learn_from_experience()
            losses.append(loss)

        if i % print_interval == 0:
            print("frames: %5d, reward: %5f, loss: %4f, epsilon: %5f, episode: %4d" %
                  (i, np.mean(all_rewards[-10:]), loss, epsilon, episode_num))
            summary_writer.add_scalar("Temporal Difference Loss", loss, i)
            summary_writer.add_scalar("Mean Reward", np.mean(all_rewards[-10:]), i)
            summary_writer.add_scalar("Epsilon", epsilon, i)

        if i % network_updata_fre == 0:
            agent.DQN_target.load_state_dict(agent.DQN.state_dict())

        if done:
            frame = env.reset()

            all_rewards.append(episode_reward)
            episode_reward = 0
            episode_num += 1
            avg_reward = float(np.mean(all_rewards[-100:]))
            if avg_reward > 15:
                torch.save(agent.DQN.state_dict(), "trained model/DQN_dict{}.pth".format(episode_num))


if __name__ == '__main__':
    main()