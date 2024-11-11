import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random

# Use namedtuple for better memory efficiency and clarity
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class QRDQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, num_quantiles, hidden_dims=[256, 256]):
        super(QRDQNNetwork, self).__init__()
        self.num_quantiles = num_quantiles
        self.action_dim = action_dim

        # Build network layers without batch normalization
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)  # Keep dropout for regularization
            ])
            prev_dim = hidden_dim

        # Final layer
        self.feature_layers = nn.Sequential(*layers)
        self.quantile_layer = nn.Linear(prev_dim, action_dim * num_quantiles)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            module.bias.data.zero_()

    def forward(self, state):
        # Ensure proper dimensions and device
        if len(state.shape) == 1:
            state = state.unsqueeze(0)  # Add batch dimension if needed

        features = self.feature_layers(state)
        quantiles = self.quantile_layer(features)
        return quantiles.view(-1, self.action_dim, self.num_quantiles)

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization to use (0 = none, 1 = full)
        self.beta = beta    # Importance sampling correction
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0

    def add(self, transition):
        max_priority = np.max(self.priorities) if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None, None, None

        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Calculate sampling probabilities
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        # Sample indices and calculate importance weights
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        weights = (len(self.buffer) * probabilities[indices]) ** -self.beta
        weights /= weights.max()

        batch = [self.buffer[idx] for idx in indices]
        return batch, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6  # Small constant to ensure non-zero priority

class QRDQN:
    def __init__(self, state_dim, action_dim, num_quantiles=51, learning_rate=0.0005,
                 gamma=0.99, buffer_size=100000, batch_size=64, tau=0.005):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau  # Soft update parameter

        # Networks
        self.network = QRDQNNetwork(state_dim, action_dim, num_quantiles).to(self.device)
        self.target_network = QRDQNNetwork(state_dim, action_dim, num_quantiles).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())

        # Optimizer with gradient clipping
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        # Replay buffer with prioritization
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size)

        # Quantile thresholds
        self.tau_hat = torch.linspace(0, 1, num_quantiles + 1)[:-1] + 0.5 / num_quantiles
        self.tau_hat = self.tau_hat.to(self.device)

        # Training info
        self.training_steps = 0
        self.epsilon = 1.0

    def huber_loss(self, td_errors, kappa=1.0):
        return torch.where(
            td_errors.abs() <= kappa,
            0.5 * td_errors.pow(2),
            kappa * (td_errors.abs() - 0.5 * kappa)
        )

    def compute_td_error(self, current_q, target_q):
        td_errors = target_q.unsqueeze(1) - current_q.unsqueeze(-1)
        return td_errors

    def train_step(self):
        result = self.replay_buffer.sample(self.batch_size)
        if result is None:
            return

        batch, indices, weights = result
        batch = Transition(*zip(*batch))

        # Convert to torch tensors
        states = torch.FloatTensor(np.array(batch.state)).to(self.device)
        if len(states.shape) == 1:
            states = states.unsqueeze(0)
        actions = torch.LongTensor(batch.action).to(self.device)
        rewards = torch.FloatTensor(batch.reward).to(self.device)
        next_states = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        if len(next_states.shape) == 1:
            next_states = next_states.unsqueeze(0)
        dones = torch.FloatTensor(batch.done).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        # Get current Q values
        current_quantiles = self.network(states)
        current_quantiles = current_quantiles[range(self.batch_size), actions]  # Shape: [batch_size, num_quantiles]

        # Get next Q values
        with torch.no_grad():
            next_quantiles = self.target_network(next_states)
            next_actions = next_quantiles.mean(dim=2).argmax(dim=1)
            next_quantiles = next_quantiles[range(self.batch_size), next_actions]

            # Reshape rewards and dones for broadcasting
            rewards = rewards.unsqueeze(1).expand(-1, self.num_quantiles)
            dones = dones.unsqueeze(1).expand(-1, self.num_quantiles)

            target_quantiles = rewards + (1 - dones) * self.gamma * next_quantiles

        # Compute TD errors
        td_errors = target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2)

        # Compute quantile Huber loss
        huber_loss = self.huber_loss(td_errors)  # Shape: [batch_size, num_quantiles, num_quantiles]

        # Compute quantile loss
        tau = self.tau_hat.view(1, -1, 1).expand(self.batch_size, self.num_quantiles, self.num_quantiles)
        quantile_loss = (tau - (td_errors < 0).float()).abs() * huber_loss

        # Apply importance sampling weights and take mean
        weights = weights.unsqueeze(1).unsqueeze(2).expand_as(quantile_loss)
        loss = (weights * quantile_loss).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()

        # Soft update target network
        self._soft_update_target_network()

        # Update priorities in replay buffer
        with torch.no_grad():
            td_errors_for_priority = td_errors.abs().mean(dim=(1, 2)).cpu().numpy()
            self.replay_buffer.update_priorities(indices, td_errors_for_priority)

        self.training_steps += 1

    def _soft_update_target_network(self):
        """Soft update target network parameters"""
        for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def select_action(self, state, epsilon=None):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            quantiles = self.network(state_tensor)
            q_values = quantiles.mean(dim=2)

            # Use softmax with temperature for action selection
            temperature = 0.1
            probs = torch.softmax(q_values / temperature, dim=1)
            action = torch.multinomial(probs, 1).item()

            return action

    def add_to_replay_buffer(self, transition):
        self.replay_buffer.add(transition)
