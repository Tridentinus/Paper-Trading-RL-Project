import numpy as np
import matplotlib.pyplot as plt
from env import TradingEnvironmentV6
from agents import BuyAndHoldAgent, MovingAverageCrossoverAgent, RSIAgent, BollingerBandsAgent, MACDAgent, VolumeWeightedMAAgent, MeanReversionAgent
from qrdqn import QRDQN
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd
import torch
class RLModelWrapper:
    """Wrapper to make stable-baselines3 models compatible with our evaluation"""
    def __init__(self, model):
        self.model = model

    def select_action(self, state, epsilon=None):
        action, _ = self.model.predict(state, deterministic=True)
        return action

# Training configuration for different algorithms
# Cell: RL Configurations
rl_configs = {
    'QR-DQN': {
        'model_params': {
            'learning_rate': 0.0001,
            'batch_size': 128,
            'num_quantiles': 41,
            'buffer_size': 100000,
            'gamma': 0.95,
            'tau': 0.007354933110186957
        },
        'train_params': {
            'num_episodes': 1000,
            'initial_epsilon': 1.0,
            'epsilon_min': 0.05,
            'epsilon_decay': 0.995,
            'eval_frequency': 500,
            'min_samples_before_training': 1000
        }
        },
    'PPO': {
        'model_params': {
            'learning_rate': 0.0003,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'verbose': 1
        },
        'train_params': {
            'total_timesteps': 100000
        }
    },
    'A2C': {
        'model_params': {
            'learning_rate': 0.0007,
            'n_steps': 5,
            'gamma': 0.99,
            'ent_coef': 0.01,
            'verbose': 1
        },
        'train_params': {
            'total_timesteps': 100000
        }
    },
    'DQN': {
        'model_params': {
            'learning_rate': 0.0001,
            'batch_size': 32,
            'buffer_size': 100000,
            'gamma': 0.99,
            'exploration_fraction': 0.1,
            'exploration_final_eps': 0.05,
            'verbose': 1
        },
        'train_params': {
            'total_timesteps': 100000
        }
    }
}

def train_qrdqn(env, agent, num_episodes=1000, initial_epsilon=1.0,
                epsilon_min=0.05, epsilon_decay=0.995, eval_frequency=500 ,
                min_samples_before_training=1000):  # Added parameter
    """
    Train the QR-DQN agent with detailed progress tracking
    """
    print("\nStarting QR-DQN training...")
    print(f"Using device: {agent.device}")

    # Tracking metrics
    episode_rewards = []
    portfolio_values = []
    evaluation_results = []
    best_reward = float('-inf')
    total_steps = 0

    epsilon = initial_epsilon

    # Initial exploration to fill replay buffer
    print(f"\nCollecting initial experiences ({min_samples_before_training} samples)...")
    state, _ = env.reset()
    for _ in range(min_samples_before_training):
        action = agent.select_action(state, epsilon=1.0)  # Pure exploration
        next_state, reward, done, _, info = env.step(action)
        agent.add_to_replay_buffer((state, action, reward, next_state, done))

        if done:
            state, _ = env.reset()
        else:
            state = next_state

        total_steps += 1
        if total_steps % 100 == 0:
            print(f"Collected {total_steps} samples")

    print("\nStarting training...")
    try:
        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0
            episode_steps = 0

            # Episode loop
            while not done:
                # Select and perform action
                action = agent.select_action(state, epsilon)
                next_state, reward, done, _, info = env.step(action)

                # Store transition and train
                agent.add_to_replay_buffer((state, action, reward, next_state, done))

                # Only train if we have enough samples
                if total_steps >= min_samples_before_training:
                    agent.train_step()

                total_reward += reward
                state = next_state
                episode_steps += 1
                total_steps += 1

            # Track metrics
            episode_rewards.append(total_reward)
            portfolio_values.append(info['portfolio_value'])

            # Decay epsilon
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            # Progress reporting
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_portfolio = np.mean(portfolio_values[-10:])
                print(f"\nEpisode {episode + 1}/{num_episodes}")
                print(f"Steps: {episode_steps}")
                print(f"Total Steps: {total_steps}")
                print(f"Epsilon: {epsilon:.3f}")
                print(f"Recent Average Reward: {avg_reward:.2f}")
                print(f"Recent Average Portfolio Value: {avg_portfolio:.2f}")
                print(f"Training Steps: {agent.training_steps}")

                # Plot progress
                if (episode + 1) % 500 == 0:
                    plt.figure(figsize=(15, 5))

                    # Plot rewards
                    plt.subplot(1, 2, 1)
                    plt.plot(episode_rewards)
                    plt.title('Training Rewards')
                    plt.xlabel('Episode')
                    plt.ylabel('Total Reward')

                    # Plot portfolio values
                    plt.subplot(1, 2, 2)
                    plt.plot(portfolio_values)
                    plt.title('Portfolio Values')
                    plt.xlabel('Episode')
                    plt.ylabel('Portfolio Value')

                    plt.tight_layout()
                    plt.show()
            # Periodic evaluation
            if (episode + 1) % eval_frequency == 0:
                print("\nRunning evaluation...")
                eval_rewards = []
                eval_portfolios = []

                # Run evaluation episodes
                for _ in range(5):
                    state, _ = env.reset()
                    done = False
                    eval_reward = 0

                    while not done:
                        action = agent.select_action(state, epsilon=0)  # No exploration during eval
                        state, reward, done, _, info = env.step(action)
                        eval_reward += reward

                    eval_rewards.append(eval_reward)
                    eval_portfolios.append(info['portfolio_value'])

                avg_eval_reward = np.mean(eval_rewards)
                avg_eval_portfolio = np.mean(eval_portfolios)
                evaluation_results.append((episode + 1, avg_eval_reward, avg_eval_portfolio))

                print(f"\nEvaluation Results (Episode {episode + 1}):")
                print(f"Average Evaluation Reward: {avg_eval_reward:.2f}")
                print(f"Average Evaluation Portfolio: {avg_eval_portfolio:.2f}")

                # Save best model
                if avg_eval_reward > best_reward:
                    best_reward = avg_eval_reward
                    torch.save({
                        'episode': episode,
                        'model_state_dict': agent.network.state_dict(),
                        'optimizer_state_dict': agent.optimizer.state_dict(),
                        'reward': best_reward,
                    }, 'best_qrdqn_model.pth')
                    print("New best model saved!")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Final plotting
        plt.figure(figsize=(15, 10))

        # Plot training rewards
        plt.subplot(2, 2, 1)
        plt.plot(episode_rewards)
        plt.title('Training Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')

        # Plot portfolio values
        plt.subplot(2, 2, 2)
        plt.plot(portfolio_values)
        plt.title('Portfolio Values')
        plt.xlabel('Episode')
        plt.ylabel('Portfolio Value')

        # Plot evaluation results
        if evaluation_results:
            episodes, eval_rewards, eval_portfolios = zip(*evaluation_results)

            plt.subplot(2, 2, 3)
            plt.plot(episodes, eval_rewards)
            plt.title('Evaluation Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Average Evaluation Reward')

            plt.subplot(2, 2, 4)
            plt.plot(episodes, eval_portfolios)
            plt.title('Evaluation Portfolio Values')
            plt.xlabel('Episode')
            plt.ylabel('Average Portfolio Value')

        plt.tight_layout()
        plt.show()

        return episode_rewards, portfolio_values, evaluation_results
def train_all_models(train_df, test_df):
    """Train and evaluate multiple RL models"""

    # Initialize training environment
    train_env = TradingEnvironmentV6(train_df)
    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.n

    # Dictionary to store trained models
    trained_models = {}

    # Train QR-DQN
    print("\nTraining QR-DQN...")
    qrdqn_agent = QRDQN(
        state_dim=state_dim,
        action_dim=action_dim,
        **rl_configs['QR-DQN']['model_params']
    )

    rewards, portfolios, eval_results = train_qrdqn(
        env=train_env,
        agent=qrdqn_agent,
        **rl_configs['QR-DQN']['train_params']
    )
    trained_models['QR-DQN'] = qrdqn_agent

    # # Train PPO
    print("\nTraining PPO...")
    ppo_env = DummyVecEnv([lambda: train_env])
    ppo_model = PPO('MlpPolicy', ppo_env, **rl_configs['PPO']['model_params'])
    ppo_model.learn(**rl_configs['PPO']['train_params'])
    trained_models['PPO'] = RLModelWrapper(ppo_model)

    # Train A2C
    print("\nTraining A2C...")
    a2c_env = DummyVecEnv([lambda: train_env])
    a2c_model = A2C('MlpPolicy', a2c_env, **rl_configs['A2C']['model_params'])
    a2c_model.learn(**rl_configs['A2C']['train_params'])
    trained_models['A2C'] = RLModelWrapper(a2c_model)

    # Train DQN
    print("\nTraining DQN...")
    dqn_env = train_env
    dqn_model = DQN('MlpPolicy', dqn_env, **rl_configs['DQN']['model_params'])
    dqn_model.learn(**rl_configs['DQN']['train_params'])
    trained_models['DQN'] = RLModelWrapper(dqn_model)

    return trained_models


class ModelEvaluator:
    """Evaluates and compares different trading models"""

    def __init__(self, test_df, initial_balance=10000):
        self.test_df = test_df
        self.initial_balance = initial_balance
        self.metrics = {}

    def calculate_metrics(self, returns):
        """Calculate various trading metrics"""

        # Convert to numpy array if needed
        if isinstance(returns, (pd.Series, list)):
            returns = np.array(returns)

        # Basic Return Metrics
        total_return = (returns + 1).prod() - 1
        avg_return = returns.mean()

        # Risk Metrics
        volatility = returns.std() * np.sqrt(252)  # Annualized
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252)
        max_drawdown = np.min(np.minimum.accumulate(returns + 1)) - 1

        # Risk-Adjusted Return Metrics
        sharpe_ratio = (avg_return * 252) / volatility if volatility != 0 else 0
        sortino_ratio = (avg_return * 252) / downside_vol if downside_vol != 0 else 0
        calmar_ratio = -total_return / max_drawdown if max_drawdown != 0 else 0

        # Trading Metrics
        num_trades = np.sum(np.diff(returns) != 0)
        win_rate = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0
        profit_factor = abs(returns[returns > 0].sum() / returns[returns < 0].sum()) if len(returns[returns < 0]) > 0 else np.inf

        return {
            'Total Return (%)': total_return * 100,
            'Annualized Return (%)': avg_return * 252 * 100,
            'Volatility (%)': volatility * 100,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Calmar Ratio': calmar_ratio,
            'Maximum Drawdown (%)': max_drawdown * 100,
            'Number of Trades': num_trades,
            'Win Rate (%)': win_rate * 100,
            'Profit Factor': profit_factor
        }

    def evaluate_model(self, model, model_name):
        """Evaluate a single model"""
        env = TradingEnvironmentV6(self.test_df)  # Or your environment class
        state, _ = env.reset()
        done = False
        portfolio_values = [self.initial_balance]
        positions = []
        returns = []

        while not done:
            action = model.select_action(state, epsilon=0)  # No exploration during evaluation
            state, reward, done, _, info = env.step(action)

            portfolio_values.append(info['portfolio_value'])
            positions.append(info['position'])
            returns.append(reward)

        # Calculate metrics
        self.metrics[model_name] = self.calculate_metrics(np.array(returns))
        self.metrics[model_name]['Portfolio Values'] = portfolio_values
        self.metrics[model_name]['Positions'] = positions

        return self.metrics[model_name]

    def compare_models(self, models_dict):
        """Compare multiple models"""
        for name, model in models_dict.items():
            print(f"\nEvaluating {name}...")
            self.evaluate_model(model, name)

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(self.metrics).round(2)

        # Plot comparisons
        self._plot_comparisons()

        return comparison_df

    def _plot_comparisons(self):
        """Plot comparative visualizations"""
        plt.style.use('default')

        # Plot Portfolio Values
        plt.figure(figsize=(15, 10))

        # Portfolio Values
        plt.subplot(2, 2, 1)
        for name in self.metrics:
            plt.plot(self.metrics[name]['Portfolio Values'], label=name)
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Time Steps')
        plt.ylabel('Portfolio Value')
        plt.legend()

        # Returns Distribution
        plt.subplot(2, 2, 2)
        for name in self.metrics:
            returns = np.diff(self.metrics[name]['Portfolio Values']) / self.metrics[name]['Portfolio Values'][:-1]
            plt.hist(returns, bins=50, alpha=0.5, label=name)
        plt.title('Returns Distribution')
        plt.xlabel('Returns')
        plt.ylabel('Frequency')
        plt.legend()

        # Drawdown
        plt.subplot(2, 2, 3)
        for name in self.metrics:
            portfolio_values = np.array(self.metrics[name]['Portfolio Values'])
            drawdown = (portfolio_values - np.maximum.accumulate(portfolio_values)) / np.maximum.accumulate(portfolio_values)
            plt.plot(drawdown, label=name)
        plt.title('Drawdown Over Time')
        plt.xlabel('Time Steps')
        plt.ylabel('Drawdown')
        plt.legend()

        # Position Distribution
        plt.subplot(2, 2, 4)
        positions_data = []
        labels = []
        for name in self.metrics:
            positions = self.metrics[name]['Positions']
            positions_data.append(positions)
            labels.append(name)
        plt.boxplot(positions_data, labels=labels)
        plt.title('Position Distribution')
        plt.ylabel('Position')

        plt.tight_layout()
        plt.show()


def evaluate_all_models(test_df, trained_models):
    """Evaluate and compare all trained models"""
    print("\nEvaluating models...")
    # Initialize baseline models
    baseline_models = {
        'Buy & Hold': BuyAndHoldAgent(),
        'MA Crossover': MovingAverageCrossoverAgent(),
        'RSI': RSIAgent(),
        'Bollinger Bands': BollingerBandsAgent(),
        'MACD': MACDAgent(),
        'VWMA': VolumeWeightedMAAgent(),
        'Mean Reversion': MeanReversionAgent(),
    }

    # Combine all models for evaluation
    all_models = {**trained_models, **baseline_models}

    # Create evaluator and run comparison
    evaluator = ModelEvaluator(test_df)
    comparison_results = evaluator.compare_models(all_models)

    # Display results
    print("\nComparison Results:")
    print(comparison_results)

    # Save results
    comparison_results.to_csv("model_comparison_results.csv")
