import pandas as pd
import numpy as np
from datetime import datetime
import torch
import json
import os
from train_evaluate import ModelEvaluator
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
import matplotlib.pyplot as plt

from qrdqn import QRDQN
from train_evaluate import train_qrdqn
from env import TradingEnvironmentV6



class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()

        return super(NumpyEncoder, self).default(obj)
class HyperparameterTuner:
    def __init__(self, train_df, test_df, base_path="./hyperparam_results"):
        self.train_df = train_df
        self.test_df = test_df
        self.base_path = base_path
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = []

        os.makedirs(base_path, exist_ok=True)

        # Define network architectures
        self.network_architectures = {
            'small': [128, 128],
            'medium': [256, 256],
            'large': [512, 256]
        }

        # Define parameter spaces matching QRDQN.__init__ parameters
        self.space = [
            Real(1e-4, 1e-3, name='learning_rate', prior='log-uniform'),
            Integer(32, 128, name='batch_size'),
            Integer(31, 101, name='num_quantiles'),
            Real(0.95, 0.99, name='gamma'),
            Real(0.001, 0.01, name='tau'),
            Categorical(['small', 'medium', 'large'], name='architecture'),
            Real(0.1, 0.3, name='dropout')
        ]

    def parse_params(self, x):
        """Convert optimization parameters to agent and network configs"""
        # Convert numpy types to Python native types
        agent_params = {
            'learning_rate': float(x[0]),
            'batch_size': int(x[1]),
            'num_quantiles': int(x[2]),
            'gamma': float(x[3]),
            'tau': float(x[4]),
            'buffer_size': 100000
        }

        network_params = {
            'hidden_dims': self.network_architectures[x[5]],
            'dropout': float(x[6])
        }


        return agent_params, network_params

    def objective_function(self, metrics):
        """Calculate overall score from metrics"""
        return -(  # Negative because skopt minimizes
            metrics['Sharpe Ratio'] * 0.4 +
            metrics['Total Return (%)'] * 0.3 +
            (-metrics['Maximum Drawdown (%)']) * 0.2 +
            metrics['Win Rate (%)'] * 0.1
        )

    def evaluate_single_config(self, x):
        try:
            agent_params, network_params = self.parse_params(x)

            # Initialize environment
            env = TradingEnvironmentV6(self.train_df)
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n

            # Initialize agent
            agent = QRDQN(
                state_dim=state_dim,
                action_dim=action_dim,
                **agent_params
            )

            # Train agent
            print(f"\nTraining with parameters:")
            print(json.dumps(agent_params, indent=2, cls=NumpyEncoder))
            print(json.dumps(network_params, indent=2, cls=NumpyEncoder))

            rewards, portfolios, eval_results = train_qrdqn(
                env=env,
                agent=agent,
                num_episodes=1000,
                initial_epsilon=1.0,
                epsilon_min=0.05,
                epsilon_decay=0.995,
                eval_frequency=500,
                min_samples_before_training=1000
            )

            # Evaluate on test set
            evaluator = ModelEvaluator(self.test_df)
            metrics = evaluator.evaluate_model(agent, "QR-DQN")

            # Convert numpy values to Python native types
            metrics = {k: float(v) if isinstance(v, np.number) else v
                      for k, v in metrics.items()}

            # Calculate score
            score = self.objective_function(metrics)

            # Save result
            result = {
                'agent_params': agent_params,
                'network_params': network_params,
                'metrics': metrics,
                'score': float(-score),
                'final_portfolio': float(portfolios[-1]) if portfolios else None,
                'avg_reward': float(np.mean(rewards)) if rewards else None
            }

            self.results.append(result)

            # Save if best so far
            if len(self.results) == 1 or result['score'] > max(r['score'] for r in self.results):
                model_path = f"{self.base_path}/best_model_{self.timestamp}.pth"
                torch.save({
                    'model_state_dict': agent.network.state_dict(),
                    'agent_params': agent_params,
                    'network_params': network_params,
                    'metrics': metrics,
                    'score': result['score']
                }, model_path)

            self.save_current_results()
            return score

        except Exception as e:
            print(f"Error evaluating parameters: {str(e)}")
            import traceback
            traceback.print_exc()
            return 1e6

    def tune(self, n_calls=20):
        """Run Bayesian optimization"""
        print("Starting Bayesian optimization...")

        # Run optimization
        result = gp_minimize(
            func=self.evaluate_single_config,
            dimensions=self.space,
            n_calls=n_calls,
            n_random_starts=5,
            noise=0.1,
            random_state=42
        )

        self.save_final_results()
        return self.get_best_params()

    def get_best_params(self):
        """Get parameters that achieved the best score"""
        if not self.results:
            return None

        best_result = max(self.results, key=lambda x: x['score'])
        return {
            'agent_params': best_result['agent_params'],
            'network_params': best_result['network_params'],
            'metrics': best_result['metrics'],
            'score': best_result['score']
        }

    def save_current_results(self):
        """Save current results to CSV"""
        results_df = self.create_results_df()
        results_df.to_csv(
            f"{self.base_path}/results_{self.timestamp}.csv",
            index=False
        )

        # Save full results as JSON
        with open(f"{self.base_path}/full_results_{self.timestamp}.json", 'w') as f:
            json.dump(self.results, f, indent=2, cls=NumpyEncoder)


    def save_final_results(self):
        """Save final results with detailed analysis"""
        results_df = self.create_results_df()

        # Save detailed results
        results_df.to_csv(f"{self.base_path}/final_results_{self.timestamp}.csv", index=False)

        # Save summary statistics
        summary_stats = {
            'Best Score': results_df['score'].max(),
            'Average Score': results_df['score'].mean(),
            'Std Score': results_df['score'].std(),
            'Best Sharpe': results_df['sharpe_ratio'].max(),
            'Best Return': results_df['total_return'].max(),
            'Best Win Rate': results_df['win_rate'].max()
        }

        pd.Series(summary_stats).to_csv(f"{self.base_path}/summary_{self.timestamp}.csv")

    def create_results_df(self):
        """Convert results to DataFrame"""
        rows = []
        for result in self.results:
            row = {
                'score': result['score'],
                'sharpe_ratio': result['metrics']['Sharpe Ratio'],
                'total_return': result['metrics']['Total Return (%)'],
                'max_drawdown': result['metrics']['Maximum Drawdown (%)'],
                'win_rate': result['metrics']['Win Rate (%)'],
                'final_portfolio': result['final_portfolio'],
                'avg_reward': result['avg_reward']
            }
            # Add parameters
            row.update({f'agent_{k}': v for k, v in result['agent_params'].items()})
            row.update({f'network_{k}': str(v) for k, v in result['network_params'].items()})
            rows.append(row)

        return pd.DataFrame(rows)

    def plot_optimization_results(self):
        """Plot optimization results"""
        results_df = self.create_results_df()

        plt.figure(figsize=(20, 10))

        # Score progression
        plt.subplot(2, 3, 1)
        plt.plot(results_df['score'])
        plt.title('Optimization Progress')
        plt.xlabel('Trial')
        plt.ylabel('Score')

        # Score vs key parameters
        params_to_plot = ['agent_learning_rate', 'agent_batch_size',
                         'agent_num_quantiles', 'network_dropout']

        for i, param in enumerate(params_to_plot, 2):
            plt.subplot(2, 3, i)
            plt.scatter(results_df[param], results_df['score'])
            plt.xlabel(param)
            plt.ylabel('Score')
            plt.title(f'Score vs {param}')

        # Distribution of scores
        plt.subplot(2, 3, 6)
        plt.hist(results_df['score'], bins=20)
        plt.title('Distribution of Scores')
        plt.xlabel('Score')
        plt.ylabel('Count')

        plt.tight_layout()
        plt.savefig(f"{self.base_path}/optimization_results_{self.timestamp}.png")
        plt.close()

def run_bayesian_optimization(train_df, test_df, n_calls=20):
    tuner = HyperparameterTuner(train_df, test_df)
    best_config = tuner.tune(n_calls=n_calls)
    tuner.plot_optimization_results()
    return best_config
