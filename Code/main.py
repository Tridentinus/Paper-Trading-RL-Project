from env import TradingEnvironmentV6
from train_evaluate import train_all_models, evaluate_all_models
from agents import BaselineAgent
from tuning import run_bayesian_optimization
from utils import download_and_prepare_data

if __name__ == "__main__":
    df = download_and_prepare_data()
    train_size = int(len(df) * 0.8)
    train_df = df[:train_size]
    test_df = df[train_size:]

    # Train models
    trained_models = train_all_models(train_df, test_df)
    
    # Evaluate models
    evaluate_all_models(train_df, test_df, trained_models)
    

    # Run hyperparameter optimization
    best_config = run_bayesian_optimization(train_df, test_df, n_calls=20)
    
