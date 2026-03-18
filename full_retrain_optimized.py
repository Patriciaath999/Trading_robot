# -*- coding: utf-8 -*-
"""
完整重新訓練：SAC + A2C + PPO
使用優化的訓練步數
"""

import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import SAC, A2C, PPO

from preprocessing.preprocessors import data_split
from config.config import *
from model.meta_learner_sac import MetaLearnerSAC
from env.EnvMultipleStock_train import StockEnvTrain
from env.EnvMultipleStock_validation import StockEnvValidation


# 優化的訓練配置
TRAINING_CONFIG = {
    'A2C': {
        'timesteps': 200_000,
        'learning_rate': 0.0007,
        'verbose': 1
    },
    'PPO': {
        'timesteps': 200_000,
        'learning_rate': 0.0003,
        'verbose': 1
    },
    'SAC': {
        'timesteps': 100_000,  # SAC 100k 已經很好
        'learning_rate': 0.0003,
        'verbose': 1
    }
}


def train_single_model(model_name, train_data, output_dir):
    """訓練單個模型"""
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "=" * 60)
    logger.info(f"訓練 {model_name}")
    logger.info("=" * 60)
    
    config = TRAINING_CONFIG[model_name]
    logger.info(f"訓練步數: {config['timesteps']:,}")
    
    env = DummyVecEnv([lambda: StockEnvTrain(train_data)])
    
    start_time = datetime.now()
    
    if model_name == 'A2C':
        model = A2C('MlpPolicy', env, learning_rate=config['learning_rate'], verbose=config['verbose'])
    elif model_name == 'PPO':
        model = PPO('MlpPolicy', env, learning_rate=config['learning_rate'], verbose=config['verbose'])
    elif model_name == 'SAC':
        model = SAC('MlpPolicy', env, learning_rate=config['learning_rate'], verbose=config['verbose'])
    
    logger.info("開始訓練...")
    model.learn(total_timesteps=config['timesteps'])
    
    train_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"✓ 訓練完成 ({train_time/60:.1f} 分鐘)")
    
    # 保存
    model_path = os.path.join(output_dir, f"{model_name}_trained")
    model.save(model_path)
    logger.info(f"✓ 模型保存: {model_path}.zip")
    
    env.close()
    
    return model, train_time


def validate_model(model, model_name, val_data, iteration_name):
    """驗證模型"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"  驗證 {model_name}...")
    
    env_val = StockEnvValidation(
        val_data.copy(),
        turbulence_threshold=140,
        iteration=iteration_name
    )
    
    obs = env_val.reset()
    done = False
    
    is_sac = (model_name == 'SAC')
    
    while not done:
        if is_sac:
            obs_2d = np.array(obs).reshape(1, -1)
            action, _ = model.predict(obs_2d, deterministic=True)
            action = action[0] if len(action.shape) > 1 else action
        else:
            action, _ = model.predict(obs, deterministic=True)
        
        obs, reward, done, info = env_val.step(action)
    
    # 計算 Sharpe
    result_path = f"results/account_value_validation_{iteration_name}.csv"
    sharpe = 0.0
    
    if os.path.exists(result_path):
        df = pd.read_csv(result_path)
        if 'account_value' in df.columns:
            returns = df['account_value'].pct_change().fillna(0)
            if returns.std() != 0:
                sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    
    logger.info(f"    Sharpe: {sharpe:.4f}, 交易: {env_val.trades}")
    
    return sharpe


def rolling_validation(models_dict, data, output_dir):
    """滾動驗證並訓練 Meta-learner"""
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "=" * 60)
    logger.info("滾動驗證 + Meta-learner 訓練")
    logger.info("=" * 60)
    
    unique_dates = data[(data.datadate >= VALIDATION_START) & 
                        (data.datadate <= TEST_END)].datadate.unique()
    unique_dates = np.sort(unique_dates)
    
    rebalance_window = 63
    validation_window = 63
    
    meta_learner = MetaLearnerSAC(model_names=['SAC', 'A2C', 'PPO'])
    weights_history = []
    
    iteration_count = 0
    for i in range(rebalance_window + validation_window, len(unique_dates), rebalance_window):
        iteration_count += 1
        
        logger.info(f"\n迭代 {iteration_count}: {i}/{len(unique_dates)}")
        
        val_start = unique_dates[i - rebalance_window - validation_window]
        val_end = unique_dates[i - rebalance_window]
        validation = data_split(data, start=val_start, end=val_end)
        
        logger.info(f"驗證期: {val_start} - {val_end}")
        
        # 驗證各模型
        performances = {}
        for name, model in models_dict.items():
            if model is not None:
                iteration_name = f"{name.lower()}_{iteration_count}"
                sharpe = validate_model(model, name, validation, iteration_name)
                performances[name] = sharpe
            else:
                performances[name] = 0.0
        
        logger.info(f"表現: {performances}")
        
        # Meta-learner
        current_date = unique_dates[i - rebalance_window]
        market_features = meta_learner.extract_market_features(data, current_date)
        meta_learner.collect_training_data(market_features, performances)
        
        if len(meta_learner.feature_history) >= 5:
            meta_learner.train(min_samples=2)
            
            if meta_learner.is_trained:
                weights = meta_learner.predict_weights(market_features)
                weights_history.append({
                    'iteration': iteration_count,
                    'date': current_date,
                    **weights,
                    **{f'{k}_sharpe': v for k, v in performances.items()}
                })
                logger.info(f"權重: {weights}")
    
    # 保存
    meta_learner.save(os.path.join(output_dir, "meta_learner.pkl"))
    
    if weights_history:
        df_weights = pd.DataFrame(weights_history)
        df_weights.to_csv(os.path.join(output_dir, "weights_history.csv"), index=False)
        
        logger.info("\n權重統計:")
        for col in ['SAC', 'A2C', 'PPO']:
            logger.info(f"  {col}: mean={df_weights[col].mean():.3f}, std={df_weights[col].std():.3f}")
        
        logger.info("\nSharpe 統計:")
        for col in ['SAC_sharpe', 'A2C_sharpe', 'PPO_sharpe']:
            logger.info(f"  {col}: mean={df_weights[col].mean():.4f}")
    
    return meta_learner


def run_test_trading(models_dict, meta_learner, test_data, output_dir):
    """執行測試期交易"""
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "=" * 60)
    logger.info("測試期交易")
    logger.info("=" * 60)
    
    env_trade = StockEnvValidation(
        test_data,
        turbulence_threshold=140,
        iteration='test_final'
    )
    
    obs = env_trade.reset()
    done = False
    step = 0
    
    while not done:
        # 獲取權重
        if meta_learner and meta_learner.is_trained:
            try:
                current_date = test_data.iloc[min(step * env_trade.stock_dim, len(test_data)-1)]['datadate']
                features = meta_learner.extract_market_features(test_data, current_date)
                weights = meta_learner.predict_weights(features)
            except:
                weights = {name: 1.0/3 for name in models_dict.keys()}
        else:
            weights = {name: 1.0/3 for name in models_dict.keys()}
        
        # 獲取動作
        actions = {}
        for name, model in models_dict.items():
            if model is None:
                actions[name] = np.zeros(env_trade.stock_dim)
                continue
            
            try:
                if name == 'SAC':
                    obs_2d = np.array(obs).reshape(1, -1)
                    action, _ = model.predict(obs_2d, deterministic=True)
                    actions[name] = action[0] if len(action.shape) > 1 else action
                else:
                    action, _ = model.predict(obs, deterministic=True)
                    actions[name] = action
            except:
                actions[name] = np.zeros(env_trade.stock_dim)
        
        # 組合
        final_action = np.zeros(env_trade.stock_dim)
        for name, action in actions.items():
            final_action += weights[name] * action
        
        obs, reward, done, info = env_trade.step(final_action)
        step += 1
        
        if step % 20 == 0:
            logger.info(f"Day {step}: value={env_trade.asset_memory[-1]:,.0f}")
    
    logger.info(f"✓ 完成: {step} 天, {env_trade.trades} 筆交易")
    
    # 計算結果
    result_path = "results/account_value_validation_test_final.csv"
    if os.path.exists(result_path):
        df = pd.read_csv(result_path)
        if 'account_value' in df.columns:
            final = df['account_value'].iloc[-1]
            profit = final - 100_000_000
            rate = (profit / 100_000_000) * 100
            
            logger.info("\n最終結果:")
            logger.info(f"  利潤: {profit:,.0f} 元")
            logger.info(f"  報酬率: {rate:.2f}%")
            logger.info(f"  交易: {env_trade.trades}")
            
            summary = {
                'final_asset': final,
                'profit': profit,
                'return_rate': rate,
                'trades': env_trade.trades,
                'cost': env_trade.cost
            }
            
            pd.DataFrame([summary]).to_csv(os.path.join(output_dir, "test_summary.csv"), index=False)
            return summary
    
    return None


def main():
    """完整訓練流程"""
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("完整重新訓練：SAC + A2C + PPO（優化版）")
    logger.info("=" * 80)
    
    # 載入資料
    logger.info("\n步驟 1: 載入資料")
    data = pd.read_csv("data/train_id_2016_2025_processed_with_turbulence_20260108_041718.csv")
    logger.info(f"✓ 資料: {data.shape}")
    
    train_data = data_split(data, start=TRAIN_START, end=TRAIN_END)
    test_data = data_split(data, start=20250909, end=20251231)
    
    # 輸出目錄
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"trained_models/optimized_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"輸出: {output_dir}")
    
    # 訓練模型
    logger.info("\n步驟 2: 訓練三個模型")
    logger.info("預計時間: 90-120 分鐘")
    
    models_dict = {}
    train_times = {}
    
    for model_name in ['A2C', 'PPO', 'SAC']:
        try:
            model, train_time = train_single_model(model_name, train_data, output_dir)
            models_dict[model_name] = model
            train_times[model_name] = train_time
        except Exception as e:
            logger.error(f"❌ {model_name} 訓練失敗: {e}")
            models_dict[model_name] = None
    
    # 滾動驗證
    logger.info("\n步驟 3: 滾動驗證 + Meta-learner")
    meta_learner = rolling_validation(models_dict, data, output_dir)
    
    # 測試交易
    logger.info("\n步驟 4: 測試期交易")
    summary = run_test_trading(models_dict, meta_learner, test_data, output_dir)
    
    # 總結
    logger.info("\n" + "=" * 80)
    logger.info("✓ 訓練完成！")
    logger.info("=" * 80)
    logger.info(f"\n結果保存在: {output_dir}")
    
    if summary:
        logger.info(f"\n測試期表現:")
        logger.info(f"  報酬率: {summary['return_rate']:.2f}%")
        logger.info(f"  交易次數: {summary['trades']}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/full_retrain_optimized.log', mode='w', encoding='utf-8')
        ]
    )
    
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    print("=" * 80)
    print("完整重新訓練")
    print("=" * 80)
    print("配置:")
    print("  A2C: 200k 步 (~35 分鐘)")
    print("  PPO: 200k 步 (~35 分鐘)")
    print("  SAC: 100k 步 (~40 分鐘)")
    print("\n總預計: 2 小時")
    print("=" * 80)
    print()
    
    main()