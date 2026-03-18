"""
簡化版 PPO - 離散動作空間 + 改進獎勵函數
===================================================
改進重點：
1. 26 支股票（全保留）
2. 6 個關鍵特徵（降低複雜度 60%）
3. 離散動作空間：每股票 {賣出, 持有, 買入}
4. 改進獎勵：報酬率 - 波動懲罰 + 分散度獎勵
5. 訓練期：2021-2024（疫情後）
6. 訓練步數：500k-1M
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from gymnasium import spaces
import warnings
warnings.filterwarnings('ignore')

# ========================================
# 數據處理
# ========================================
def data_split(df, start, end):
    """分割數據"""
    data = df[(df.datadate >= start) & (df.datadate < end)].copy()
    data = data.sort_values(['datadate', 'tic'], ignore_index=True)
    data.index = data.datadate.factorize()[0]
    return data

# ========================================
# 簡化版訓練環境（離散動作）
# ========================================
class StockEnvTrainDiscrete(gym.Env):
    """
    簡化版 PPO 訓練環境
    - 26 支股票
    - 6 個特徵
    - 離散動作空間
    - 改進獎勵函數
    """
    
    metadata = {'render_modes': []}
    
    def __init__(self, df, initial_amount=1_000_000, transaction_cost_pct=0.001):
        super().__init__()
        self.df = df
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        
        # 6 個關鍵特徵
        self.tech_indicators = [
            'macd',              # 趨勢
            'momentum_20d',      # 動量
            'rsi',               # 超買超賣
            'close_to_sma_20',  # 價格位置
            'volume_ratio_20',   # 成交量
            'turbulence'         # 風險
        ]
        
        self.stock_dim = len(df.tic.unique())
        self.state_dim = 1 + self.stock_dim * (1 + len(self.tech_indicators))  # 1 + 26*7 = 183
        
        # 離散動作空間：每支股票 3 種選擇 {0:賣, 1:持有, 2:買}
        self.action_space = spaces.MultiDiscrete([3] * self.stock_dim)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.state_dim,), 
            dtype=np.float32
        )
        
        self._initialize_state()
    
    def _initialize_state(self):
        """初始化狀態"""
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        
        self.portfolio_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        
        # 持股數量
        self.stocks = np.zeros(self.stock_dim)
        # 剩餘現金
        self.cash = self.initial_amount
        # 交易記錄
        self.cost = 0
        self.trades = 0
        
        # 用於計算波動率
        self.return_history = []
    
    def reset(self, seed=None, options=None):
        """重置環境 (Gymnasium API)

        Accepts `seed` and `options` to be compatible with Gymnasium / SB3.
        """
        if seed is not None:
            np.random.seed(seed)
        self._initialize_state()
        return self._get_state(), {}
    
    def _get_state(self):
        """獲取狀態向量"""
        # 1. 現金（歸一化）
        cash_norm = self.cash / self.initial_amount
        
        # 2. 持股數量（歸一化）
        holdings = []
        for i in range(self.stock_dim):
            price = self.data.iloc[i]['adjcp']
            holding_value = self.stocks[i] * price
            holdings.append(holding_value / self.initial_amount)
        
        # 3. 技術指標
        indicators = []
        for i in range(self.stock_dim):
            for tech in self.tech_indicators:
                val = self.data.iloc[i].get(tech, 0)
                
                # 歸一化
                if tech == 'rsi':
                    val = val / 100
                elif tech == 'macd':
                    val = np.tanh(val / 10)  # 壓縮到 [-1, 1]
                elif tech in ['momentum_20d', 'close_to_sma_20']:
                    val = np.tanh(val / 20)
                elif tech == 'volume_ratio_20':
                    val = np.tanh((val - 1) / 2)
                elif tech == 'turbulence':
                    val = np.clip(val / 500, 0, 1)
                else:
                    val = np.tanh(val / 10)
                
                indicators.append(val)
        
        state = [cash_norm] + holdings + indicators
        return np.array(state, dtype=np.float32)
    
    def _calculate_reward(self):
        """
        改進的獎勵函數
        reward = 報酬率 - 0.5 × 波動懲罰 + 0.1 × 多樣化獎勵
        """
        # 1. 報酬率
        returns = (self.portfolio_value - self.asset_memory[-2]) / self.asset_memory[-2]
        self.return_history.append(returns)
        
        # 2. 波動懲罰（用最近 20 個交易日計算）
        if len(self.return_history) >= 20:
            volatility = np.std(self.return_history[-20:])
        else:
            volatility = 0
        
        # 3. 多樣化獎勵（持股分散度）
        total_stock_value = sum(self.stocks[i] * self.data.iloc[i]['adjcp'] 
                               for i in range(self.stock_dim))
        
        if total_stock_value > 0:
            # 計算持股集中度（0=完全分散, 1=all-in 單一股票）
            stock_weights = []
            for i in range(self.stock_dim):
                weight = (self.stocks[i] * self.data.iloc[i]['adjcp']) / total_stock_value
                stock_weights.append(weight)
            
            # Herfindahl index（越小越分散）
            concentration = sum(w**2 for w in stock_weights)
            diversity_bonus = 1 - concentration  # 轉成獎勵（分散=高分）
        else:
            diversity_bonus = 0
        
        # 組合獎勵
        reward = returns - 0.5 * volatility + 0.1 * diversity_bonus
        
        return reward
    
    def step(self, actions):
        """執行動作"""
        self.terminal = (self.day >= len(self.df.index.unique()) - 1)
        
        if self.terminal:
            # 最後一天，賣出所有持股
            final_value = self.cash
            for i in range(self.stock_dim):
                final_value += self.stocks[i] * self.data.iloc[i]['adjcp']
            
            self.portfolio_value = final_value
            self.asset_memory.append(final_value)
            
            return_rate = (final_value - self.initial_amount) / self.initial_amount
            
            return self._get_state(), return_rate, True, False, {}
        
        # 執行交易
        # actions: [0,1,2] * 26
        # 0=賣出, 1=持有, 2=買入
        
        for i in range(self.stock_dim):
            action = actions[i]
            price = self.data.iloc[i]['adjcp']
            
            if action == 0:  # 賣出
                if self.stocks[i] > 0:
                    sell_amount = self.stocks[i]
                    self.cash += price * sell_amount * (1 - self.transaction_cost_pct)
                    self.cost += price * sell_amount * self.transaction_cost_pct
                    self.stocks[i] = 0
                    self.trades += 1
            
            elif action == 2:  # 買入
                # 用剩餘資金的 5%（保守策略）
                invest_amount = self.cash * 0.05
                available_shares = int(invest_amount / (price * (1 + self.transaction_cost_pct)))
                
                if available_shares > 0:
                    buy_amount = min(available_shares, 100)  # 最多 100 股
                    cost = price * buy_amount * (1 + self.transaction_cost_pct)
                    
                    if cost <= self.cash:
                        self.cash -= cost
                        self.stocks[i] += buy_amount
                        self.cost += price * buy_amount * self.transaction_cost_pct
                        self.trades += 1
            
            # action == 1: 持有（不動作）
        
        # 移到下一天
        self.day += 1
        self.data = self.df.loc[self.day, :]
        
        # 計算投資組合價值
        portfolio_value = self.cash
        for i in range(self.stock_dim):
            portfolio_value += self.stocks[i] * self.data.iloc[i]['adjcp']
        
        self.portfolio_value = portfolio_value
        self.asset_memory.append(portfolio_value)
        
        # 計算獎勵
        reward = self._calculate_reward()
        self.portfolio_return_memory.append(reward)
        
        return self._get_state(), reward, False, False, {}

# ========================================
# 驗證環境
# ========================================
class StockEnvValidationDiscrete(gym.Env):
    """驗證環境（離散動作）"""
    
    metadata = {'render_modes': []}
    
    def __init__(self, df, turbulence_threshold=350, initial_amount=100_000_000, 
                 transaction_cost_pct=0.001, iteration=''):
        super().__init__()
        self.df = df
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.turbulence_threshold = turbulence_threshold
        self.iteration = iteration
        
        self.tech_indicators = [
            'macd', 'momentum_20d', 'rsi', 'close_to_sma_20', 
            'volume_ratio_20', 'turbulence'
        ]
        
        self.stock_dim = len(df.tic.unique())
        self.state_dim = 1 + self.stock_dim * (1 + len(self.tech_indicators))
        
        self.action_space = spaces.MultiDiscrete([3] * self.stock_dim)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.state_dim,), 
            dtype=np.float32
        )
        
        self._initialize_state()
    
    def _initialize_state(self):
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        
        self.portfolio_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.date_memory = [self._get_date()]
        
        self.stocks = np.zeros(self.stock_dim)
        self.cash = self.initial_amount
        self.cost = 0
        self.trades = 0
    
    def _get_date(self):
        return self.data.iloc[0]['datadate'] if len(self.data) > 0 else 0
    
    def reset(self, seed=None, options=None):
        """重置環境 (Gymnasium API)

        Accepts `seed` and `options` to be compatible with Gymnasium / SB3.
        """
        if seed is not None:
            np.random.seed(seed)
        self._initialize_state()
        return self._get_state(), {}
    
    def _get_state(self):
        """與訓練環境相同"""
        cash_norm = self.cash / self.initial_amount
        
        holdings = []
        for i in range(self.stock_dim):
            price = self.data.iloc[i]['adjcp']
            holding_value = self.stocks[i] * price
            holdings.append(holding_value / self.initial_amount)
        
        indicators = []
        for i in range(self.stock_dim):
            for tech in self.tech_indicators:
                val = self.data.iloc[i].get(tech, 0)
                
                if tech == 'rsi':
                    val = val / 100
                elif tech == 'macd':
                    val = np.tanh(val / 10)
                elif tech in ['momentum_20d', 'close_to_sma_20']:
                    val = np.tanh(val / 20)
                elif tech == 'volume_ratio_20':
                    val = np.tanh((val - 1) / 2)
                elif tech == 'turbulence':
                    val = np.clip(val / 500, 0, 1)
                else:
                    val = np.tanh(val / 10)
                
                indicators.append(val)
        
        state = [cash_norm] + holdings + indicators
        return np.array(state, dtype=np.float32)
    
    def step(self, actions):
        self.terminal = (self.day >= len(self.df.index.unique()) - 1)
        
        if self.terminal:
            final_value = self.cash
            for i in range(self.stock_dim):
                final_value += self.stocks[i] * self.data.iloc[i]['adjcp']
            
            self.portfolio_value = final_value
            self.asset_memory.append(final_value)
            self.date_memory.append(self._get_date())
            
            return_rate = (final_value - self.initial_amount) / self.initial_amount
            
            # 保存結果
            df_result = pd.DataFrame({
                'date': self.date_memory,
                'asset': self.asset_memory
            })
            
            os.makedirs('results', exist_ok=True)
            df_result.to_csv(f'results/test_{self.iteration}.csv', index=False)
            
            return self._get_state(), return_rate, True, False, {}
        
        # 風險控制
        turbulence = self.data.iloc[0].get('turbulence', 0)
        if turbulence >= self.turbulence_threshold:
            actions = np.zeros(self.stock_dim, dtype=int)  # 全部賣出
        
        # 執行交易
        for i in range(self.stock_dim):
            action = actions[i]
            price = self.data.iloc[i]['adjcp']
            
            if action == 0:  # 賣出
                if self.stocks[i] > 0:
                    sell_amount = self.stocks[i]
                    self.cash += price * sell_amount * (1 - self.transaction_cost_pct)
                    self.cost += price * sell_amount * self.transaction_cost_pct
                    self.stocks[i] = 0
                    self.trades += 1
            
            elif action == 2:  # 買入
                invest_amount = self.cash * 0.05
                available_shares = int(invest_amount / (price * (1 + self.transaction_cost_pct)))
                
                if available_shares > 0:
                    buy_amount = min(available_shares, 100)
                    cost = price * buy_amount * (1 + self.transaction_cost_pct)
                    
                    if cost <= self.cash:
                        self.cash -= cost
                        self.stocks[i] += buy_amount
                        self.cost += price * buy_amount * self.transaction_cost_pct
                        self.trades += 1
        
        self.day += 1
        self.data = self.df.loc[self.day, :]
        
        portfolio_value = self.cash
        for i in range(self.stock_dim):
            portfolio_value += self.stocks[i] * self.data.iloc[i]['adjcp']
        
        self.portfolio_value = portfolio_value
        self.asset_memory.append(portfolio_value)
        self.date_memory.append(self._get_date())
        
        reward = (portfolio_value - self.asset_memory[-2]) / self.asset_memory[-2]
        
        return self._get_state(), reward, False, False, {}

# ========================================
# 訓練回調
# ========================================
class ProgressCallback(BaseCallback):
    def __init__(self, check_freq, total_timesteps, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.total_timesteps = total_timesteps
        self.start_time = None
    
    def _on_training_start(self):
        self.start_time = datetime.now()
    
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            progress = self.n_calls / self.total_timesteps
            remaining = (elapsed / progress - elapsed) if progress > 0 else 0
            
            print(f"  進度: {self.n_calls:>7,} / {self.total_timesteps:,} ({progress*100:>5.1f}%) | "
                  f"用時: {elapsed/60:>5.1f} min | 剩餘: {remaining/60:>5.1f} min")
        return True

# ========================================
# 主程式
# ========================================
def main():
    print("=" * 80)
    print("🚀 簡化版 PPO - 離散動作 + 改進獎勵")
    print("=" * 80)
    print("改進重點:")
    print("  ✓ 26 支股票（全保留）")
    print("  ✓ 6 個關鍵特徵（降低 60% 複雜度）")
    print("  ✓ 離散動作：{賣出, 持有, 買入}")
    print("  ✓ 改進獎勵：報酬 - 波動 + 分散")
    print("  ✓ 訓練期：2021-2024（疫情後）")
    print("=" * 80)
    
    # 載入數據
    print("\n[1/4] 載入數據...")
    data_path = r"C:\Users\nancy\Desktop\資料系統\Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020\data\train_id_2016_2025_processed_with_turbulence_20260108_041718.csv"
    
    data = pd.read_csv(data_path)
    print(f"  總數據: {data.shape}")
    
    # 時期劃分（疫情後）
    TRAIN_START = 20210101
    TRAIN_END = 20241231
    VAL_START = 20250101
    VAL_END = 20250901
    TEST_START = 20250909
    TEST_END = 20260109  # 修正到正確日期
    
    train_data = data_split(data, TRAIN_START, TRAIN_END)
    val_data = data_split(data, VAL_START, VAL_END)
    test_data = data_split(data, TEST_START, TEST_END)
    
    print(f"  訓練期 (2021-2024): {train_data.shape}")
    print(f"  驗證期 (2025/01-08): {val_data.shape}")
    print(f"  測試期 (2025/09-2026/01/09): {test_data.shape}")
    
    # 檢查特徵
    required = ['macd', 'momentum_20d', 'rsi', 'close_to_sma_20', 
                'volume_ratio_20', 'turbulence']
    missing = [f for f in required if f not in train_data.columns]
    
    if missing:
        print(f"\n❌ 缺少特徵: {missing}")
        return
    
    print(f"  ✓ 所有 6 個關鍵特徵存在")
    print(f"  狀態維度: 1 + 26×7 = 183 維（vs 原本 391 維）")
    
    # 訓練 PPO
    TOTAL_STEPS = 500_000
    
    print(f"\n[2/4] 訓練簡化版 PPO...")
    print(f"  訓練步數: {TOTAL_STEPS:,}")
    print(f"  預計時間: 2.5-3.5 小時")
    print()
    
    env_train = DummyVecEnv([lambda: StockEnvTrainDiscrete(train_data)])
    
    model = PPO(
        'MlpPolicy',
        env_train,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # 鼓勵探索
        verbose=0
    )
    
    callback = ProgressCallback(check_freq=25000, total_timesteps=TOTAL_STEPS)
    
    train_start = datetime.now()
    model.learn(total_timesteps=TOTAL_STEPS, callback=callback)
    train_time = (datetime.now() - train_start).total_seconds()
    
    print(f"\n  ✓ 訓練完成 ({train_time/60:.1f} 分鐘)")
    
    # 保存模型
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"trained_models/ppo_simplified_discrete_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, "PPO_simplified")
    model.save(model_path)
    print(f"  ✓ 模型保存: {model_path}.zip")
    
    # 驗證（找最佳 threshold）
    print("\n[3/4] 驗證最佳風險閾值...")
    thresholds = [250, 350, 500, 1000, float('inf')]
    val_results = []
    
    for threshold in thresholds:
        env_val = StockEnvValidationDiscrete(
            val_data,
            turbulence_threshold=threshold,
            iteration=f'val_{threshold}'
        )
        
        obs, _ = env_val.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env_val.step(action)
            done = terminated or truncated
        
        final = env_val.asset_memory[-1]
        profit = final - 100_000_000
        rate = (profit / 100_000_000) * 100
        
        val_results.append({
            'threshold': threshold,
            'return_rate': rate,
            'trades': env_val.trades,
            'final_asset': final
        })
        
        threshold_str = f"{threshold:.0f}" if threshold != float('inf') else "無限制"
        print(f"  Threshold={threshold_str:>6}: {rate:>6.2f}% | 交易={env_val.trades}")
    
    df_val = pd.DataFrame(val_results)
    best_threshold = df_val.loc[df_val['return_rate'].idxmax()]['threshold']
    best_val_return = df_val['return_rate'].max()
    
    print(f"\n  ✓ 最佳閾值: {best_threshold}")
    print(f"  ✓ 驗證期報酬: {best_val_return:.2f}%")
    
    # 測試
    print("\n[4/4] 測試最終表現...")
    env_test = StockEnvValidationDiscrete(
        test_data,
        turbulence_threshold=best_threshold,
        iteration='test_final'
    )
    
    obs, _ = env_test.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_test.step(action)
        done = terminated or truncated
    
    final_value = env_test.asset_memory[-1]
    profit = final_value - 100_000_000
    return_rate = (profit / 100_000_000) * 100
    
    print(f"\n  最終資產: NT$ {final_value:,.0f}")
    print(f"  獲利: NT$ {profit:,.0f}")
    print(f"  報酬率: {return_rate:.2f}%")
    print(f"  交易次數: {env_test.trades}")
    
    # 保存結果
    results_summary = {
        'train_period': '2021-2024',
        'train_time_min': train_time / 60,
        'train_steps': TOTAL_STEPS,
        'best_threshold': best_threshold,
        'val_return_rate': best_val_return,
        'test_return_rate': return_rate,
        'test_profit': profit,
        'test_trades': env_test.trades,
        'final_asset': final_value
    }
    
    pd.DataFrame([results_summary]).to_csv(
        os.path.join(output_dir, 'summary.csv'), 
        index=False
    )
    
    df_val.to_csv(
        os.path.join(output_dir, 'validation_results.csv'),
        index=False
    )
    
    print("\n" + "=" * 80)
    print("🎉 訓練完成！")
    print("=" * 80)
    print(f"模型保存: {output_dir}")
    print(f"\n結果比較:")
    print(f"  簡化版 PPO（離散動作）: {return_rate:.2f}%")
    print(f"  原始 PPO（連續動作）:    0.56%")
    print(f"  技術分析策略:           11.05%")
    print(f"  手動策略:                5.90%")
    print("=" * 80)
    
    # 改進建議
    if return_rate < 3:
        print("\n💡 如果結果不理想，可以嘗試:")
        print("  1. 增加訓練步數到 1M")
        print("  2. 調整買入比例（目前 5%）")
        print("  3. 調整獎勵函數權重")
    elif return_rate < 8:
        print("\n✓ 結果有改善！進一步優化:")
        print("  1. 微調超參數")
        print("  2. 增加訓練步數")
    else:
        print("\n🎉 表現優異！")

if __name__ == "__main__":
    main()