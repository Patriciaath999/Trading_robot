import pandas as pd
import numpy as np
from datetime import datetime
import os
import gym  # <--- [修正1] 加入這行
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import warnings
warnings.filterwarnings('ignore')

# ========================================
# 1. 數據分割函數
# ========================================
def data_split(df, start, end):
    """分割數據"""
    data = df[(df.datadate >= start) & (df.datadate < end)]
    data = data.sort_values(['datadate', 'tic'], ignore_index=True)
    data.index = data.datadate.factorize()[0]
    return data

# ========================================
# 2. 優化的訓練環境 (已修正繼承)
# ========================================
class StockEnvTrain(gym.Env):  # <--- [修正2] 這裡加上 (gym.Env)
    """訓練環境 - 使用 15 個特徵"""
    
    def __init__(self, df, initial_amount=1_000_000, transaction_cost_pct=0.001):
        self.df = df
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        
        # 15 個精選特徵
        self.tech_indicators = [
            'macd', 'rsi', 'cci', 'adx', 'momentum_20d', 'roc_10',
            'close_to_sma_20', 'macd_hist', 'bb_position', 'bb_width',
            'volatility_20d', 'volume_ratio_20', 'atr_pct', 'obv_ratio', 'turbulence'
        ]
        
        self.stock_dim = len(df.tic.unique())
        self.state_dim = 1 + self.stock_dim * (1 + len(self.tech_indicators))
        
        # 定義動作與觀察空間 (gym 要求)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.stock_dim,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        
        self._initialize_state()

    def _initialize_state(self):
        """初始化狀態"""
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        
        self.portfolio_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        
        self.stocks = np.zeros(self.stock_dim)
        self.cash = self.initial_amount
        self.cost = 0
        self.trades = 0

    def reset(self):
        """重置環境"""
        self._initialize_state()
        return self._get_state()

    def _get_state(self):
        """獲取當前狀態向量"""
        cash_norm = [self.cash / self.initial_amount]
        prices = []
        indicators = []
        
        for i in range(self.stock_dim):
            prices.append(self.data.iloc[i]['adjcp'] / 100)
            for tech in self.tech_indicators:
                val = self.data.iloc[i].get(tech, 0)
                # 簡單歸一化處理
                if tech == 'rsi': val = val / 100
                elif tech == 'turbulence': val = np.clip(val / 500, 0, 1)
                else: val = np.clip(val / 100, -1, 1)
                indicators.append(val)
        
        state = cash_norm + prices + indicators
        return np.array(state, dtype=np.float32)

    def step(self, actions):
        """執行動作"""
        self.terminal = (self.day >= len(self.df.index.unique()) - 1)
        
        if self.terminal:
            final_value = self.cash
            for i in range(self.stock_dim):
                final_value += self.stocks[i] * self.data.iloc[i]['adjcp']
            
            self.portfolio_value = final_value
            self.asset_memory.append(final_value)
            return_rate = (final_value - self.initial_amount) / self.initial_amount
            
            return self._get_state(), return_rate, self.terminal, {}
        
        # 交易邏輯
        actions = np.array(actions) # 確保是 numpy array
        argsort_actions = np.argsort(actions)
        sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
        buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]
        
        # 賣出
        for index in sell_index:
            if self.stocks[index] > 0:
                price = self.data.iloc[index]['adjcp']
                sell_num = self.stocks[index]
                self.cash += price * sell_num * (1 - self.transaction_cost_pct)
                self.stocks[index] = 0
                self.cost += price * sell_num * self.transaction_cost_pct
                self.trades += 1
        
        # 買入
        for index in buy_index:
            price = self.data.iloc[index]['adjcp']
            available_cash = self.cash
            # 簡化買入邏輯：最多買 100 股或現金允許的最大量
            if available_cash > price * (1 + self.transaction_cost_pct):
                 # 這裡簡單設定：如果有錢就買，這裡可以優化成根據 action 強度買
                max_buy = available_cash // (price * (1 + self.transaction_cost_pct))
                buy_num = min(max_buy, 100) 
                
                self.cash -= price * buy_num * (1 + self.transaction_cost_pct)
                self.stocks[index] += buy_num
                self.cost += price * buy_num * self.transaction_cost_pct
                self.trades += 1
        
        self.day += 1
        self.data = self.df.loc[self.day, :]
        
        portfolio_value = self.cash
        for i in range(self.stock_dim):
            portfolio_value += self.stocks[i] * self.data.iloc[i]['adjcp']
            
        self.portfolio_value = portfolio_value
        self.asset_memory.append(portfolio_value)
        
        reward = (portfolio_value - self.asset_memory[-2]) / self.asset_memory[-2]
        
        return self._get_state(), reward, self.terminal, {}

# ========================================
# 3. 驗證環境 (已修正繼承)
# ========================================
class StockEnvValidation(gym.Env):  # <--- [修正3] 這裡也加上 (gym.Env)
    """驗證環境"""
    
    def __init__(self, df, turbulence_threshold=350, initial_amount=100_000_000, 
                 transaction_cost_pct=0.001, iteration=''):
        self.df = df
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.turbulence_threshold = turbulence_threshold
        self.iteration = iteration
        
        self.tech_indicators = [
            'macd', 'rsi', 'cci', 'adx', 'momentum_20d', 'roc_10',
            'close_to_sma_20', 'macd_hist', 'bb_position', 'bb_width',
            'volatility_20d', 'volume_ratio_20', 'atr_pct', 'obv_ratio', 'turbulence'
        ]
        
        self.stock_dim = len(df.tic.unique())
        self.state_dim = 1 + self.stock_dim * (1 + len(self.tech_indicators))
        
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.stock_dim,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        
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

    def reset(self):
        self._initialize_state()
        return self._get_state()

    def _get_state(self):
        cash_norm = [self.cash / self.initial_amount]
        prices = []
        indicators = []
        
        for i in range(self.stock_dim):
            prices.append(self.data.iloc[i]['adjcp'] / 100)
            for tech in self.tech_indicators:
                val = self.data.iloc[i].get(tech, 0)
                if tech == 'rsi': val = val / 100
                elif tech == 'turbulence': val = np.clip(val / 500, 0, 1)
                else: val = np.clip(val / 100, -1, 1)
                indicators.append(val)
        
        state = cash_norm + prices + indicators
        return np.array(state, dtype=np.float32)

    def step(self, actions):
        self.terminal = (self.day >= len(self.df.index.unique()) - 1)
        
        if self.terminal:
            final_value = self.cash
            for i in range(self.stock_dim):
                final_value += self.stocks[i] * self.data.iloc[i]['adjcp']
            
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
            
            return self._get_state(), return_rate, self.terminal, {}
        
        # 風險控制
        turbulence = self.data.iloc[0].get('turbulence', 0)
        if turbulence >= self.turbulence_threshold:
            actions = np.array([-1] * self.stock_dim)
        
        actions = np.array(actions)
        argsort_actions = np.argsort(actions)
        sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
        buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]
        
        # 賣出
        for index in sell_index:
            if self.stocks[index] > 0:
                price = self.data.iloc[index]['adjcp']
                sell_num = self.stocks[index]
                self.cash += price * sell_num * (1 - self.transaction_cost_pct)
                self.stocks[index] = 0
                self.cost += price * sell_num * self.transaction_cost_pct
                self.trades += 1
        
        # 買入
        for index in buy_index:
            price = self.data.iloc[index]['adjcp']
            available_cash = self.cash
            if available_cash > price * (1 + self.transaction_cost_pct):
                max_buy = available_cash // (price * (1 + self.transaction_cost_pct))
                buy_num = min(max_buy, 100)
                self.cash -= price * buy_num * (1 + self.transaction_cost_pct)
                self.stocks[index] += buy_num
                self.cost += price * buy_num * self.transaction_cost_pct
                self.trades += 1
        
        self.day += 1
        self.data = self.df.loc[self.day, :]
        
        portfolio_value = self.cash
        for i in range(self.stock_dim):
            portfolio_value += self.stocks[i] * self.data.iloc[i]['adjcp']
        
        self.asset_memory.append(portfolio_value)
        self.date_memory.append(self._get_date())
        
        reward = (portfolio_value - self.asset_memory[-2]) / self.asset_memory[-2]
        
        return self._get_state(), reward, self.terminal, {}

# ========================================
# 4. 訓練回調
# ========================================
class ProgressCallback(BaseCallback):
    def __init__(self, check_freq, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.start_time = None

    def _on_training_start(self):
        self.start_time = datetime.now()

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            progress = self.n_calls / self.locals.get('total_timesteps', 200000)
            print(f"  進度: {self.n_calls:,} steps ({progress*100:.1f}%) | "
                  f"用時: {elapsed/60:.1f} min")
        return True

# ========================================
# 5. 主程式
# ========================================
def main():
    print("=" * 80)
    print("🚀 優化版 PPO 訓練 (Fixed v2)")
    print("=" * 80)
    
    # 載入數據
    print("\n[1/4] 載入數據...")
    try:
        data = pd.read_csv("data/train_id_2016_2025_processed_with_turbulence_20260108_041718.csv")
    except FileNotFoundError:
        print("❌ 錯誤：找不到數據檔案 'data/train_id_2016_2025_processed_with_turbulence_20260108_041718.csv'。")
        print("請確認檔案路徑是否正確。")
        return

    # 時期劃分
    TRAIN_START = 20230101
    TRAIN_END = 20241231
    VAL_START = 20250101
    VAL_END = 20250901
    TEST_START = 20250909
    TEST_END = 20251231
    
    train_data = data_split(data, TRAIN_START, TRAIN_END)
    val_data = data_split(data, VAL_START, VAL_END)
    test_data = data_split(data, TEST_START, TEST_END)
    
    print(f"  訓練期: {train_data.shape}, 驗證期: {val_data.shape}, 測試期: {test_data.shape}")
    
    # 訓練 PPO
    print("\n[2/4] 訓練 PPO 模型...")
    env_train = DummyVecEnv([lambda: StockEnvTrain(train_data)])
    
    model = PPO(
        'MlpPolicy',
        env_train,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=0
    )
    
    callback = ProgressCallback(check_freq=10000)
    train_start = datetime.now()
    model.learn(total_timesteps=200_000, callback=callback)
    
    # 保存模型
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"trained_models/ppo_optimized_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    model.save(os.path.join(output_dir, "PPO_optimized"))
    print(f"  ✓ 模型已保存至: {output_dir}")
    
    # 驗證
    print("\n[3/4] 驗證最佳風險閾值...")
    thresholds = [250, 350, 500, float('inf')]
    val_results = []
    
    for threshold in thresholds:
        env_val = StockEnvValidation(val_data, turbulence_threshold=threshold)
        obs = env_val.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env_val.step(action)
        
        final = env_val.asset_memory[-1]
        rate = (final - 100_000_000) / 100_000_000 * 100
        val_results.append({'threshold': threshold, 'return_rate': rate})
        print(f"  Threshold={threshold}: {rate:.2f}%")
        
    best_threshold = sorted(val_results, key=lambda x: x['return_rate'], reverse=True)[0]['threshold']
    print(f"  ✓ 最佳閾值: {best_threshold}")
    
    # 測試
    print("\n[4/4] 測試最終表現...")
    env_test = StockEnvValidation(test_data, turbulence_threshold=best_threshold, iteration='test_final')
    obs = env_test.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env_test.step(action)
        
    final = env_test.asset_memory[-1]
    profit = final - 100_000_000
    print(f"\n最終結果:")
    print(f"  資產: {final:,.0f}")
    print(f"  獲利: {profit:,.0f} ({(profit/100000000)*100:.2f}%)")
    print(f"  結果已存於: results/test_test_final.csv")

if __name__ == "__main__":
    main()