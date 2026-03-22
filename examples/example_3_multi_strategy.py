"""
example_3_multi_strategy.py - 多策略组合示例

策略组合方式：
  1. 子策略1: 双均线策略 (趋势跟踪)
  2. 子策略2: RSI超卖策略 (均值回归)
  3. 子策略3: 布林带突破策略 (波动率)

组合方式:
  - 等权重分配资金
  - 或按风险平价分配
  - 或按信号数量动态分配

使用说明：
  python examples/example_3_multi_strategy.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from typing import List, Dict

from backtest.backtest_config import Config, BacktestConfig
from data.tushare_client import TushareClient
from data.data_fetcher import DataFetcher


# ============== 子策略定义 ==============

class MAStrategy:
    """双均线策略"""

    def __init__(self, short: int = 5, long: int = 20):
        self.short = short
        self.long = long
        self.name = f"MA_{short}_{long}"

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成均线策略信号"""
        df = data.copy()
        df["ma_short"] = df["close"].rolling(self.short).mean()
        df["ma_long"] = df["close"].rolling(self.long).mean()

        # 持仓信号
        df["position"] = 0
        df.loc[df["ma_short"] > df["ma_long"], "position"] = 1

        # 金叉/死叉交易信号
        df["trade_signal"] = df["position"].diff().fillna(0)
        df.loc[df["trade_signal"] == 1, "trade_signal"] = 1
        df.loc[df["trade_signal"] == -1, "trade_signal"] = -1

        return df[["trade_date", "ts_code", "close", "position", "trade_signal"]]


class RSIStrategy:
    """RSI 超买超卖策略"""

    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.name = f"RSI_{period}"

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成 RSI 策略信号"""
        df = data.copy()

        # 计算 RSI
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.period).mean()

        rs = gain / loss.replace(0, np.inf)
        df["rsi"] = 100 - (100 / (1 + rs))

        # 持仓信号：RSI < 30 买入，RSI > 70 卖出
        df["position"] = 0
        df.loc[df["rsi"] < self.oversold, "position"] = 1
        df.loc[df["rsi"] > self.overbought, "position"] = 0

        df["trade_signal"] = df["position"].diff().fillna(0)
        df.loc[df["trade_signal"] == 1, "trade_signal"] = 1
        df.loc[df["trade_signal"] == -1, "trade_signal"] = -1

        return df[["trade_date", "ts_code", "close", "position", "trade_signal", "rsi"]]


class BollingerStrategy:
    """布林带突破策略"""

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        self.period = period
        self.std_dev = std_dev
        self.name = f"BB_{period}_{std_dev}"

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成布林带策略信号"""
        df = data.copy()

        # 计算布林带
        df["bb_mid"] = df["close"].rolling(self.period).mean()
        df["bb_std"] = df["close"].rolling(self.period).std()
        df["bb_upper"] = df["bb_mid"] + self.std_dev * df["bb_std"]
        df["bb_lower"] = df["bb_mid"] - self.std_dev * df["bb_std"]

        # 持仓信号：价格触及下轨买入，触及上轨卖出
        df["position"] = 0
        df.loc[df["close"] < df["bb_lower"], "position"] = 1
        df.loc[df["close"] > df["bb_upper"], "position"] = 0

        df["trade_signal"] = df["position"].diff().fillna(0)
        df.loc[df["trade_signal"] == 1, "trade_signal"] = 1
        df.loc[df["trade_signal"] == -1, "trade_signal"] = -1

        return df[["trade_date", "ts_code", "close", "position", "trade_signal",
                   "bb_upper", "bb_lower"]]


# ============== 策略组合器 ==============

class StrategyComposer:
    """多策略组合器"""

    def __init__(self, strategies: List, weights: List[float] = None):
        """
        初始化组合器

        Args:
            strategies: 子策略列表
            weights: 各策略权重，默认等权
        """
        self.strategies = strategies
        self.n = len(strategies)

        if weights is None:
            self.weights = [1.0 / self.n] * self.n
        else:
            assert len(weights) == self.n, "权重数量必须等于策略数量"
            total = sum(weights)
            self.weights = [w / total for w in weights]

    def combine_by_voting(self, signals_list: List[pd.DataFrame]) -> pd.DataFrame:
        """
        按多数投票法合并信号

        Args:
            signals_list: 各策略信号列表

        Returns:
            合并后的信号 DataFrame
        """
        n = len(signals_list)

        # 以第一个信号为基础
        result = signals_list[0][["trade_date", "ts_code", "close", "trade_signal"]].copy()
        result = result.rename(columns={"trade_signal": "sig_0"})

        for i in range(1, n):
            sig = signals_list[i][["trade_date", "ts_code", "trade_signal"]].copy()
            result = result.merge(
                sig.rename(columns={"trade_signal": f"sig_{i}"}),
                on=["trade_date", "ts_code"],
                how="outer"
            )

        # 加权投票
        sig_cols = [f"sig_{i}" for i in range(n)]
        for i in range(n):
            result[f"weighted_sig_{i}"] = result[sig_cols[i]].fillna(0) * self.weights[i]

        weighted_cols = [f"weighted_sig_{i}" for i in range(n)]
        result["vote_score"] = result[weighted_cols].sum(axis=1)

        # 超过50%权重则执行
        result["final_signal"] = 0
        result.loc[result["vote_score"] > 0.5, "final_signal"] = 1
        result.loc[result["vote_score"] < -0.5, "final_signal"] = -1

        return result


# ============== 组合策略回测 ==============

def multi_strategy_backtest(data: pd.DataFrame,
                             config: BacktestConfig,
                             strategies: List,
                             weights: List[float] = None) -> dict:
    """
    多策略组合回测

    Args:
        data: 行情数据
        config: 回测配置
        strategies: 子策略列表
        weights: 策略权重

    Returns:
        回测结果字典
    """
    # 1. 生成各策略信号
    print("生成各子策略信号...")
    signals_list = []
    for strat in strategies:
        sig = strat.generate(data)
        sig["strategy"] = strat.name
        signals_list.append(sig)
        print(f"  - {strat.name}: {len(sig)} 条信号")

    # 2. 合并信号
    print("合并策略信号...")
    composer = StrategyComposer(strategies, weights)
    combined = composer.combine_by_voting(signals_list)

    # 3. 模拟回测
    print("执行组合策略回测...")
    cash = config.initial_cash
    position = 0
    entry_price = 0
    equity_curve = []

    sorted_data = combined.sort_values("trade_date")

    for idx, row in sorted_data.iterrows():
        trade_signal = row.get("final_signal", 0)

        if trade_signal == 1 and position == 0:
            # 买入
            shares = int(cash / row["close"] / 100) * 100
            if shares > 0:
                cost = shares * row["close"] * (1 + config.commission_rate)
                if cost <= cash:
                    position = shares
                    entry_price = row["close"]
                    cash -= cost

        elif trade_signal == -1 and position > 0:
            # 卖出
            revenue = position * row["close"] * (1 - config.commission_rate - config.stamp_tax_rate)
            cash += revenue
            position = 0
            entry_price = 0

        # 止损检查
        if position > 0 and entry_price > 0:
            loss_pct = (row["close"] - entry_price) / entry_price
            if loss_pct < -config.stop_loss:
                revenue = position * row["close"] * (1 - config.commission_rate - config.stamp_tax_rate)
                cash += revenue
                position = 0
                entry_price = 0

        equity = cash + position * row["close"]
        equity_curve.append({
            "date": row["trade_date"],
            "equity": equity,
            "position": position,
        })

    # 平仓
    if position > 0:
        final_price = sorted_data.iloc[-1]["close"]
        cash += position * final_price * (1 - config.commission_rate - config.stamp_tax_rate)

    equity_df = pd.DataFrame(equity_curve)
    equity_df["returns"] = equity_df["equity"].pct_change()

    return {
        "equity_curve": equity_df,
        "combined_signals": combined,
        "strategies": [s.name for s in strategies],
        "weights": weights or [1.0/len(strategies)] * len(strategies),
        "final_equity": equity_df.iloc[-1]["equity"],
        "total_return": (equity_df.iloc[-1]["equity"] - config.initial_cash) / config.initial_cash,
    }


def print_multi_results(results: dict, config: BacktestConfig) -> None:
    """打印多策略组合回测结果"""
    strategies = ", ".join(results["strategies"])

    print("\n" + "=" * 50)
    print(f"策略: 多策略组合 ({strategies})")
    print(f"权重: {results['weights']}")
    print(f"回测期: {config.start_date} ~ {config.end_date}")
    print("=" * 50)
    print(f"初始资金:      ¥{config.initial_cash:,.2f}")
    print(f"最终权益:      ¥{results['final_equity']:,.2f}")
    print(f"总收益率:      {results['total_return']:.2%}")

    equity_df = results["equity_curve"]
    returns = equity_df["returns"].dropna()

    if len(returns) > 0:
        annual_return = returns.mean() * 252
        annual_vol = returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        max_dd = calculate_max_drawdown(equity_df["equity"].tolist())

        print(f"年化收益率:    {annual_return:.2%}")
        print(f"年化波动率:    {annual_vol:.2%}")
        print(f"夏普比率:      {sharpe:.2f}")
        print(f"最大回撤:      {max_dd:.2%}")

    print("=" * 50)


def calculate_max_drawdown(equity_curve: list) -> float:
    """计算最大回撤"""
    equity = pd.Series(equity_curve)
    peak = equity.expanding(min_periods=1).max()
    drawdown = (equity - peak) / peak
    return abs(drawdown.min())


def create_sample_data() -> pd.DataFrame:
    """创建示例数据"""
    np.random.seed(42)
    n = 500
    dates = pd.date_range(start="2020-01-01", periods=n, freq="B")

    returns = np.random.normal(0.0005, 0.02, n)
    price = 10.0 * np.exp(np.cumsum(returns))

    return pd.DataFrame({
        "trade_date": dates,
        "ts_code": "000001.SZ",
        "open": price * (1 + np.random.uniform(-0.005, 0.005, n)),
        "high": price * (1 + np.random.uniform(0.001, 0.015, n)),
        "low": price * (1 + np.random.uniform(-0.015, -0.001, n)),
        "close": price,
        "volume": np.random.randint(1e6, 1e8, n),
    })


def main():
    """主函数"""
    print("=" * 50)
    print("示例3: 多策略组合")
    print("=" * 50)

    # 加载配置
    config = Config.from_yaml(
        os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    )

    # 初始化 DataFetcher（启用缓存）
    client = TushareClient(token=config.tushare.token)
    fetcher = DataFetcher(
        client,
        cache_dir=os.path.join(os.path.dirname(__file__), "..", "data", "cache"),
        enable_cache=True,
        cache_ttl=86400
    )

    # 创建示例数据（模拟数据，用于演示策略组合）
    print("生成示例数据...")
    data = create_sample_data()
    print(f"数据量: {len(data)} 条")

    # 定义子策略
    strategies = [
        MAStrategy(short=5, long=20),
        RSIStrategy(period=14, oversold=30, overbought=70),
        BollingerStrategy(period=20, std_dev=2.0),
    ]

    # 等权重组合
    weights = [1.0/3, 1.0/3, 1.0/3]

    # 执行回测
    results = multi_strategy_backtest(data, config.backtest, strategies, weights)

    # 打印结果
    print_multi_results(results, config.backtest)

    # 保存结果
    os.makedirs(config.output.results_dir, exist_ok=True)

    signals_path = os.path.join(config.output.results_dir, "multi_strategy_signals.csv")
    results["combined_signals"].to_csv(signals_path, index=False)
    print(f"\n合并信号已保存: {signals_path}")

    # 保存资金曲线
    equity_path = os.path.join(config.output.results_dir, "multi_strategy_equity.csv")
    results["equity_curve"].to_csv(equity_path, index=False)
    print(f"资金曲线已保存: {equity_path}")

    return results


if __name__ == "__main__":
    main()
