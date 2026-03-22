"""
example_1_simple_ma.py - 简单双均线策略示例

策略逻辑：
  - 买入信号：短期均线(MAS)上穿长期均线(MAL)
  - 卖出信号：短期均线(MAS)下穿长期均线(MAL)

使用说明：
  python examples/example_1_simple_ma.py
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime

from backtest import BacktestConfig
from backtest.backtest_config import Config, StrategyConfig
from data.tushare_client import TushareClient
from data.data_fetcher import DataFetcher


def calculate_ma(data: pd.DataFrame, short: int, long: int) -> pd.DataFrame:
    """
    计算简单移动平均线
    
    Args:
        data: 包含 'close' 列的价格数据
        short: 短期均线周期
        long: 长期均线周期
        
    Returns:
        添加了 ma_short, ma_long, signal 列的 DataFrame
    """
    df = data.copy()
    df["ma_short"] = df["close"].rolling(window=short).mean()
    df["ma_long"] = df["close"].rolling(window=long).mean()
    
    # 金叉买入信号：short 上穿 long
    df["signal"] = 0
    df.loc[df["ma_short"] > df["ma_long"], "signal"] = 1
    # 死叉卖出信号：short 下穿 long
    df.loc[df["ma_short"] < df["ma_long"], "signal"] = -1
    
    return df


def generate_signals(data: pd.DataFrame, config: BacktestConfig, strategy_config: StrategyConfig) -> pd.DataFrame:
    """
    生成交易信号
    
    Args:
        data: OHLCV 数据
        config: 回测配置
        strategy_config: 策略配置
        
    Returns:
        带交易信号的 DataFrame
    """
    ma_short = strategy_config.ma_short
    ma_long = strategy_config.ma_long
    
    # 计算均线
    df = calculate_ma(data, short=ma_short, long=ma_long)
    
    # 生成交易信号 (1=买入, -1=卖出, 0=持有)
    df["trade_signal"] = 0
    
    # 找到金叉和死叉点
    prev_signal = df["signal"].shift(1).fillna(0)
    
    # 金叉点：前一天 short <= long，当天 short > long
    golden_cross = (df["signal"] == 1) & (prev_signal != 1)
    # 死叉点：前一天 short >= long，当天 short < long
    death_cross = (df["signal"] == -1) & (prev_signal != -1)
    
    df.loc[golden_cross, "trade_signal"] = 1   # 买入
    df.loc[death_cross, "trade_signal"] = -1   # 卖出
    
    return df


def simulate_backtest(signals: pd.DataFrame, config: BacktestConfig) -> dict:
    """
    简化版回测模拟（仅用于示例演示）

    实际项目中应使用 BacktestEngine + Backtrader

    Returns:
        回测结果字典
    """
    cash = config.initial_cash
    position = 0  # 初始化持仓
    equity_curve = []

    for idx, row in signals.iterrows():
        if row["trade_signal"] == 1 and position == 0:
            # 买入
            shares = int(cash / (row["close"] * (1 + config.commission_rate)))
            if shares > 0:
                cost = shares * row["close"] * (1 + config.commission_rate)
                cash -= cost
                position = shares
        
        elif row["trade_signal"] == -1 and position > 0:
            # 卖出
            revenue = position * row["close"] * (1 - config.commission_rate - config.stamp_tax_rate)
            cash += revenue
            position = 0
        
        # 当日权益
        equity = cash + position * row["close"]
        equity_curve.append(equity)
    
    # 最后如果还有持仓，按最后收盘价平仓
    if position > 0:
        final_price = signals.iloc[-1]["close"]
        cash += position * final_price * (1 - config.commission_rate - config.stamp_tax_rate)
        position = 0
    
    signals["equity"] = equity_curve
    
    # 计算收益率
    returns = pd.Series(equity_curve).pct_change().dropna()
    
    results = {
        "final_equity": cash,
        "total_return": (cash - config.initial_cash) / config.initial_cash,
        "equity_curve": equity_curve,
        "returns": returns,
        "trades": len(signals[signals["trade_signal"] != 0]),
        "signals": signals,
    }
    
    return results


def print_results(results: dict, config: BacktestConfig, strategy_config: StrategyConfig) -> None:
    """打印回测结果"""
    print("\n" + "=" * 50)
    print(f"策略: 双均线策略 (MA{strategy_config.ma_short}/MA{strategy_config.ma_long})")
    print(f"回测期: {config.start_date} ~ {config.end_date}")
    print("=" * 50)
    print(f"初始资金:      ¥{config.initial_cash:,.2f}")
    print(f"最终权益:      ¥{results['final_equity']:,.2f}")
    print(f"总收益率:      {results['total_return']:.2%}")
    print(f"交易次数:      {results['trades']}")
    
    # 计算年化收益率
    returns = results["returns"]
    if len(returns) > 0:
        # 简化年化：假设252个交易日
        annual_return = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_volatility if annual_volatility > 0 else 0
        max_drawdown = calculate_max_drawdown(results["equity_curve"])
        
        print(f"年化收益率:    {annual_return:.2%}")
        print(f"年化波动率:   {annual_volatility:.2%}")
        print(f"夏普比率:      {sharpe:.2f}")
        print(f"最大回撤:      {max_drawdown:.2%}")
    
    print("=" * 50)


def calculate_max_drawdown(equity_curve: list) -> float:
    """计算最大回撤"""
    equity = pd.Series(equity_curve)
    peak = equity.expanding(min_periods=1).max()
    drawdown = (equity - peak) / peak
    return abs(drawdown.min())


def create_sample_data() -> pd.DataFrame:
    """
    从 Tushare 获取真实数据（使用缓存）
    
    实际使用时应从 Tushare 或 DataFetcher 获取真实数据
    """
    print("从 Tushare 获取数据（带缓存）...")
    config = Config.from_yaml(
        os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    )

    client = TushareClient(token=config.tushare.token)
    
    # 使用 DataFetcher（启用缓存）
    fetcher = DataFetcher(
        client,
        cache_dir=os.path.join(os.path.dirname(__file__), "..", "data", "cache"),
        enable_cache=True,
        cache_ttl=86400  # 24小时
    )

    # 获取真实日线数据
    df = fetcher.get_daily_data(
        ts_code="000001.SZ",
        start_date=config.backtest.start_date,
        end_date=config.backtest.end_date
    )

    # 重命名列以匹配示例格式
    df = df.rename(columns={
        'trade_date': 'trade_date',
        'ts_code': 'ts_code',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume'
    })

    print(f"数据量: {len(df)} 条")
    return df


def main():
    """主函数"""
    print("=" * 50)
    print("示例1: 双均线策略（真实数据）")
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

    # 从 Tushare 获取数据
    print("从 Tushare 获取数据（带缓存）...")
    data = fetcher.get_daily_data(
        ts_code="000001.SZ",
        start_date=config.backtest.start_date,
        end_date=config.backtest.end_date
    )
    print(f"数据量: {len(data)} 条")
    
    # 生成信号
    print("生成交易信号...")
    signals = generate_signals(data, config.backtest, config.strategy)
    
    # 执行回测
    print("执行回测...")
    results = simulate_backtest(signals, config.backtest)
    
    # 打印结果
    print_results(results, config.backtest, config.strategy)
    
    # 打印缓存统计
    if fetcher.cache:
        stats = fetcher.cache.get_stats()
        print()
        print("=" * 50)
        print("📦 缓存统计")
        print("=" * 50)
        print(f"命中率:   {stats['hit_rate']:.1%}")
        print(f"命中:     {stats['hits']}")
        print(f"未命中:   {stats['misses']}")
        print(f"磁盘缓存: {stats['disk_items']} 项")
    
    # 保存信号到CSV（可选）
    output_path = os.path.join(config.output.results_dir, "ma_signals.csv")
    os.makedirs(config.output.results_dir, exist_ok=True)
    signals.to_csv(output_path, index=False)
    print(f"\n信号已保存: {output_path}")
    
    return results


if __name__ == "__main__":
    main()
