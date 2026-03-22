"""
example_2_factor_strategy.py - 因子策略示例

策略逻辑：
  - 选股因子：PE、ROE、成交量突变
  - 多因子综合评分
  - 持有评分最高的几只股票，定期调仓

使用说明：
  python examples/example_2_factor_strategy.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime

from backtest.backtest_config import Config, BacktestConfig
from data.tushare_client import TushareClient
from data.data_fetcher import DataFetcher


def calculate_factors(data: pd.DataFrame) -> pd.DataFrame:
    """
    计算多因子指标
    
    Args:
        data: OHLCV 数据
        
    Returns:
        添加了因子列的 DataFrame
    """
    df = data.copy()
    
    # 1. 估值因子 - PE (简化版：取固定值，实际应从财务数据获取)
    df["pe"] = np.random.uniform(8, 30, len(df))
    
    # 2. 盈利能力因子 - ROE (简化版)
    df["roe"] = np.random.uniform(0.05, 0.25, len(df))
    
    # 3. 成长因子 - 营收增速 (简化版)
    df["revenue_growth"] = np.random.uniform(-0.1, 0.5, len(df))
    
    # 4. 动量因子 - 20日收益率
    df["momentum_20d"] = df["close"].pct_change(20)
    
    # 5. 成交量突变因子 - 量比
    df["vol_ma5"] = df["volume"].rolling(window=5).mean()
    df["vol_ma20"] = df["volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["vol_ma5"] / df["vol_ma20"]
    
    # 6. 趋势因子 - MA角度
    df["ma20"] = df["close"].rolling(window=20).mean()
    df["ma_slope"] = df["ma20"].pct_change(5)
    
    return df


def factor_scoring(df: pd.DataFrame) -> pd.DataFrame:
    """
    因子打分并计算综合评分
    
    评分方法：Z-Score 标准化后加权求和
    
    Args:
        df: 包含因子列的 DataFrame
        
    Returns:
        添加了 factor_score 列的 DataFrame
    """
    result = df.copy()
    
    # 定义因子权重和方向 (+1=越大越好, -1=越小越好)
    factor_weights = {
        "pe": (-1, 0.15),      # PE 越低越好
        "roe": (1, 0.25),      # ROE 越高越好
        "revenue_growth": (1, 0.20),  # 营收增速越高越好
        "momentum_20d": (1, 0.15),   # 动量越高越好
        "volume_ratio": (1, 0.10),    # 成交量放大越好
        "ma_slope": (1, 0.15),        # 趋势向上越好
    }
    
    scores = pd.DataFrame(index=result.index)
    
    for factor, (direction, weight) in factor_weights.items():
        if factor not in result.columns:
            continue
        
        # 去极值 (3倍标准差截断)
        col = result[factor].copy()
        mean = col.mean()
        std = col.std()
        col = col.clip(lower=mean - 3*std, upper=mean + 3*std)
        
        # Z-Score 标准化
        if std > 0:
            zscore = (col - mean) / std
        else:
            zscore = 0
        
        # 方向调整
        scores[factor] = zscore * direction * weight
    
    # 综合评分
    result["factor_score"] = scores.sum(axis=1)
    
    # 百分位排名
    result["factor_rank"] = result["factor_score"].rank(pct=True)
    
    return result


def generate_portfolio(signals: pd.DataFrame, 
                       top_n: int = 5,
                       hold_days: int = 20) -> pd.DataFrame:
    """
    根据因子评分生成调仓组合
    
    Args:
        signals: 带评分的信号数据
        top_n: 每次持有评分最高的几只股票
        hold_days: 持有周期(天)
        
    Returns:
        持仓记录 DataFrame
    """
    # 模拟多只股票的情况
    results = []
    
    # 按日期分组（简化：假设每天都有信号）
    for date, group in signals.groupby("trade_date"):
        # 选择评分最高的 top_n 只
        top_stocks = group.nlargest(top_n, "factor_score")
        
        for _, row in top_stocks.iterrows():
            results.append({
                "trade_date": date,
                "ts_code": row["ts_code"],
                "factor_score": row["factor_score"],
                "close": row["close"],
            })
    
    return pd.DataFrame(results)


def simulate_factor_backtest(data: pd.DataFrame, 
                             config: BacktestConfig,
                             top_n: int = 5) -> dict:
    """
    因子策略回测模拟
    
    Args:
        data: 原始行情数据
        config: 回测配置
        top_n: 持仓数量
        
    Returns:
        回测结果字典
    """
    # 计算因子
    print("计算多因子...")
    df = calculate_factors(data)
    
    # 因子打分
    df = factor_scoring(df)
    
    # 生成调仓信号
    print("生成调仓组合...")
    portfolio = generate_portfolio(df, top_n=top_n)
    
    # 模拟资金曲线
    cash = config.initial_cash
    position_value = 0
    equity_curve = []
    
    # 简化模拟：每日按收盘价计算
    dates = sorted(df["trade_date"].unique())
    
    # 模拟每日调仓（简化版）
    rebalance_dates = dates[::20]  # 每20天调仓一次
    
    current_holdings = {}  # {ts_code: shares}
    
    for i, date in enumerate(dates):
        day_data = df[df["trade_date"] == date]
        
        # 调仓日
        if date in rebalance_dates and i > 0:
            # 卖出所有持仓
            for ts, shares in current_holdings.items():
                price = day_data[day_data["ts_code"] == ts]["close"].values
                if len(price) > 0:
                    cash += shares * price[0] * (1 - config.commission_rate - config.stamp_tax_rate)
            current_holdings = {}
            
            # 买入新持仓（等权分配）
            top_stocks = day_data.nlargest(top_n, "factor_score")
            per_stock_cash = cash / len(top_stocks)
            
            for _, row in top_stocks.iterrows():
                shares = int(per_stock_cash / row["close"] / 100) * 100
                if shares > 0:
                    cost = shares * row["close"] * (1 + config.commission_rate)
                    if cost <= per_stock_cash:
                        current_holdings[row["ts_code"]] = shares
                        cash -= cost
        
        # 计算当日权益
        pos_val = 0
        for ts, shares in current_holdings.items():
            price = day_data[day_data["ts_code"] == ts]["close"].values
            if len(price) > 0:
                pos_val += shares * price[0]
        
        equity = cash + pos_val
        equity_curve.append({
            "date": date,
            "equity": equity,
            "cash": cash,
            "position_value": pos_val,
        })
    
    equity_df = pd.DataFrame(equity_curve)
    equity_df["returns"] = equity_df["equity"].pct_change()
    
    return {
        "equity_curve": equity_df,
        "portfolio": portfolio,
        "factor_data": df,
        "final_equity": equity_df.iloc[-1]["equity"],
        "total_return": (equity_df.iloc[-1]["equity"] - config.initial_cash) / config.initial_cash,
    }


def print_factor_results(results: dict, config: BacktestConfig) -> None:
    """打印因子策略回测结果"""
    print("\n" + "=" * 50)
    print(f"策略: 多因子选股策略 (Top {5})")
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


def create_multi_stock_sample() -> pd.DataFrame:
    """
    创建多只股票的示例数据
    
    真实场景替换为 DataFetcher.fetch_all_stocks()
    """
    np.random.seed(42)
    n = 500  # 每个股票500个交易日
    stocks = ["000001.SZ", "000002.SZ", "000004.SZ", "000005.SZ", "000006.SZ"]
    
    all_data = []
    
    for ts_code in stocks:
        dates = pd.date_range(start="2020-01-01", periods=n, freq="B")
        
        # 每只股票不同的随机游走参数
        mu = np.random.uniform(-0.0002, 0.001)
        sigma = np.random.uniform(0.015, 0.025)
        
        returns = np.random.normal(mu, sigma, n)
        price = 10.0 * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            "trade_date": dates,
            "ts_code": ts_code,
            "open": price * (1 + np.random.uniform(-0.005, 0.005, n)),
            "high": price * (1 + np.random.uniform(0.001, 0.015, n)),
            "low": price * (1 + np.random.uniform(-0.015, -0.001, n)),
            "close": price,
            "volume": np.random.randint(1e6, 1e8, n),
        })
        
        all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True)


def main():
    """主函数"""
    print("=" * 50)
    print("示例2: 多因子选股策略")
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
    
    # 创建示例数据（模拟数据，用于演示因子策略）
    print("生成多只股票示例数据...")
    data = create_multi_stock_sample()
    print(f"数据量: {len(data)} 条, 股票数: {data['ts_code'].nunique()}")
    
    # 执行因子策略回测
    results = simulate_factor_backtest(data, config.backtest, top_n=3)
    
    # 打印结果
    print_factor_results(results, config.backtest)
    
    # 保存结果
    os.makedirs(config.output.results_dir, exist_ok=True)
    
    # 保存因子数据
    factor_path = os.path.join(config.output.results_dir, "factor_scores.csv")
    results["factor_data"].to_csv(factor_path, index=False)
    print(f"\n因子评分已保存: {factor_path}")
    
    # 保存持仓记录
    portfolio_path = os.path.join(config.output.results_dir, "factor_portfolio.csv")
    results["portfolio"].to_csv(portfolio_path, index=False)
    print(f"持仓记录已保存: {portfolio_path}")

    # 保存资金曲线
    equity_path = os.path.join(config.output.results_dir, "factor_equity.csv")
    results["equity_curve"].to_csv(equity_path, index=False)
    print(f"资金曲线已保存: {equity_path}")

    return results


if __name__ == "__main__":
    main()
