# -*- coding: utf-8 -*-
"""
example_4_factor_library.py - 因子库完整示例

完整演示策略层和回测层的集成：
1. 使用 technical_factors 计算技术因子 (10+ 因子)
2. 使用 fundamental_factors 计算基本面因子 (5+ 因子)
3. 使用 FactorLibrary 管理因子
4. 生成交易信号
5. 使用 PerformanceAnalyzer 分析性能
6. 使用 RiskAnalyzer 分析风险
7. 使用 ResultVisualizer 可视化
8. 使用 DataStorage 保存结果

使用说明：
  python3 example_4_factor_library.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============== 策略层 ==============
from strategy.technical_factors import (
    MAFactor, EMAFactor, RSIFactor, MACDFactor, 
    KDJFactor, ATRFactor, BBandsFactor, ADXFactor,
    ROCFactor, CCRFactor, OBVFactor
)
from strategy.fundamental_factors import (
    PEFactor, ROEFactor, ROAFactor, 
    ProfitGrowthFactor, RevenueGrowthFactor,
    GrossMarginFactor, NetMarginFactor
)
from strategy.factor_library import FactorLibrary

# ============== 回测层 ==============
from backtest import PerformanceAnalyzer, RiskAnalyzer, ResultVisualizer

# ============== 数据层 ==============
from data.data_storage import DataStorage


def generate_sample_data():
    """生成模拟数据"""
    print("📊 生成模拟数据...")
    
    # 生成200天的模拟数据
    dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
    dates = [d for d in dates if d.weekday() < 5][:180]
    
    np.random.seed(42)
    
    # 生成带趋势的价格数据
    base_price = 100
    trend = np.linspace(0, 30, len(dates))
    noise = np.random.randn(len(dates)) * 3
    prices = base_price + trend + noise
    prices = np.maximum(prices, 10)
    
    df = pd.DataFrame({
        'ts_code': '000001.SZ',
        'trade_date': [d.strftime('%Y%m%d') for d in dates],
        'open': [p * (1 + np.random.randn() * 0.01) for p in prices],
        'high': [p * (1 + abs(np.random.randn()) * 0.02) for p in prices],
        'low': [p * (1 - abs(np.random.randn()) * 0.02) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, len(dates)),
        'amount': [p * v for p, v in zip(prices, np.random.randint(1000000, 10000000, len(dates)))],
    })
    
    print(f"   生成 {len(df)} 条数据")
    return df


def calculate_technical_factors(df):
    """计算技术因子"""
    print("\n📈 计算技术因子...")
    
    factors_result = {}
    
    # 移动平均线
    for period in [5, 10, 20, 60]:
        ma = MAFactor({'period': period})
        factors_result[f'ma{period}'] = ma.calculate(df)
    print(f"   ✓ MA (5,10,20,60)")
    
    # EMA
    for period in [12, 26]:
        ema = EMAFactor({'period': period})
        factors_result[f'ema{period}'] = ema.calculate(df)
    print(f"   ✓ EMA (12,26)")
    
    # RSI
    rsi = RSIFactor({'period': 14})
    factors_result['rsi'] = rsi.calculate(df)
    print(f"   ✓ RSI (14)")
    
    # MACD
    macd = MACDFactor({'fast': 12, 'slow': 26, 'signal': 9})
    factors_result['macd'] = macd.calculate(df)
    print(f"   ✓ MACD (12,26,9)")
    
    # KDJ
    kdj = KDJFactor({'period': 9, 'k_period': 3, 'd_period': 3})
    factors_result['kdj_k'] = kdj.calculate(df)
    factors_result['kdj_d'] = kdj.calculate(df)  # KDJ返回的是 Series
    print(f"   ✓ KDJ")
    
    # ATR
    atr = ATRFactor({'period': 14})
    factors_result['atr'] = atr.calculate(df)
    print(f"   ✓ ATR (14)")
    
    # 布林带
    boll = BBandsFactor({'period': 20, 'std_dev': 2})
    factors_result['bb_upper'] = boll.calculate(df)
    factors_result['bb_middle'] = boll.calculate(df)
    factors_result['bb_lower'] = boll.calculate(df)
    print(f"   ✓ Bollinger Bands")
    
    # ADX
    adx = ADXFactor({'period': 14})
    factors_result['adx'] = adx.calculate(df)
    print(f"   ✓ ADX (14)")
    
    # ROC
    roc = ROCFactor({'period': 12})
    factors_result['roc'] = roc.calculate(df)
    print(f"   ✓ ROC (12)")
    
    # CCR
    ccr = CCRFactor({'period': 12})
    factors_result['ccr'] = ccr.calculate(df)
    print(f"   ✓ CCR (12)")
    
    # OBV (需要 vol 列)
    df_copy = df.copy()
    df_copy['vol'] = df_copy['volume']  # 重命名为 vol
    obv = OBVFactor({})
    factors_result['obv'] = obv.calculate(df_copy)
    print(f"   ✓ OBV")
    
    # 合并所有因子
    factors_df = pd.DataFrame(factors_result)
    print(f"   共计算 {len(factors_df.columns)} 个技术因子")
    
    return factors_df


def calculate_fundamental_factors(df):
    """计算基本面因子"""
    print("\n📊 计算基本面因子...")
    
    factors_result = {}
    
    # PE
    pe = PEFactor({})
    factors_result['pe'] = pe.calculate(df)
    print(f"   ✓ PE")
    
    # ROE
    roe = ROEFactor({})
    factors_result['roe'] = roe.calculate(df)
    print(f"   ✓ ROE")
    
    # ROA
    roa = ROAFactor({})
    factors_result['roa'] = roa.calculate(df)
    print(f"   ✓ ROA")
    
    # 净利润增长率
    profit_growth = ProfitGrowthFactor({})
    factors_result['profit_growth'] = profit_growth.calculate(df)
    print(f"   ✓ Profit Growth")
    
    # 营收增长率
    revenue_growth = RevenueGrowthFactor({})
    factors_result['revenue_growth'] = revenue_growth.calculate(df)
    print(f"   ✓ Revenue Growth")
    
    # 毛利率
    gross_margin = GrossMarginFactor({})
    factors_result['gross_margin'] = gross_margin.calculate(df)
    print(f"   ✓ Gross Margin")
    
    # 净利率
    net_margin = NetMarginFactor({})
    factors_result['net_margin'] = net_margin.calculate(df)
    print(f"   ✓ Net Margin")
    
    # 合并所有因子
    factors_df = pd.DataFrame(factors_result)
    print(f"   共计算 {len(factors_df.columns)} 个基本面因子")
    
    return factors_df


def use_factor_library():
    """使用 FactorLibrary 管理因子"""
    print("\n📚 使用 FactorLibrary...")
    
    library = FactorLibrary()
    
    # 注册因子类（简化示例：只注册几个）
    library.register_factor(MAFactor, category="技术因子")
    library.register_factor(RSIFactor, category="技术因子")
    library.register_factor(MACDFactor, category="技术因子")
    library.register_factor(PEFactor, category="基本面因子")
    library.register_factor(ROEFactor, category="基本面因子")
    
    # 列出所有因子
    tech_factors = library.list_factors("技术因子")
    fund_factors = library.list_factors("基本面因子")
    
    print(f"   ✓ 技术因子: {tech_factors}")
    print(f"   ✓ 基本面因子: {fund_factors}")
    
    return library


def generate_signals(df, tech_factors):
    """生成交易信号"""
    print("\n🎯 生成交易信号...")
    
    result = df.copy()
    
    # 合并技术因子
    for col in tech_factors.columns:
        result[col] = tech_factors[col].values
    
    # 策略：MA5 > MA20 且 RSI < 70 买入，MA5 < MA20 且 RSI > 30 卖出
    ma5 = result['ma5']
    ma20 = result['ma20']
    rsi = result['rsi']
    
    # 买入信号：MA5 上穿 MA20 且 RSI 未超买
    buy_signal = (ma5 > ma20) & (ma5.shift(1) <= ma20.shift(1)) & (rsi < 70)
    
    # 卖出信号：MA5 下穿 MA20 且 RSI 未超卖
    sell_signal = (ma5 < ma20) & (ma5.shift(1) >= ma20.shift(1)) & (rsi > 30)
    
    result['signal'] = 0
    result.loc[buy_signal, 'signal'] = 1
    result.loc[sell_signal, 'signal'] = -1
    
    # 计算策略收益
    result['return'] = result['close'].pct_change()
    result['strategy_return'] = result['signal'].shift(1) * result['return']
    
    # 计算权益曲线
    result['equity'] = 100000 * (1 + result['strategy_return'].fillna(0)).cumprod()
    
    print(f"   买入信号: {(result['signal'] == 1).sum()} 次")
    print(f"   卖出信号: {(result['signal'] == -1).sum()} 次")
    
    return result


def analyze_performance(result):
    """分析策略性能"""
    print("\n📊 分析策略性能...")
    
    equity = result['equity']
    returns = result['strategy_return'].fillna(0)
    
    # 使用 PerformanceAnalyzer
    perf_analyzer = PerformanceAnalyzer(equity)
    
    total_return = perf_analyzer.calculate_returns(method='total')
    annual_return = perf_analyzer.calculate_annual_return()
    sharpe = perf_analyzer.calculate_sharpe_ratio()
    max_dd, _, _ = perf_analyzer.calculate_max_drawdown()
    win_stats = perf_analyzer.calculate_win_rate()
    
    print(f"   总收益率: {total_return*100:.2f}%")
    print(f"   年化收益率: {annual_return*100:.2f}%")
    print(f"   夏普比率: {sharpe:.2f}")
    print(f"   最大回撤: {max_dd*100:.2f}%")
    print(f"   胜率: {win_stats['win_rate']*100:.2f}%")
    
    # 使用 RiskAnalyzer
    risk_analyzer = RiskAnalyzer(returns)
    
    volatility = risk_analyzer.calculate_volatility()
    var_95 = risk_analyzer.calculate_var(confidence=0.95)
    cvar_95 = risk_analyzer.calculate_cvar(confidence=0.95)
    sortino = risk_analyzer.calculate_sortino_ratio()
    
    print(f"   年化波动率: {volatility*100:.2f}%")
    print(f"   VaR(95%): {var_95*100:.2f}%")
    print(f"   CVaR(95%): {cvar_95*100:.2f}%")
    print(f"   Sortino比率: {sortino:.2f}")
    
    return equity, returns


def visualize_results(equity, returns):
    """可视化结果"""
    print("\n📈 可视化结果...")
    
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output", "charts")
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用 ResultVisualizer
    viz = ResultVisualizer(equity, output_dir=output_dir)
    
    try:
        viz.plot_returns()
        print("   ✓ 收益率曲线")
    except Exception as e:
        print(f"   ⏭ 收益率曲线跳过: {e}")
    
    try:
        viz.plot_drawdown()
        print("   ✓ 回撤曲线")
    except Exception as e:
        print(f"   ⏭ 回撤曲线跳过: {e}")
    
    try:
        viz.plot_rolling_metrics()
        print("   ✓ 滚动指标")
    except Exception as e:
        print(f"   ⏭ 滚动指标跳过: {e}")
    
    try:
        viz.plot_monthly_returns()
        print("   ✓ 月度收益")
    except Exception as e:
        print(f"   ⏭ 月度收益跳过: {e}")


def save_results(df, tech_factors, fund_factors):
    """保存结果"""
    print("\n💾 保存结果...")
    
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output", "results")
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用 DataStorage
    storage = DataStorage(base_dir=output_dir)
    
    # 保存原始数据
    storage.save_to_csv(df, "factor_library_data.csv")
    print("   ✓ 原始数据")
    
    # 保存技术因子
    storage.save_to_csv(tech_factors, "factor_library_tech_factors.csv")
    print("   ✓ 技术因子")
    
    # 保存基本面因子
    storage.save_to_csv(fund_factors, "factor_library_fund_factors.csv")
    print("   ✓ 基本面因子")
    
    # 保存交易信号
    signals_df = df[['trade_date', 'close', 'signal', 'return', 'strategy_return', 'equity']]
    storage.save_to_csv(signals_df, "factor_library_signals.csv")
    print("   ✓ 交易信号")


def main():
    """主函数"""
    print("=" * 60)
    print("📊 因子库完整示例 - 策略层 + 回测层")
    print("=" * 60)
    
    # 1. 生成数据
    df = generate_sample_data()
    
    # 2. 计算技术因子
    tech_factors = calculate_technical_factors(df)
    
    # 3. 计算基本面因子
    fund_factors = calculate_fundamental_factors(df)
    
    # 4. 使用 FactorLibrary
    library = use_factor_library()
    
    # 5. 生成信号并回测
    result = generate_signals(df, tech_factors)
    
    # 6. 性能分析
    equity, returns = analyze_performance(result)
    
    # 7. 可视化
    visualize_results(equity, returns)
    
    # 8. 保存结果
    save_results(result, tech_factors, fund_factors)
    
    print("\n" + "=" * 60)
    print("✅ 因子库示例运行完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
