# -*- coding: utf-8 -*-
"""
example_5_full_backtest.py - 完整回测示例

直接演示回测层所有模块：
1. BacktestConfig - 配置
2. BacktestEngine - 回测引擎（简化使用）
3. PerformanceAnalyzer - 性能分析
4. RiskAnalyzer - 风险分析
5. ResultVisualizer - 可视化
6. BacktestReport - 报告生成

使用说明：
  python3 example_5_full_backtest.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============== 回测层 ==============
from backtest import (
    BacktestConfig, 
    PerformanceAnalyzer,
    RiskAnalyzer,
    ResultVisualizer,
    BacktestReport
)

# ============== 数据层 ==============
from data.data_storage import DataStorage


def generate_sample_data():
    """生成模拟数据"""
    print("📊 生成模拟数据...")
    
    # 生成252个交易日（一年）
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=252, freq='B')[:200]
    
    # 生成模拟的OHLCV数据（带趋势）
    base_price = 100
    trend = np.linspace(0, 20, len(dates))
    noise = np.random.randn(len(dates)) * 3
    prices = base_price + trend + noise
    prices = np.maximum(prices, 10)
    
    # 生成买入持有信号的收益
    daily_returns = pd.Series(prices).pct_change().fillna(0)
    
    equity = 100000 * (1 + daily_returns).cumprod()
    
    df = pd.DataFrame({
        'trade_date': [d.strftime('%Y%m%d') for d in dates],
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, len(dates)),
        'returns': daily_returns,
        'equity': equity,
    })
    
    print(f"   生成 {len(df)} 条数据")
    return df


def test_performance_analyzer(df):
    """测试性能分析模块"""
    print("\n📈 测试 PerformanceAnalyzer...")
    
    equity = df['equity']
    returns = df['returns']
    
    analyzer = PerformanceAnalyzer(equity_curve=equity)
    
    # 计算各项指标
    returns_result = analyzer.calculate_returns(method="all")
    total_return = returns_result['total']
    print(f"   ✓ 总收益率: {total_return*100:.2f}%")
    
    annual_return = analyzer.calculate_annual_return()
    print(f"   ✓ 年化收益率: {annual_return*100:.2f}%")
    
    sharpe = analyzer.calculate_sharpe_ratio()
    print(f"   ✓ 夏普比率: {sharpe:.2f}")
    
    max_dd, dd_start, dd_end = analyzer.calculate_max_drawdown()
    print(f"   ✓ 最大回撤: {max_dd*100:.2f}%")
    
    win_rate = analyzer.calculate_win_rate()
    win_rate_val = win_rate['win_rate'] if isinstance(win_rate, dict) else win_rate
    print(f"   ✓ 胜率: {win_rate_val*100:.2f}%")
    
    calmar = analyzer.calculate_calmar_ratio()
    print(f"   ✓ 卡尔玛比率: {calmar:.2f}")
    
    profit_factor = analyzer.calculate_profit_factor()
    print(f"   ✓ 盈利因子: {profit_factor:.2f}")
    
    # 完整分析
    full_analysis = analyzer.get_full_analysis()
    print(f"   ✓ 完整分析完成")
    
    return df['returns'], equity


def test_risk_analyzer(returns):
    """测试风险分析模块"""
    print("\n📉 测试 RiskAnalyzer...")
    
    analyzer = RiskAnalyzer(returns=returns)
    
    # 波动率
    volatility = analyzer.calculate_volatility()
    print(f"   ✓ 年化波动率: {volatility*100:.2f}%")
    
    # VaR / CVaR
    var_95 = analyzer.calculate_var(confidence=0.95)
    var_99 = analyzer.calculate_var(confidence=0.99)
    print(f"   ✓ VaR(95%): {var_95*100:.2f}%")
    print(f"   ✓ VaR(99%): {var_99*100:.2f}%")
    
    cvar_95 = analyzer.calculate_cvar(confidence=0.95)
    cvar_99 = analyzer.calculate_cvar(confidence=0.99)
    print(f"   ✓ CVaR(95%): {cvar_95*100:.2f}%")
    print(f"   ✓ CVaR(99%): {cvar_99*100:.2f}%")
    
    # Sortino 比率
    sortino = analyzer.calculate_sortino_ratio()
    print(f"   ✓ Sortino 比率: {sortino:.2f}")
    
    # 完整风险分析
    full_risk = analyzer.get_full_risk_analysis()
    print(f"   ✓ 完整风险分析完成")
    
    return volatility, var_95


def test_result_visualizer(returns, equity):
    """测试可视化模块"""
    print("\n📊 测试 ResultVisualizer...")
    
    output_dir = os.path.join(
        os.path.dirname(__file__), 
        "..", "output", "charts"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = ResultVisualizer(equity_curve=equity, output_dir=output_dir)
    
    # 绘制收益率曲线
    fig1 = visualizer.plot_returns()
    print(f"   ✓ 收益率曲线")
    
    # 绘制回撤曲线
    fig2 = visualizer.plot_drawdown()
    print(f"   ✓ 回撤曲线")
    
    # 绘制收益分布 (需要 scipy，跳过)
    # fig3 = visualizer.plot_factor_distribution()
    print(f"   ⏭ 收益分布 (需要scipy，跳过)")
    
    # 绘制滚动指标
    fig4 = visualizer.plot_rolling_metrics()
    print(f"   ✓ 滚动指标")
    
    # 绘制月度收益
    fig5 = visualizer.plot_monthly_returns()
    print(f"   ✓ 月度收益")
    
    return True


def test_backtest_report(df, equity, returns, volatility, var_95):
    """测试报告生成模块"""
    print("\n📄 测试 BacktestReport...")
    
    output_dir = os.path.join(
        os.path.dirname(__file__), 
        "..", "output"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备报告数据
    analyzer = PerformanceAnalyzer(equity_curve=equity)
    
    report_data = {
        'strategy': '示例策略',
        'initial_cash': 100000,
        'final_cash': equity.iloc[-1],
        'total_return': analyzer.calculate_returns(method="all")['total'],
        'annual_return': analyzer.calculate_annual_return(),
        'sharpe_ratio': analyzer.calculate_sharpe_ratio(),
        'max_drawdown': analyzer.calculate_max_drawdown()[0],
        'win_rate': analyzer.calculate_win_rate()['win_rate'],
        'volatility': volatility,
        'var_95': var_95,
        'trades': 10,
        'winning_trades': 6,
        'losing_trades': 4,
    }
    
    # 生成HTML报告
    html_path = os.path.join(output_dir, "report_test.html")
    report_obj = BacktestReport(report_data, equity_curve=equity)
    report_obj.generate_html(html_path)
    print(f"   ✓ HTML报告: {html_path}")
    
    # 生成JSON报告
    json_path = os.path.join(output_dir, "report_test.json")
    report_obj.generate_json(json_path)
    print(f"   ✓ JSON报告: {json_path}")
    
    # 生成Markdown报告
    md_path = os.path.join(output_dir, "report_test.md")
    report_obj.generate_markdown(md_path)
    print(f"   ✓ Markdown报告: {md_path}")
    
    # ========== 8. 使用 DataStorage 保存结果 ==========
    print("\n💾 使用 DataStorage 保存结果...")
    
    storage = DataStorage(base_dir=output_dir)
    
    # 保存原始数据
    storage.save_to_csv(df, "sample_data.csv")
    print("   ✓ 原始数据已保存")
    
    # 保存权益曲线
    equity_df = pd.DataFrame({'equity': equity})
    storage.save_to_csv(equity_df, "equity_curve.csv")
    print("   ✓ 权益曲线已保存")
    
    # 保存收益数据
    returns_df = pd.DataFrame({'returns': returns})
    storage.save_to_csv(returns_df, "returns.csv")
    print("   ✓ 收益数据已保存")
    
    # 保存为 Parquet 格式（压缩，需要 pyarrow）
    try:
        storage.save_to_parquet(equity_df, "equity_curve.parquet")
        print("   ✓ 权益曲线(Parquet)已保存")
    except (ImportError, IOError) as e:
        if "pyarrow" in str(e):
            print("   ⏭ 权益曲线(Parquet)跳过，需要安装 pyarrow")
        else:
            raise
    
    return report_data


def test_backtest_config():
    """测试配置模块"""
    print("\n⚙️ 测试 BacktestConfig...")
    
    config = BacktestConfig(
        initial_cash=100000,
        commission_rate=0.0003,
        stamp_tax_rate=0.001,
        slippage=0.001,
    )
    
    print(f"   ✓ 初始资金: {config.initial_cash}")
    print(f"   ✓ 佣金: {config.commission_rate}")
    print(f"   ✓ 印花税: {config.stamp_tax_rate}")
    print(f"   ✓ 滑点: {config.slippage}")
    
    return config


def main():
    """主函数"""
    print("=" * 60)
    print("📊 完整回测示例 - 覆盖回测层所有模块")
    print("=" * 60)
    
    # 1. 生成数据
    df = generate_sample_data()
    
    # 2. 测试配置模块
    test_backtest_config()
    
    # 3. 测试性能分析模块
    returns, equity = test_performance_analyzer(df)
    
    # 4. 测试风险分析模块
    volatility, var_95 = test_risk_analyzer(returns)
    
    # 5. 测试可视化模块
    test_result_visualizer(returns, equity)
    
    # 6. 测试报告生成模块
    test_backtest_report(df, equity, returns, volatility, var_95)
    
    print("\n" + "=" * 60)
    print("✅ 完整回测示例运行完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
