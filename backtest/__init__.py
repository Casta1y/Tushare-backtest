# -*- coding: utf-8 -*-
"""
回测层模块 (Backtest Layer)

提供完整的回测框架，包括回测引擎、策略适配、性能分析、风险分析、可视化和报告生成。

主要模块
--------
backtest_config       : 配置管理
backtest_engine       : 回测引擎
backtest_strategy     : Backtrader 策略适配
performance_analyzer  : 性能分析（收益率、夏普比率、最大回撤等）
risk_analyzer        : 风险分析（波动率、Beta、VaR等）
result_visualizer    : 结果可视化
backtest_report      : 报告生成

使用示例
--------
>>> from backtest import BacktestConfig, BacktestEngine
>>> config = BacktestConfig.from_yaml("config.yaml")
>>> engine = BacktestEngine(config)
>>> results = engine.run(strategy, data)
"""

from .backtest_config import Config, BacktestConfig
from .backtest_engine import BacktestEngine
from .backtest_strategy import (
    BaseStrategy, 
    MAStrategy, 
    RSIStrategy, 
    DualSignalStrategy,
    BreakoutStrategy,
    MeanReversionStrategy,
    MultiSignalStrategy,
    FactorStrategy,
)
from .performance_analyzer import PerformanceAnalyzer
from .risk_analyzer import RiskAnalyzer
from .result_visualizer import ResultVisualizer
from .backtest_report import BacktestReport

__all__ = [
    # 配置
    "Config",
    "BacktestConfig",
    # 引擎
    "BacktestEngine",
    # 策略
    "BaseStrategy",
    "MAStrategy",
    "RSIStrategy",
    "DualSignalStrategy",
    "BreakoutStrategy",
    "MeanReversionStrategy",
    "MultiSignalStrategy",
    "FactorStrategy",
    # 分析
    "PerformanceAnalyzer",
    "RiskAnalyzer",
    # 可视化
    "ResultVisualizer",
    # 报告
    "BacktestReport",
]
