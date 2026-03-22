# -*- coding: utf-8 -*-
"""
策略层模块 (Strategy Layer)

提供因子计算、信号生成、策略组合等量化策略核心功能。

主要模块
--------
factor_base        : 因子基类
technical_factors  : 技术因子（MA、MACD、RSI、KDJ 等 46 个）
fundamental_factors: 基本面因子（PE、PB、ROE、ROA 等 47 个）
signal_generator   : 交易信号生成器
strategy_composer  : 多策略组合器
factor_library     : 因子注册与管理

使用示例
--------
>>> from strategy.factor_base import FactorBase
>>> from strategy.technical_factors import MAFactor, EMAFactor, RSIFactor
>>> from strategy.signal_generator import SignalGenerator

>>> # 计算因子
>>> ma = MAFactor({'period': 5})
>>> result = ma.calculate(df)

>>> # 生成信号
>>> generator = SignalGenerator()
>>> signals = generator.generate(df, {"buy_threshold": 0.5})
"""

__version__ = "1.0.0"

from .factor_base import FactorBase
from .signal_generator import SignalGenerator, MultiFactorSignalGenerator
from .strategy_composer import StrategyComposer, StrategyPool
from .factor_library import FactorLibrary, FactorCalculator

__all__ = [
    # 因子基类
    "FactorBase",
    # 信号生成
    "SignalGenerator",
    "MultiFactorSignalGenerator",
    # 策略组合
    "StrategyComposer",
    "StrategyPool",
    # 因子库
    "FactorLibrary",
    "FactorCalculator",
]
