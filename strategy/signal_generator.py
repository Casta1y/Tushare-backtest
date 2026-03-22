# -*- coding: utf-8 -*-
"""
信号生成器模块 (Signal Generator)

根据技术指标和基本面因子生成交易信号。

主要功能：
- 基于单因子的买入/卖出信号
- 多因子组合信号
- 信号强度计算
- 信号过滤和优化
"""

import logging
from typing import Optional, Dict, Any, List, Union, Tuple
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """信号类型枚举"""
    BUY = 1      # 买入信号
    SELL = -1    # 卖出信号
    HOLD = 0     # 持有/观望
    STRONG_BUY = 2   # 强买入
    STRONG_SELL = -2 # 强卖出


class SignalGenerator:
    """
    信号生成器
    
    根据因子值和阈值生成交易信号。
    
    Parameters
    ----------
    method : str, optional
        信号生成方法: "threshold", "crossover", "percentile", 默认"threshold"
    
    Examples
    --------
    >>> generator = SignalGenerator(method="threshold")
    >>> signals = generator.generate(df, factor_name="RSI", 
    ...                              buy_threshold=30, sell_threshold=70)
    """

    def __init__(self, method: str = "threshold"):
        """
        初始化信号生成器。
        
        Parameters
        ----------
        method : str
            信号生成方法
        """
        self.method = method
        self.signals_cache = {}

    def generate(
        self,
        data: pd.DataFrame,
        factor_name: str,
        buy_threshold: Optional[float] = None,
        sell_threshold: Optional[float] = None,
        buy_condition: Optional[callable] = None,
        sell_condition: Optional[callable] = None,
        signal_type: str = "binary",
    ) -> pd.DataFrame:
        """
        生成交易信号。
        
        Parameters
        ----------
        data : pd.DataFrame
            包含因子值的数据
        factor_name : str
            因子名称
        buy_threshold : float, optional
            买入阈值
        sell_threshold : float, optional
            卖出阈值
        buy_condition : callable, optional
            自定义买入条件函数
        sell_condition : callable, optional
            自定义卖出条件函数
        signal_type : str
            信号类型: "binary", "strength", "direction"
        
        Returns
        -------
        pd.DataFrame
            信号DataFrame，包含以下列:
            - signal: 信号值 (1=买入, -1=卖出, 0=持有)
            - signal_strength: 信号强度 (0-1)
            - signal_type: 信号类型 (BUY/SELL/HOLD)
        """
        if factor_name not in data.columns:
            raise ValueError(f"Factor '{factor_name}' not found in data.")
        
        factor_values = data[factor_name]
        
        if self.method == "threshold":
            signals = self._generate_threshold(
                factor_values, buy_threshold, sell_threshold, signal_type
            )
        elif self.method == "crossover":
            signals = self._generate_crossover(
                factor_values, buy_threshold, sell_threshold, signal_type
            )
        elif self.method == "percentile":
            signals = self._generate_percentile(
                factor_values, buy_threshold, sell_threshold, signal_type
            )
        elif self.method == "custom":
            signals = self._generate_custom(
                factor_values, buy_condition, sell_condition, signal_type
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return signals

    def _generate_threshold(
        self,
        factor_values: pd.Series,
        buy_threshold: Optional[float],
        sell_threshold: Optional[float],
        signal_type: str,
    ) -> pd.DataFrame:
        """基于阈值的信号生成"""
        n = len(factor_values)
        
        if signal_type == "binary":
            signal = np.zeros(n)
            signal[factor_values <= buy_threshold] = SignalType.BUY.value
            signal[factor_values >= sell_threshold] = SignalType.SELL.value
            
        elif signal_type == "strength":
            # 信号强度基于距离阈值的远近
            signal = np.zeros(n)
            signal_strength = np.zeros(n)
            
            if buy_threshold is not None:
                # 越低越买入
                distance = (buy_threshold - factor_values) / buy_threshold
                signal_strength = np.clip(distance, 0, 1)
                signal[factor_values <= buy_threshold] = SignalType.BUY.value
            
            if sell_threshold is not None:
                distance = (factor_values - sell_threshold) / sell_threshold
                sell_strength = np.clip(distance, 0, 1)
                signal_strength = np.maximum(signal_strength, sell_strength)
                signal[factor_values >= sell_threshold] = SignalType.SELL.value
            
            return pd.DataFrame({
                "signal": signal,
                "signal_strength": signal_strength,
            }, index=factor_values.index)
        
        return pd.DataFrame({
            "signal": signal,
            "signal_strength": np.abs(signal),
        }, index=factor_values.index)

    def _generate_crossover(
        self,
        factor_values: pd.Series,
        buy_threshold: Optional[float],
        sell_threshold: Optional[float],
        signal_type: str,
    ) -> pd.DataFrame:
        """基于交叉的信号生成"""
        n = len(factor_values)
        signal = np.zeros(n)
        
        if buy_threshold is not None:
            # 金叉：因子从下往上穿过阈值
            below = factor_values <= buy_threshold
            above = factor_values > buy_threshold
            cross_up = (below.shift(1).fillna(True)) & above
            signal[cross_up] = SignalType.BUY.value
        
        if sell_threshold is not None:
            # 死叉：因子从上往下穿过阈值
            above = factor_values >= sell_threshold
            below = factor_values < sell_threshold
            cross_down = (above.shift(1).fillna(True)) & below
            signal[cross_down] = SignalType.SELL.value
        
        return pd.DataFrame({
            "signal": signal,
            "signal_strength": np.abs(signal),
        }, index=factor_values.index)

    def _generate_percentile(
        self,
        factor_values: pd.Series,
        buy_threshold: Optional[float],
        sell_threshold: Optional[float],
        signal_type: str,
    ) -> pd.Series:
        """基于百分位的信号生成"""
        n = len(factor_values)
        signal = np.zeros(n)
        
        if buy_threshold is not None:
            # 买入信号：因子值在历史百分位的底部
            lower_percentile = np.nanpercentile(factor_values, buy_threshold)
            signal[factor_values <= lower_percentile] = SignalType.BUY.value
        
        if sell_threshold is not None:
            # 卖出信号：因子值在历史百分位的顶部
            upper_percentile = np.nanpercentile(factor_values, sell_threshold)
            signal[factor_values >= upper_percentile] = SignalType.SELL.value
        
        return pd.DataFrame({
            "signal": signal,
            "signal_strength": np.abs(signal),
        }, index=factor_values.index)

    def _generate_custom(
        self,
        factor_values: pd.Series,
        buy_condition: Optional[callable],
        sell_condition: Optional[callable],
        signal_type: str,
    ) -> pd.DataFrame:
        """基于自定义条件的信号生成"""
        n = len(factor_values)
        signal = np.zeros(n)
        
        if buy_condition is not None:
            signal[buy_condition(factor_values)] = SignalType.BUY.value
        
        if sell_condition is not None:
            signal[sell_condition(factor_values)] = SignalType.SELL.value
        
        return pd.DataFrame({
            "signal": signal,
            "signal_strength": np.abs(signal),
        }, index=factor_values.index)


class MultiFactorSignalGenerator:
    """
    多因子信号生成器
    
    组合多个因子的信号，生成综合信号。
    
    Parameters
    ----------
    weights : dict, optional
        各因子权重字典
    combination_method : str, optional
        组合方法: "weighted_sum", "majority", "and", "or"
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        combination_method: str = "weighted_sum",
    ):
        self.weights = weights or {}
        self.combination_method = combination_method
        self.signal_generators = {}

    def add_factor_signal(
        self,
        factor_name: str,
        generator: SignalGenerator,
        buy_threshold: Optional[float] = None,
        sell_threshold: Optional[float] = None,
        weight: Optional[float] = None,
    ) -> None:
        """
        添加一个因子的信号配置。
        
        Parameters
        ----------
        factor_name : str
            因子名称
        generator : SignalGenerator
            信号生成器实例
        buy_threshold : float, optional
            买入阈值
        sell_threshold : float, optional
            卖出阈值
        weight : float, optional
            因子权重
        """
        self.signal_generators[factor_name] = {
            "generator": generator,
            "buy_threshold": buy_threshold,
            "sell_threshold": sell_threshold,
            "weight": weight or 1.0,
        }
        
        if weight is not None:
            self.weights[factor_name] = weight

    def generate(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        生成多因子综合信号。
        
        Parameters
        ----------
        data : pd.DataFrame
            包含所有因子值的数据
        
        Returns
        -------
        pd.DataFrame
            综合信号DataFrame
        """
        all_signals = []
        
        for factor_name, config in self.signal_generators.items():
            if factor_name not in data.columns:
                logger.warning(f"Factor '{factor_name}' not in data, skipping.")
                continue
            
            generator = config["generator"]
            signals = generator.generate(
                data,
                factor_name,
                buy_threshold=config["buy_threshold"],
                sell_threshold=config["sell_threshold"],
            )
            signals.columns = [f"{factor_name}_{col}" for col in signals.columns]
            all_signals.append(signals)
        
        if not all_signals:
            raise ValueError("No valid signals generated.")
        
        # 合并所有信号
        combined = pd.concat(all_signals, axis=1)
        
        # 组合信号
        if self.combination_method == "weighted_sum":
            result = self._weighted_sum(combined)
        elif self.combination_method == "majority":
            result = self._majority_vote(combined)
        elif self.combination_method == "and":
            result = self._and_combination(combined)
        elif self.combination_method == "or":
            result = self._or_combination(combined)
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")
        
        return result

    def _weighted_sum(self, combined: pd.DataFrame) -> pd.DataFrame:
        """加权求和"""
        signal_cols = [col for col in combined.columns if col.endswith("_signal")]
        
        weighted_sum = pd.Series(0.0, index=combined.index)
        total_weight = 0
        
        for col in signal_cols:
            factor_name = col.replace("_signal", "")
            weight = self.weights.get(factor_name, 1.0)
            weighted_sum += combined[col] * weight
            total_weight += weight
        
        if total_weight > 0:
            weighted_sum /= total_weight
        
        # 归一化到 [-1, 1]
        signal = np.sign(weighted_sum)
        strength = np.abs(weighted_sum)
        
        return pd.DataFrame({
            "signal": signal,
            "signal_strength": strength,
        }, index=combined.index)

    def _majority_vote(self, combined: pd.DataFrame) -> pd.DataFrame:
        """多数投票"""
        signal_cols = [col for col in combined.columns if col.endswith("_signal")]
        
        votes = combined[signal_cols].sum(axis=1)
        
        # 多数买入
        signal = np.where(votes > 0, SignalType.BUY.value, 
                np.where(votes < 0, SignalType.SELL.value, SignalType.HOLD.value))
        
        strength = np.abs(votes) / len(signal_cols)
        
        return pd.DataFrame({
            "signal": signal,
            "signal_strength": strength,
        }, index=combined.index)

    def _and_combination(self, combined: pd.DataFrame) -> pd.DataFrame:
        """AND组合：所有因子都发出信号时才执行"""
        signal_cols = [col for col in combined.columns if col.endswith("_signal")]
        
        min_signal = combined[signal_cols].min(axis=1)
        
        return pd.DataFrame({
            "signal": np.sign(min_signal),
            "signal_strength": np.abs(min_signal),
        }, index=combined.index)

    def _or_combination(self, combined: pd.DataFrame) -> pd.DataFrame:
        """OR组合：任一因子发出信号时就执行"""
        signal_cols = [col for col in combined.columns if col.endswith("_signal")]
        
        max_signal = combined[signal_cols].max(axis=1).abs()
        
        signal = combined[signal_cols].sum(axis=1)
        
        return pd.DataFrame({
            "signal": np.sign(signal),
            "signal_strength": max_signal,
        }, index=combined.index)


# ============================================================
# 预设信号生成器
# ============================================================

class RSISignalGenerator(SignalGenerator):
    """
    RSI信号生成器
    
    预设RSI超买超卖信号:
    - RSI < 30: 买入信号
    - RSI > 70: 卖出信号
    """
    
    def __init__(self, period: int = 14):
        super().__init__(method="threshold")
        self.period = period
        self.buy_threshold = 30
        self.sell_threshold = 70

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成RSI信号"""
        if "RSI" not in data.columns:
            raise ValueError("RSI factor not found.")
        
        return super().generate(
            data, "RSI", 
            buy_threshold=self.buy_threshold, 
            sell_threshold=self.sell_threshold
        )


class MACDSignalGenerator(SignalGenerator):
    """
    MACD信号生成器
    
    预设MACD金叉死叉信号:
    - DIF上穿DEA: 买入信号
    - DIF下穿DEA: 卖出信号
    """
    
    def __init__(self):
        super().__init__(method="crossover")

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成MACD信号"""
        if "MACD_DIFF" not in data.columns or "MACD_DEA" not in data.columns:
            raise ValueError("MACD factors not found.")
        
        # DIF与DEA交叉
        signal = np.zeros(len(data))
        diff = data["MACD_DIFF"].values
        dea = data["MACD_DEA"].values
        
        # 金叉
        for i in range(1, len(data)):
            if diff[i-1] <= dea[i-1] and diff[i] > dea[i]:
                signal[i] = SignalType.BUY.value
            elif diff[i-1] >= dea[i-1] and diff[i] < dea[i]:
                signal[i] = SignalType.SELL.value
        
        return pd.DataFrame({
            "signal": signal,
            "signal_strength": np.abs(signal),
        }, index=data.index)


class KDJSignalGenerator(SignalGenerator):
    """
    KDJ信号生成器
    
    预设KDJ超买超卖信号:
    - KDJ < 20: 买入信号
    - KDJ > 80: 卖出信号
    - J < 0: 超卖买入
    - J > 100: 超买卖出
    """
    
    def __init__(self):
        super().__init__(method="threshold")
        self.buy_threshold = 20
        self.sell_threshold = 80

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成KDJ信号"""
        if "KDJ" not in data.columns:
            raise ValueError("KDJ factor not found.")
        
        return super().generate(
            data, "KDJ",
            buy_threshold=self.buy_threshold,
            sell_threshold=self.sell_threshold
        )


class BollingerBandSignalGenerator(SignalGenerator):
    """
    布林带信号生成器
    
    - 价格突破下轨: 买入信号
    - 价格突破上轨: 卖出信号
    """
    
    def __init__(self):
        super().__init__(method="custom")

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成布林带信号"""
        if "BBANDS_POSITION" not in data.columns:
            raise ValueError("BBANDS_POSITION factor not found.")
        
        position = data["BBANDS_POSITION"]
        
        signal = np.zeros(len(data))
        signal[position < 0] = SignalType.BUY.value
        signal[position > 1] = SignalType.SELL.value
        
        return pd.DataFrame({
            "signal": signal,
            "signal_strength": np.abs(signal),
        }, index=data.index)


class MovingAverageSignalGenerator(SignalGenerator):
    """
    移动平均线信号生成器
    
    - 价格上穿MA: 买入信号
    - 价格下穿MA: 卖出信号
    """
    
    def __init__(self, period: int = 20):
        super().__init__(method="crossover")
        self.period = period

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成均线信号"""
        ma_col = f"MA_{self.period}"
        if ma_col not in data.columns:
            # 尝试通用MA
            if "MA" not in data.columns:
                raise ValueError("MA factor not found.")
            ma_col = "MA"
        
        signal = np.zeros(len(data))
        
        for i in range(1, len(data)):
            if pd.notna(data[ma_col].iloc[i-1]) and pd.notna(data[ma_col].iloc[i]):
                # 价格上穿均线
                if data["close"].iloc[i-1] <= data[ma_col].iloc[i-1] and \
                   data["close"].iloc[i] > data[ma_col].iloc[i]:
                    signal[i] = SignalType.BUY.value
                # 价格下穿均线
                elif data["close"].iloc[i-1] >= data[ma_col].iloc[i-1] and \
                     data["close"].iloc[i] < data[ma_col].iloc[i]:
                    signal[i] = SignalType.SELL.value
        
        return pd.DataFrame({
            "signal": signal,
            "signal_strength": np.abs(signal),
        }, index=data.index)


def create_default_signal_config() -> Dict[str, Dict[str, Any]]:
    """
    创建默认信号配置
    
    Returns
    -------
    dict
        默认信号配置字典
    """
    return {
        "RSI": {
            "generator": RSISignalGenerator(),
            "buy_threshold": 30,
            "sell_threshold": 70,
            "weight": 1.0,
        },
        "MACD": {
            "generator": MACDSignalGenerator(),
            "weight": 1.5,
        },
        "KDJ": {
            "generator": KDJSignalGenerator(),
            "buy_threshold": 20,
            "sell_threshold": 80,
            "weight": 1.0,
        },
        "BBANDS": {
            "generator": BollingerBandSignalGenerator(),
            "weight": 1.0,
        },
    }
