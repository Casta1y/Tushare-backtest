# -*- coding: utf-8 -*-
"""
策略组合器模块 (Strategy Composer)

将多个策略组合成复合策略，支持：
- 策略权重配置
- 策略排序和筛选
- 动态权重调整
- 策略池管理
"""

import logging
from typing import Optional, Dict, Any, List, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CombineMethod(Enum):
    """策略组合方法"""
    EQUAL_WEIGHT = "equal_weight"           # 等权重
    WEIGHTED_SUM = "weighted_sum"           # 加权求和
    RANK_AVERAGE = "rank_average"           # 排序平均
    MAX_CONCENTRATION = "max"               # 取最大
    MIN_CONCENTRATION = "min"               # 取最小
    MULTIPLICATIVE = "multiplicative"       # 乘法组合


class StrategyType(Enum):
    """策略类型"""
    MOMENTUM = "momentum"            # 动量策略
    REVERSAL = "reversal"           # 反转策略
    TREND = "trend"                 # 趋势策略
    MEAN_REVERSION = "mean_reversion"  # 均值回归
    BREAKOUT = "breakout"           # 突破策略
    VOLATILITY = "volatility"       # 波动率策略
    FUNDAMENTAL = "fundamental"     # 基本面策略
    QUANTITATIVE = "quantitative"   # 量化策略


@dataclass
class StrategyConfig:
    """策略配置数据类"""
    name: str
    strategy_type: Union[str, StrategyType]
    weight: float = 1.0
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    performance: Optional[Dict[str, float]] = None


@dataclass
class StrategyResult:
    """策略执行结果"""
    name: str
    signal: pd.Series
    weight: float
    performance: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class StrategyComposer:
    """
    策略组合器
    
    将多个策略组合成复合策略。
    
    Parameters
    ----------
    combine_method : str, optional
        组合方法，默认"equal_weight"
    
    Examples
    --------
    >>> composer = StrategyComposer(combine_method="weighted_sum")
    >>> composer.add_strategy("MA5", signal_ma5, weight=1.0)
    >>> composer.add_strategy("RSI", signal_rsi, weight=1.5)
    >>> combined = composer.combine()
    """

    def __init__(self, combine_method: str = "equal_weight"):
        """
        初始化策略组合器。
        
        Parameters
        ----------
        combine_method : str
            策略组合方法
        """
        self.combine_method = CombineMethod(combine_method)
        self.strategies: Dict[str, StrategyConfig] = {}
        self.strategy_signals: Dict[str, pd.Series] = {}
        self.strategy_performance: Dict[str, Dict[str, float]] = {}
        
        # 权重调整参数
        self.weight_adjustment_method: Optional[str] = None
        self.performance_window: int = 20

    def add_strategy(
        self,
        name: str,
        signal: Union[pd.Series, pd.DataFrame],
        weight: float = 1.0,
        strategy_type: Optional[Union[str, StrategyType]] = None,
        params: Optional[Dict[str, Any]] = None,
        description: str = "",
        enabled: bool = True,
    ) -> None:
        """
        添加策略。
        
        Parameters
        ----------
        name : str
            策略名称
        signal : pd.Series or pd.DataFrame
            策略信号
        weight : float, optional
            策略权重
        strategy_type : str or StrategyType, optional
            策略类型
        params : dict, optional
            策略参数字典
        description : str, optional
            策略描述
        enabled : bool, optional
            是否启用
        
        Raises
        ------
        ValueError
            信号长度不匹配时抛出
        """
        if isinstance(signal, pd.DataFrame):
            # 取signal列或第一列
            signal = signal["signal"] if "signal" in signal.columns else signal.iloc[:, 0]
        
        self.strategies[name] = StrategyConfig(
            name=name,
            strategy_type=strategy_type or StrategyType.QUANTITATIVE,
            weight=weight,
            enabled=enabled,
            params=params or {},
            description=description,
        )
        self.strategy_signals[name] = signal

    def remove_strategy(self, name: str) -> None:
        """移除策略"""
        if name in self.strategies:
            del self.strategies[name]
        if name in self.strategy_signals:
            del self.strategy_signals[name]
        if name in self.strategy_performance:
            del self.strategy_performance[name]

    def set_weight(self, name: str, weight: float) -> None:
        """设置策略权重"""
        if name in self.strategies:
            self.strategies[name].weight = weight
        else:
            raise ValueError(f"Strategy '{name}' not found.")

    def enable_strategy(self, name: str) -> None:
        """启用策略"""
        if name in self.strategies:
            self.strategies[name].enabled = True

    def disable_strategy(self, name: str) -> None:
        """禁用策略"""
        if name in self.strategies:
            self.strategies[name].enabled = False

    def get_enabled_strategies(self) -> List[str]:
        """获取启用的策略列表"""
        return [name for name, cfg in self.strategies.items() if cfg.enabled]

    def combine(
        self,
        method: Optional[str] = None,
        normalize: bool = True,
    ) -> pd.DataFrame:
        """
        组合策略信号。
        
        Parameters
        ----------
        method : str, optional
            组合方法，默认使用初始化时的方法
        normalize : bool, optional
            是否归一化信号
        
        Returns
        -------
        pd.DataFrame
            组合后的信号，包含列:
            - signal: 综合信号值
            - signal_strength: 信号强度
            - strategy_count: 参与的策略数量
        """
        enabled = self.get_enabled_strategies()
        
        if not enabled:
            warnings.warn("No enabled strategies to combine.")
            return pd.DataFrame({
                "signal": pd.Series(dtype=float),
                "signal_strength": pd.Series(dtype=float),
                "strategy_count": pd.Series(dtype=int),
            })
        
        # 获取所有信号
        signals = [self.strategy_signals[name] for name in enabled]
        
        # 确保索引对齐
        ref_index = signals[0].index
        for s in signals[1:]:
            if not s.index.equals(ref_index):
                s = s.reindex(ref_index)
        
        combine_method = CombineMethod(method) if method else self.combine_method
        
        if combine_method == CombineMethod.EQUAL_WEIGHT:
            result = self._combine_equal_weight(signals)
        elif combine_method == CombineMethod.WEIGHTED_SUM:
            result = self._combine_weighted_sum(enabled, signals)
        elif combine_method == CombineMethod.RANK_AVERAGE:
            result = self._combine_rank_average(signals)
        elif combine_method == CombineMethod.MAX_CONCENTRATION:
            result = self._combine_max(signals)
        elif combine_method == CombineMethod.MIN_CONCENTRATION:
            result = self._combine_min(signals)
        elif combine_method == CombineMethod.MULTIPLICATIVE:
            result = self._combine_multiplicative(signals)
        else:
            raise ValueError(f"Unknown combine method: {method}")
        
        # 统计参与策略数量
        signal_array = np.array([s.values for s in signals])
        strategy_count = np.sum(signal_array != 0, axis=0)
        
        result["strategy_count"] = strategy_count
        
        if normalize:
            result["signal"] = np.sign(result["signal"])
            result["signal_strength"] = np.clip(result["signal_strength"], 0, 1)
        
        return result

    def _combine_equal_weight(self, signals: List[pd.Series]) -> pd.DataFrame:
        """等权重组合"""
        combined = np.mean([s.values for s in signals], axis=0)
        
        return pd.DataFrame({
            "signal": combined,
            "signal_strength": np.abs(combined),
        }, index=signals[0].index)

    def _combine_weighted_sum(
        self, 
        enabled: List[str], 
        signals: List[pd.Series]
    ) -> pd.DataFrame:
        """加权求和组合"""
        weights = np.array([self.strategies[name].weight for name in enabled])
        weights = weights / weights.sum()  # 归一化权重
        
        weighted = np.sum([s.values * w for s, w in zip(signals, weights)], axis=0)
        
        return pd.DataFrame({
            "signal": weighted,
            "signal_strength": np.abs(weighted),
        }, index=signals[0].index)

    def _combine_rank_average(self, signals: List[pd.Series]) -> pd.DataFrame:
        """排序平均组合"""
        ranked = []
        for s in signals:
            # 转换为排序
            rank = pd.Series(s.values).rank(pct=True)
            ranked.append(rank.values)
        
        avg_rank = np.mean(ranked, axis=0)
        # 映射回 [-1, 1]
        combined = 2 * avg_rank - 1
        
        return pd.DataFrame({
            "signal": combined,
            "signal_strength": np.abs(combined),
        }, index=signals[0].index)

    def _combine_max(self, signals: List[pd.Series]) -> pd.DataFrame:
        """取最大信号"""
        combined = np.max([s.values for s in signals], axis=0)
        
        return pd.DataFrame({
            "signal": combined,
            "signal_strength": np.abs(combined),
        }, index=signals[0].index)

    def _combine_min(self, signals: List[pd.Series]) -> pd.DataFrame:
        """取最小信号"""
        combined = np.min([s.values for s in signals], axis=0)
        
        return pd.DataFrame({
            "signal": combined,
            "signal_strength": np.abs(combined),
        }, index=signals[0].index)

    def _combine_multiplicative(self, signals: List[pd.Series]) -> pd.DataFrame:
        """乘法组合"""
        # 避免0值
        combined = np.prod([np.sign(s.values) * np.maximum(np.abs(s.values), 0.01) 
                          for s in signals], axis=0)
        
        return pd.DataFrame({
            "signal": np.sign(combined),
            "signal_strength": np.minimum(np.abs(combined), 1.0),
        }, index=signals[0].index)

    def update_weights_by_performance(
        self,
        returns: pd.Series,
        method: str = "inverse_volatility",
        window: int = 20,
    ) -> None:
        """
        根据历史表现更新策略权重。
        
        Parameters
        ----------
        returns : pd.Series
            资产收益序列
        method : str
            权重调整方法
        window : int
            计算窗口
        """
        self.weight_adjustment_method = method
        self.performance_window = window
        
        if method == "inverse_volatility":
            self._update_by_inverse_volatility(returns, window)
        elif method == "momentum":
            self._update_by_momentum(returns, window)
        elif method == "sharpe":
            self._update_by_sharpe(returns, window)
        else:
            raise ValueError(f"Unknown weight adjustment method: {method}")

    def _update_by_inverse_volatility(
        self, 
        returns: pd.Series, 
        window: int
    ) -> None:
        """基于逆波动率更新权重"""
        enabled = self.get_enabled_strategies()
        
        for name in enabled:
            signal = self.strategy_signals[name]
            # 计算策略收益
            strategy_returns = returns * signal
            volatility = strategy_returns.rolling(window).std().iloc[-1]
            
            if volatility > 0:
                self.strategies[name].weight = 1 / volatility

    def _update_by_momentum(
        self, 
        returns: pd.Series, 
        window: int
    ) -> None:
        """基于动量更新权重"""
        enabled = self.get_enabled_strategies()
        
        for name in enabled:
            signal = self.strategy_signals[name]
            strategy_returns = returns * signal
            momentum = strategy_returns.rolling(window).sum().iloc[-1]
            self.strategies[name].weight = max(0, momentum)

    def _update_by_sharpe(
        self, 
        returns: pd.Series, 
        window: int
    ) -> None:
        """基于夏普比率更新权重"""
        enabled = self.get_enabled_strategies()
        
        for name in enabled:
            signal = self.strategy_signals[name]
            strategy_returns = returns * signal
            mean = strategy_returns.rolling(window).mean().iloc[-1]
            std = strategy_returns.rolling(window).std().iloc[-1]
            
            if std > 0:
                self.strategies[name].weight = mean / std

    def rank_strategies(
        self,
        returns: pd.Series,
        metric: str = "return",
        window: int = 20,
    ) -> pd.DataFrame:
        """
        对策略进行排名。
        
        Parameters
        ----------
        returns : pd.Series
            资产收益序列
        metric : str
            评估指标: "return", "sharpe", "max_drawdown"
        window : int
            评估窗口
        
        Returns
        -------
        pd.DataFrame
            策略排名表
        """
        results = []
        
        for name, config in self.strategies.items():
            signal = self.strategy_signals[name]
            strategy_returns = returns * signal
            
            if metric == "return":
                perf = strategy_returns.rolling(window).sum().iloc[-1]
            elif metric == "sharpe":
                mean = strategy_returns.rolling(window).mean().iloc[-1]
                std = strategy_returns.rolling(window).std().iloc[-1]
                perf = mean / std if std > 0 else 0
            elif metric == "max_drawdown":
                cumret = (1 + strategy_returns).cumprod()
                rolling_max = cumret.rolling(window, min_periods=1).max()
                drawdown = (cumret - rolling_max) / rolling_max
                perf = drawdown.min()
            else:
                perf = 0
            
            results.append({
                "name": name,
                "type": config.strategy_type.value if isinstance(config.strategy_type, StrategyType) else config.strategy_type,
                "weight": config.weight,
                "performance": perf,
                "enabled": config.enabled,
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values("performance", ascending=False)
        
        return df

    def select_top_strategies(
        self,
        returns: pd.Series,
        n: int = 3,
        metric: str = "return",
        window: int = 20,
    ) -> List[str]:
        """
        选择表现最好的N个策略。
        
        Parameters
        ----------
        returns : pd.Series
            资产收益序列
        n : int
            选择数量
        metric : str
            评估指标
        window : int
            评估窗口
        
        Returns
        -------
        list
            选中的策略名称列表
        """
        ranking = self.rank_strategies(returns, metric, window)
        ranking = ranking[ranking["enabled"]].head(n)
        
        # 禁用未选中的策略
        selected = ranking["name"].tolist()
        for name in self.strategies:
            if name in selected:
                self.enable_strategy(name)
            else:
                self.disable_strategy(name)
        
        return selected

    def get_composition_report(self) -> Dict[str, Any]:
        """
        获取策略组合报告。
        
        Returns
        -------
        dict
            组合报告
        """
        enabled = self.get_enabled_strategies()
        
        report = {
            "total_strategies": len(self.strategies),
            "enabled_strategies": len(enabled),
            "combine_method": self.combine_method.value,
            "strategies": [],
            "total_weight": 0,
        }
        
        for name in enabled:
            cfg = self.strategies[name]
            report["strategies"].append({
                "name": name,
                "type": cfg.strategy_type.value if isinstance(cfg.strategy_type, StrategyType) else cfg.strategy_type,
                "weight": cfg.weight,
                "description": cfg.description,
            })
            report["total_weight"] += cfg.weight
        
        return report

    def __repr__(self) -> str:
        enabled = len(self.get_enabled_strategies())
        return f"StrategyComposer(strategies={len(self.strategies)}, enabled={enabled})"


class StrategyPool:
    """
    策略池管理器
    
    管理多个策略组合器，支持策略池的回测和选择。
    
    Parameters
    ----------
    name : str
        策略池名称
    """
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.composers: Dict[str, StrategyComposer] = {}
        self.pool_performance: Dict[str, Dict[str, float]] = {}
    
    def add_composer(
        self,
        composer: StrategyComposer,
        name: str = None,
    ) -> None:
        """添加策略组合器"""
        composer_name = name or f"composer_{len(self.composers)}"
        self.composers[composer_name] = composer
    
    def remove_composer(self, name: str) -> None:
        """移除策略组合器"""
        if name in self.composers:
            del self.composers[name]
        if name in self.pool_performance:
            del self.pool_performance[name]
    
    def evaluate_pool(
        self,
        returns: pd.Series,
        metric: str = "return",
    ) -> pd.DataFrame:
        """
        评估策略池中所有组合的表现。
        
        Parameters
        ----------
        returns : pd.Series
            资产收益序列
        metric : str
            评估指标
        
        Returns
        -------
        pd.DataFrame
            评估结果
        """
        results = []
        
        for name, composer in self.composers.items():
            combined = composer.combine()
            strategy_returns = returns * combined["signal"]
            
            if metric == "return":
                perf = strategy_returns.sum()
            elif metric == "sharpe":
                mean = strategy_returns.mean()
                std = strategy_returns.std()
                perf = mean / std if std > 0 else 0
            elif metric == "max_drawdown":
                cumret = (1 + strategy_returns).cumprod()
                rolling_max = cumret.expanding().max()
                drawdown = (cumret - rolling_max) / rolling_max
                perf = drawdown.min()
            else:
                perf = 0
            
            results.append({
                "name": name,
                "performance": perf,
                "strategies": len(composer.get_enabled_strategies()),
            })
            
            self.pool_performance[name] = {"performance": perf}
        
        return pd.DataFrame(results).sort_values("performance", ascending=False)
    
    def get_best_composer(
        self,
        returns: pd.Series,
        metric: str = "return",
    ) -> tuple:
        """
        获取表现最好的组合器。
        
        Returns
        -------
        tuple
            (名称, 组合器)
        """
        evaluation = self.evaluate_pool(returns, metric)
        
        if len(evaluation) == 0:
            return None, None
        
        best_name = evaluation.iloc[0]["name"]
        return best_name, self.composers[best_name]


def create_momentum_composer() -> StrategyComposer:
    """
    创建动量策略组合器
    
    Returns
    -------
    StrategyComposer
        动量策略组合器
    """
    composer = StrategyComposer(combine_method="weighted_sum")
    return composer


def create_mean_reversion_composer() -> StrategyComposer:
    """
    创建均值回归策略组合器
    
    Returns
    -------
    StrategyComposer
        均值回归策略组合器
    """
    composer = StrategyComposer(combine_method="equal_weight")
    return composer
