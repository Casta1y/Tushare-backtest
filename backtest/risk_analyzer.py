"""
risk_analyzer.py - 风险分析模块

提供策略风险分析功能，包括波动率、Beta、Sortino比率、信息比率等指标的计算。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime


class RiskAnalyzer:
    """
    风险分析器
    
    用于计算策略的各项风险指标：
    - 波动率
    - Beta系数
    - Sortino比率
    - 信息比率
    - VaR和CVaR
    
    用法:
        analyzer = RiskAnalyzer(returns, benchmark_returns)
        vol = analyzer.calculate_volatility()
        beta = analyzer.calculate_beta()
        sortino = analyzer.calculate_sortino_ratio()
    """
    
    def __init__(
        self,
        returns: Union[pd.Series, np.ndarray, List, None] = None,
        benchmark_returns: Optional[Union[pd.Series, np.ndarray, List]] = None,
        trading_days: int = 252,
    ):
        """
        初始化风险分析器
        
        Args:
            returns: 策略收益率序列（可选，后续可通过 set_data 设置）
            benchmark_returns: 基准收益率序列 (可选，用于计算Beta等)
            trading_days: 每年交易日数
        """
        self.trading_days = trading_days
        self.returns = None
        self.benchmark_returns = None
        
        # 如果传入了数据，则处理
        if returns is not None:
            self.set_data(returns, benchmark_returns)
    
    def set_data(
        self,
        returns: Union[pd.Series, np.ndarray, List],
        benchmark_returns: Optional[Union[pd.Series, np.ndarray, List]] = None,
    ) -> None:
        """
        设置收益率数据
        
        Args:
            returns: 策略收益率序列
            benchmark_returns: 基准收益率序列 (可选)
        """
        # 处理收益率序列
        if isinstance(returns, pd.Series):
            self.returns = returns.values
        elif isinstance(returns, (list, np.ndarray)):
            self.returns = np.array(returns)
        else:
            raise ValueError("returns 必须是 Series, List 或 np.ndarray")
        
        # 处理基准收益率
        if benchmark_returns is not None:
            if isinstance(benchmark_returns, pd.Series):
                self.benchmark_returns = benchmark_returns.values
            elif isinstance(benchmark_returns, (list, np.ndarray)):
                self.benchmark_returns = np.array(benchmark_returns)
    
    def _check_data(self) -> None:
        """检查数据是否已设置"""
        if self.returns is None:
            raise ValueError("请先调用 set_data() 设置收益率数据")
    
    def calculate_volatility(
        self,
        method: str = "std",
        annualize: bool = True,
    ) -> float:
        """
        计算波动率 (Volatility)
        
        Args:
            method: 计算方法
                - "std": 标准差
                - "mad": 平均绝对偏差
                - "ewm": 指数加权移动平均
            annualize: 是否年化
            
        Returns:
            波动率
        """
        self._check_data()
        if len(self.returns) < 2:
            return 0.0
        
        returns = self.returns[1:]  # 排除第一个(为0)
        
        if method == "std":
            vol = np.std(returns, ddof=1)
        elif method == "mad":
            vol = np.mean(np.abs(returns - np.mean(returns)))
        elif method == "ewm":
            # 指数加权移动平均
            vol = pd.Series(returns).ewm(span=30).std().iloc[-1]
        else:
            raise ValueError(f"未知的计算方法: {method}")
        
        if annualize:
            vol = vol * np.sqrt(self.trading_days)
        
        return vol
    
    def calculate_beta(self) -> float:
        """
        计算 Beta 系数
        
        Beta = Cov(策略收益, 基准收益) / Var(基准收益)
        
        Returns:
            Beta系数
        """
        if self.benchmark_returns is None:
            raise ValueError("需要提供 benchmark_returns 才能计算 Beta")
        
        if len(self.returns) < 2 or len(self.benchmark_returns) < 2:
            return 0.0
        
        # 过滤NaN
        mask = ~(np.isnan(self.returns) | np.isnan(self.benchmark_returns))
        r = self.returns[mask]
        br = self.benchmark_returns[mask]
        
        if len(r) < 2:
            return 0.0
        
        # 计算协方差和方差
        cov = np.cov(r, br, ddof=1)[0, 1]
        var_bench = np.var(br, ddof=1)
        
        if var_bench == 0:
            return 0.0
        
        return cov / var_bench
    
    def calculate_sortino_ratio(
        self,
        target_return: float = 0.0,
        method: str = "excess",
    ) -> float:
        """
        计算 Sortino 比率
        
        Sortino = (策略收益 - 目标收益) / 下行波动率
        
        Args:
            target_return: 目标收益率
            method: 计算方法
                - "excess": 超过目标收益的部分 / 下行波动率
                - "total": 策略收益 / 下行波动率
                
        Returns:
            Sortino比率
        """
        if len(self.returns) < 2:
            return 0.0
        
        returns = self.returns[1:]
        
        # 计算下行偏差 (Downside Deviation)
        if method == "excess":
            # 只考虑低于目标收益的部分
            excess = returns - target_return
            downside_returns = excess[excess < 0]
        else:
            # 考虑所有负收益
            downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf') if np.mean(returns) > target_return else 0.0
        
        downside_deviation = np.std(downside_returns, ddof=1)
        
        if downside_deviation == 0:
            return 0.0
        
        # 年化
        if method == "excess":
            sortino = (np.mean(returns) - target_return) / downside_deviation * np.sqrt(self.trading_days)
        else:
            sortino = np.mean(returns) / downside_deviation * np.sqrt(self.trading_days)
        
        return sortino
    
    def calculate_information_ratio(
        self,
        benchmark_returns: Optional[Union[pd.Series, np.ndarray, List]] = None,
    ) -> float:
        """
        计算信息比率 (Information Ratio)
        
        信息比率 = 超额收益 / 跟踪误差
        
        Args:
            benchmark_returns: 基准收益率（可选，如果初始化时已提供）
            
        Returns:
            信息比率
        """
        # 确定使用的基准收益
        if benchmark_returns is not None:
            if isinstance(benchmark_returns, pd.Series):
                br = benchmark_returns.values
            else:
                br = np.array(benchmark_returns)
        elif self.benchmark_returns is not None:
            br = self.benchmark_returns
        else:
            raise ValueError("需要提供 benchmark_returns 才能计算信息比率")
        
        if len(self.returns) < 2 or len(br) < 2:
            return 0.0
        
        # 过滤NaN
        min_len = min(len(self.returns), len(br))
        r = self.returns[:min_len]
        br = br[:min_len]
        
        mask = ~(np.isnan(r) | np.isnan(br))
        r = r[mask]
        br = br[mask]
        
        if len(r) < 2:
            return 0.0
        
        # 计算超额收益
        excess_returns = r - br
        
        # 信息比率 = 超额收益均值 / 超额收益标准差
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns, ddof=1)
        
        if std_excess == 0:
            return 0.0
        
        # 年化
        ir = mean_excess / std_excess * np.sqrt(self.trading_days)
        
        return ir
    
    def calculate_var(
        self,
        confidence: float = 0.95,
        method: str = "historical",
    ) -> float:
        """
        计算 Value at Risk (VaR)
        
        Args:
            confidence: 置信水平
            method: 计算方法
                - "historical": 历史模拟法
                - "parametric": 参数法 (正态分布)
                
        Returns:
            VaR值 (正值表示损失)
        """
        if len(self.returns) < 2:
            return 0.0
        
        returns = self.returns[1:]
        
        if method == "historical":
            # 历史模拟法
            var = -np.percentile(returns, (1 - confidence) * 100)
        elif method == "parametric":
            # 参数法 (假设正态分布)
            mu = np.mean(returns)
            sigma = np.std(returns, ddof=1)
            z = -stats.norm.ppf(1 - confidence)
            var = -(mu + z * sigma)
        else:
            raise ValueError(f"未知的计算方法: {method}")
        
        return var
    
    def calculate_cvar(
        self,
        confidence: float = 0.95,
        method: str = "historical",
    ) -> float:
        """
        计算 Conditional VaR (CVaR) / Expected Shortfall
        
        CVaR = E[损失 | 损失 > VaR]
        
        Args:
            confidence: 置信水平
            method: 计算方法
                
        Returns:
            CVaR值
        """
        if len(self.returns) < 2:
            return 0.0
        
        returns = self.returns[1:]
        
        # 计算VaR阈值
        if method == "historical":
            var_threshold = np.percentile(returns, (1 - confidence) * 100)
            # CVaR是超出VaR部分的平均损失
            cvar = -np.mean(returns[returns <= var_threshold])
        else:
            # 简化处理，使用历史法
            var_threshold = np.percentile(returns, (1 - confidence) * 100)
            cvar = -np.mean(returns[returns <= var_threshold])
        
        return cvar
    
    def calculate_tracking_error(
        self,
        benchmark_returns: Optional[Union[pd.Series, np.ndarray, List]] = None,
    ) -> float:
        """
        计算跟踪误差 (Tracking Error)
        
        跟踪误差 = 超额收益的标准差
        
        Args:
            benchmark_returns: 基准收益率
            
        Returns:
            跟踪误差 (年化)
        """
        # 确定使用的基准收益
        if benchmark_returns is not None:
            if isinstance(benchmark_returns, pd.Series):
                br = benchmark_returns.values
            else:
                br = np.array(benchmark_returns)
        elif self.benchmark_returns is not None:
            br = self.benchmark_returns
        else:
            raise ValueError("需要提供 benchmark_returns 才能计算跟踪误差")
        
        if len(self.returns) < 2 or len(br) < 2:
            return 0.0
        
        # 过滤NaN
        min_len = min(len(self.returns), len(br))
        r = self.returns[:min_len]
        br = br[:min_len]
        
        mask = ~(np.isnan(r) | np.isnan(br))
        r = r[mask]
        br = br[mask]
        
        if len(r) < 2:
            return 0.0
        
        # 计算跟踪误差
        excess_returns = r - br
        te = np.std(excess_returns, ddof=1) * np.sqrt(self.trading_days)
        
        return te
    
    def calculate_treynor_ratio(
        self,
        benchmark_returns: Optional[Union[pd.Series, np.ndarray, List]] = None,
    ) -> float:
        """
        计算 Treynor 比率
        
        Treynor = (策略收益 - 无风险收益) / Beta
        
        Args:
            benchmark_returns: 基准收益率
            
        Returns:
            Treynor比率
        """
        if benchmark_returns is not None:
            if isinstance(benchmark_returns, pd.Series):
                br = benchmark_returns.values
            else:
                br = np.array(benchmark_returns)
        elif self.benchmark_returns is not None:
            br = self.benchmark_returns
        else:
            raise ValueError("需要提供 benchmark_returns 才能计算 Treynor 比率")
        
        try:
            beta = self.calculate_beta()
        except:
            beta = 0.0
        
        if beta == 0:
            return 0.0
        
        returns = self.returns[1:]
        mean_return = np.mean(returns) * self.trading_days
        rf = 0.03  # 假设无风险利率
        
        treynor = (mean_return - rf) / beta
        
        return treynor
    
    def calculate_omega_ratio(
        self,
        threshold: float = 0.0,
    ) -> float:
        """
        计算 Omega 比率
        
        Omega = 超过阈值的收益之和 / 低于阈值的损失之和
        
        Args:
            threshold: 阈值收益率
            
        Returns:
            Omega比率
        """
        if len(self.returns) < 2:
            return 0.0
        
        returns = self.returns[1:]
        
        gains = np.sum(np.maximum(returns - threshold, 0))
        losses = np.sum(np.maximum(threshold - returns, 0))
        
        if losses == 0:
            return float('inf') if gains > 0 else 0.0
        
        return gains / losses
    
    def get_full_risk_analysis(
        self,
        risk_free_rate: float = 0.03,
    ) -> Dict[str, float]:
        """
        获取完整的风险分析报告
        
        Args:
            risk_free_rate: 无风险利率
            
        Returns:
            包含所有风险指标的字典
        """
        result = {
            # 波动率
            'volatility': self.calculate_volatility(annualize=True),
            
            # Beta
            'beta': 0.0,
            
            # 风险调整收益
            'sortino_ratio': self.calculate_sortino_ratio(
                target_return=risk_free_rate / self.trading_days
            ),
            'information_ratio': 0.0,
            'treynor_ratio': 0.0,
            
            # VaR/CVaR
            'var_95': self.calculate_var(confidence=0.95),
            'cvar_95': self.calculate_cvar(confidence=0.95),
            'var_99': self.calculate_var(confidence=0.99),
            'cvar_99': self.calculate_cvar(confidence=0.99),
            
            # 其他
            'tracking_error': 0.0,
            'omega_ratio': self.calculate_omega_ratio(),
        }
        
        # 如果有基准收益率，计算Beta和IR
        if self.benchmark_returns is not None:
            try:
                result['beta'] = self.calculate_beta()
                result['information_ratio'] = self.calculate_information_ratio()
                result['tracking_error'] = self.calculate_tracking_error()
                result['treynor_ratio'] = self.calculate_treynor_ratio()
            except:
                pass
        
        return result


# ---------- 便捷函数 ----------

def analyze_risk(
    returns: Union[pd.Series, np.ndarray, List],
    benchmark_returns: Optional[Union[pd.Series, np.ndarray, List]] = None,
    trading_days: int = 252,
    risk_free_rate: float = 0.03,
) -> Dict[str, float]:
    """
    便捷风险分析函数
    
    Args:
        returns: 策略收益率
        benchmark_returns: 基准收益率
        trading_days: 每年交易日数
        risk_free_rate: 无风险利率
        
    Returns:
        风险指标字典
    """
    analyzer = RiskAnalyzer(returns, benchmark_returns, trading_days)
    return analyzer.get_full_risk_analysis(risk_free_rate)
