"""
performance_analyzer.py - 性能分析模块

提供策略性能分析功能，包括收益率、夏普比率、最大回撤等指标的计算。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime


class PerformanceAnalyzer:
    """
    性能分析器

    用于计算策略的各项性能指标：
    - 收益率指标
    - 风险调整收益指标
    - 回撤指标
    - 胜率指标

    用法:
        analyzer = PerformanceAnalyzer(equity_curve)
        returns = analyzer.calculate_returns()
        sharpe = analyzer.calculate_sharpe_ratio()
        drawdown = analyzer.calculate_max_drawdown()
    """

    def __init__(
        self,
        equity_curve: Union[pd.DataFrame, pd.Series, List, np.ndarray, None] = None,
        risk_free_rate: float = 0.03,
        trading_days: int = 252,
    ):
        """
        初始化性能分析器

        Args:
            equity_curve: 权益曲线（可选），可以是:
                - DataFrame: 包含 'date' 和 'value' 列
                - Series: 索引为日期，值为权益
                - List/np.ndarray: 权益值序列
                - None: 后续通过 set_data() 设置
            risk_free_rate: 无风险利率 (年化)
            trading_days: 每年交易日数
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
        self.equity = None
        self.returns = None

        # 如果传入了数据，则处理
        if equity_curve is not None:
            self.set_data(equity_curve)

    def set_data(self, equity_curve: Union[pd.DataFrame, pd.Series, List, np.ndarray]) -> None:
        """
        设置权益曲线数据

        Args:
            equity_curve: 权益曲线
        """
        # 处理 equity_curve
        if isinstance(equity_curve, pd.DataFrame):
            if 'value' in equity_curve.columns:
                self.equity = equity_curve['value'].values
            elif 'equity' in equity_curve.columns:
                self.equity = equity_curve['equity'].values
            else:
                self.equity = equity_curve.iloc[:, 0].values
        elif isinstance(equity_curve, pd.Series):
            self.equity = equity_curve.values
        elif isinstance(equity_curve, (list, np.ndarray)):
            self.equity = np.array(equity_curve)
        else:
            raise ValueError("equity_curve 必须是 DataFrame, Series, List 或 np.ndarray")

        # 计算收益率序列
        self.returns = self._calculate_returns()
    
    def _check_data(self) -> None:
        """检查数据是否已设置"""
        if self.equity is None:
            raise ValueError("请先调用 set_data() 设置权益曲线数据")
    
    def _calculate_returns(self) -> np.ndarray:
        """
        计算收益率序列

        Returns:
            收益率数组
        """
        if len(self.equity) < 2:
            return np.array([0.0])

        # 使用百分比收益率
        returns = np.diff(self.equity) / self.equity[:-1]
        returns = np.insert(returns, 0, 0.0)  # 第一个为0

        return returns

    def calculate_returns(
        self,
        method: str = "total"
    ) -> Union[float, Dict[str, float]]:
        """
        计算收益率

        Args:
            method: 计算方法
                - "total": 总收益率
                - "daily": 日均收益率
                - "annual": 年化收益率
                - "all": 返回所有收益率指标

        Returns:
            收益率值或包含多个指标的字典
        """
        self._check_data()
        
        if len(self.equity) < 2:
            return 0.0 if method != "all" else {
                'total': 0.0, 'daily': 0.0, 'annual': 0.0
            }

        total_return = (self.equity[-1] / self.equity[0] - 1) if self.equity[0] != 0 else 0

        # 日均收益率
        daily_return = np.mean(self.returns[1:])

        # 年化收益率
        n_days = len(self.equity)
        years = n_days / self.trading_days
        if years > 0 and self.equity[0] != 0:
            annual_return = (self.equity[-1] / self.equity[0]) ** (1 / years) - 1
        else:
            annual_return = 0.0

        if method == "total":
            return total_return
        elif method == "daily":
            return daily_return
        elif method == "annual":
            return annual_return
        elif method == "all":
            return {
                'total': total_return,
                'daily': daily_return,
                'annual': annual_return,
            }
        else:
            raise ValueError(f"未知的计算方法: {method}")

    def calculate_sharpe_ratio(
        self,
        period: str = "daily"
    ) -> float:
        """
        计算夏普比率 (Sharpe Ratio)

        夏普比率 = (策略收益 - 无风险收益) / 策略收益标准差

        Args:
            period: 收益周期
                - "daily": 日收益
                - "monthly": 月收益
                - "annual": 年收益

        Returns:
            夏普比率
        """
        self._check_data()
        
        period = period.lower()

        if len(self.returns) < 2:
            return 0.0

        # 计算收益标准差
        returns_std = np.std(self.returns[1:], ddof=1)

        if returns_std == 0:
            return 0.0

        # 根据周期调整
        if period == "daily":
            # 日夏普转年化
            daily_rf = self.risk_free_rate / self.trading_days
            excess_returns = self.returns[1:] - daily_rf
            sharpe = np.mean(excess_returns) / returns_std * np.sqrt(self.trading_days)
        elif period == "monthly":
            # 月夏普转年化
            monthly_rf = self.risk_free_rate / 12
            excess_returns = self.returns[1:] - monthly_rf
            sharpe = np.mean(excess_returns) / returns_std * np.sqrt(12)
        elif period == "annual":
            sharpe = (np.mean(self.returns[1:]) - self.risk_free_rate) / returns_std
        else:
            raise ValueError(f"未知的周期: {period}")

        return sharpe

    def calculate_max_drawdown(self) -> Tuple[float, int, int]:
        """
        计算最大回撤 (Maximum Drawdown)

        Returns:
            Tuple[float, int, int]: (最大回撤值, 回撤开始日期索引, 回撤结束日期索引)
        """
        self._check_data()
        if len(self.equity) < 2:
            return (0.0, 0, 0)

        # 计算累计最高值
        cummax = np.maximum.accumulate(self.equity)

        # 计算回撤
        drawdown = (self.equity - cummax) / cummax

        # 找到最大回撤
        max_dd = np.min(drawdown)
        max_dd_idx = np.argmin(drawdown)

        # 找到回撤开始的日期（最高点）
        start_idx = np.argmax(self.equity[:max_dd_idx + 1])

        return (float(max_dd), int(start_idx), int(max_dd_idx))

    def calculate_annual_return(
        self,
        method: str = "compound"
    ) -> float:
        """
        计算年化收益率

        Args:
            method: 计算方法
                - "compound": 复合年化收益率 (CAGR)
                - "simple": 简单年化收益率

        Returns:
            年化收益率
        """
        self._check_data()
        
        method = method.lower()
        if len(self.equity) < 2:
            return 0.0

        n_days = len(self.equity)
        years = n_days / self.trading_days

        if years <= 0:
            return 0.0

        total_return = self.equity[-1] / self.equity[0] - 1

        if method == "compound":
            # 复合年化收益率 (CAGR)
            annual_return = (self.equity[-1] / self.equity[0]) ** (1 / years) - 1
        elif method == "simple":
            # 简单年化收益率
            annual_return = total_return / years
        else:
            raise ValueError(f"未知的计算方法: {method}")

        return annual_return

    def calculate_win_rate(
        self,
        threshold: float = 0.0
    ) -> Dict[str, Union[float, int]]:
        """
        计算胜率

        Args:
            threshold: 判定为盈利的阈值

        Returns:
            包含胜率统计的字典:
            - win_rate: 胜率
            - total_trades: 总交易次数
            - winning_trades: 盈利交易次数
            - losing_trades: 亏损交易次数
            - avg_win: 平均盈利
            - avg_loss: 平均亏损
        """
        self._check_data()
        if len(self.returns) < 2:
            return {
                'win_rate': 0.0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
            }

        # 过滤掉收益为0的交易（不交易的日子）
        trade_returns = self.returns[1:]

        winning = trade_returns > threshold
        losing = trade_returns < -threshold

        winning_trades = int(np.sum(winning))
        losing_trades = int(np.sum(losing))
        total_trades = winning_trades + losing_trades

        if total_trades == 0:
            return {
                'win_rate': 0.0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
            }

        win_rate = winning_trades / total_trades

        # 计算平均盈利和亏损
        if winning_trades > 0:
            avg_win = np.mean(trade_returns[winning])
        else:
            avg_win = 0.0

        if losing_trades > 0:
            avg_loss = np.mean(trade_returns[losing])
        else:
            avg_loss = 0.0

        return {
            'win_rate': win_rate,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
        }

    def calculate_calmar_ratio(self) -> float:
        """
        计算卡尔玛比率 (Calmar Ratio)

        卡尔玛比率 = 年化收益率 / 最大回撤

        Returns:
            卡尔玛比率
        """
        self._check_data()
        
        annual_return = self.calculate_annual_return()
        max_dd, _, _ = self.calculate_max_drawdown()

        if max_dd == 0:
            return 0.0

        return annual_return / abs(max_dd)

    def calculate_profit_factor(self) -> float:
        """
        计算盈利因子 (Profit Factor)

        盈利因子 = 总盈利 / 总亏损

        Returns:
            盈利因子
        """
        self._check_data()
        
        if len(self.returns) < 2:
            return 0.0

        trade_returns = self.returns[1:]

        gross_profit = np.sum(trade_returns[trade_returns > 0])
        gross_loss = np.abs(np.sum(trade_returns[trade_returns < 0]))

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def get_full_analysis(self) -> Dict[str, float]:
        """
        获取完整的性能分析报告

        Returns:
            包含所有性能指标的字典
        """
        returns = self.calculate_returns("all")
        max_dd, dd_start, dd_end = self.calculate_max_drawdown()
        win_stats = self.calculate_win_rate()

        return {
            # 收益率指标
            'total_return': returns['total'],
            'daily_return': returns['daily'],
            'annual_return': returns['annual'],

            # 风险调整收益
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            'calmar_ratio': self.calculate_calmar_ratio(),
            'profit_factor': self.calculate_profit_factor(),

            # 回撤指标
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd * 100,

            # 胜率统计
            'win_rate': win_stats['win_rate'],
            'total_trades': win_stats['total_trades'],
            'winning_trades': win_stats['winning_trades'],
            'losing_trades': win_stats['losing_trades'],
            'avg_win': win_stats['avg_win'],
            'avg_loss': win_stats['avg_loss'],

            # 波动率
            'volatility': np.std(self.returns[1:]) * np.sqrt(self.trading_days),
        }


# ---------- 便捷函数 ----------

def analyze_performance(
    equity_curve: Union[pd.DataFrame, pd.Series, List, np.ndarray],
    risk_free_rate: float = 0.03,
    trading_days: int = 252,
) -> Dict[str, float]:
    """
    便捷性能分析函数

    Args:
        equity_curve: 权益曲线
        risk_free_rate: 无风险利率
        trading_days: 每年交易日数

    Returns:
        性能指标字典
    """
    analyzer = PerformanceAnalyzer(equity_curve, risk_free_rate, trading_days)
    return analyzer.get_full_analysis()
