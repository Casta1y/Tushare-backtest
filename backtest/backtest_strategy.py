"""
backtest_strategy.py - Backtrader 策略适配

提供策略基类和常用策略实现，用于回测系统。
支持信号驱动和因子驱动两种策略模式。
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

import backtrader as bt
import pandas as pd
import numpy as np


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ---------- 策略基类 ----------

class BaseStrategy(bt.Strategy):
    """
    策略基类
    
    提供通用的策略功能：
    - 订单管理
    - 持仓管理
    - 信号处理
    - 日志记录
    
    子类应重写以下方法：
    - __init__: 初始化指标和参数
    - next: 主逻辑（每个bar调用）
    - signal_generate: 生成交易信号（可选）
    """
    
    # 类级别的日志记录器
    logger = logger
    
    # 订单管理
    order = None  # 当前挂单
    
    # 策略参数（可在 Config 中配置）
    params = (
        ("printlog", False),  # 是否打印交易日志
        ("print_signal", False),  # 是否打印信号
    )
    
    def __init__(self):
        """
        初始化策略
        
        子类应在 super().__init__() 后添加：
        - 技术指标计算
        - 信号变量初始化
        """
        # 持仓信息
        self.order = None  # 当前挂单
        self.buy_price = None  # 买入价格
        self.buy_comm = None  # 买入佣金
        self.buy_size = None  # 买入数量
        
        # 订单追踪
        self.pending_orders = []  # 待执行订单
        
        # 交易统计
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        
        # 数据别名（简化访问）
        self.data0 = self.datas[0]
        self.dataclose = self.data0.close
        self.dataopen = self.data0.open
        self.datahigh = self.data0.high
        self.datalow = self.data0.low
        self.datavolume = self.data0.volume
        
        # 订单通知回调
        self.order_notify = True
        
        if self.params.printlog:
            self.logger.info(
                f"[{self.__class__.__name__}] 策略初始化完成"
            )
    
    def log(self, txt: str, dt: Optional[datetime] = None) -> None:
        """
        输出日志
        
        Args:
            txt: 日志内容
            dt: 日期时间（默认当前bar时间）
        """
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            self.logger.info(f"[{self.__class__.__name__}] {dt.isoformat()} - {txt}")
    
    def log_signal(self, signal: str, details: str = "") -> None:
        """
        输出信号日志
        
        Args:
            signal: 信号类型 (BUY/SELL/HOLD)
            details: 详细信息
        """
        if self.params.print_signal:
            dt = self.datas[0].datetime.date(0)
            self.logger.info(
                f"[{self.__class__.__name__}] {dt.isoformat()} - "
                f"Signal: {signal} | Price: {self.dataclose[0]:.2f} | {details}"
            )
    
    def notify_order(self, order) -> None:
        """
        订单状态通知
        
        Args:
            order: 订单对象
        """
        if order.status in [order.Submitted, order.Accepted]:
            # 订单已提交或已接受
            return
        
        if order.status in [order.Completed]:
            # 订单已完成
            if order.isbuy():
                self.buy_price = order.executed.price
                self.buy_comm = order.executed.comm
                self.buy_size = order.executed.size
                self.log(
                    f"BUY EXECUTED: Price={order.executed.price:.2f}, "
                    f"Size={order.executed.size}, Cost={order.executed.value:.2f}, "
                    f"Comm={order.executed.comm:.2f}"
                )
            elif order.issell():
                self.log(
                    f"SELL EXECUTED: Price={order.executed.price:.2f}, "
                    f"Size={order.executed.size}, Cost={order.executed.value:.2f}, "
                    f"Comm={order.executed.comm:.2f}"
                )
                self._record_trade(order)
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            # 订单被取消/保证金不足/拒绝
            self.log(f"ORDER CANCELED/REJECTED: Status={order.status}")
        
        # 清除挂单引用
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None
    
    def notify_trade(self, trade) -> None:
        """
        交易通知（平仓时触发）
        
        Args:
            trade: 交易对象
        """
        if not trade.isclosed:
            return
        
        self.log(
            f"TRADE PNL: Gross={trade.pnl:.2f}, Net={trade.pnlcomm:.2f}"
        )
        
        # 统计胜率
        if trade.pnlcomm > 0:
            self.win_count += 1
        elif trade.pnlcomm < 0:
            self.loss_count += 1
        
        self.trade_count += 1
    
    def _record_trade(self, order) -> None:
        """
        记录交易
        
        Args:
            order: 订单对象
        """
        # 可以在这里添加更详细的交易记录逻辑
        # 例如保存到列表或数据库
        pass
    
    def get_position_size(
        self,
        price: float,
        target_pct: float = 1.0,
    ) -> int:
        """
        计算持仓数量
        
        Args:
            price: 当前价格
            target_pct: 目标仓位比例 (0.0 - 1.0)
            
        Returns:
            持仓数量（整数）
        """
        # 获取可用资金
        cash = self.broker.getcash()
        value = self.broker.getvalue()
        
        # 计算目标金额
        target_value = value * target_pct
        
        # 计算可买入数量（向下取整到100的整数倍）
        size = int(target_value / price / 100) * 100
        size = max(size, 0)  # 确保非负
        
        return size
    
    def get_current_position_pct(self) -> float:
        """
        获取当前持仓比例
        
        Returns:
            持仓比例 (0.0 - 1.0)
        """
        if self.position:
            return self.position.size * self.data0.close[0] / self.broker.getvalue()
        return 0.0
    
    def next(self) -> None:
        """
        主逻辑（每个bar调用）
        
        子类应重写此方法实现交易逻辑
        """
        # 默认不交易
        pass
    
    def signal_generate(self) -> Tuple[int, str]:
        """
        生成交易信号
        
        子类可重写此方法
        
        Returns:
            (signal, reason):
                signal: 1 (买入), -1 (卖出), 0 (持有)
                reason: 信号原因
        """
        return 0, "NO_SIGNAL"


# ---------- 简单策略实现 ----------

class MAStrategy(BaseStrategy):
    """
    简单移动平均线策略
    
    金叉买入，死叉卖出
    """
    
    params = (
        ("fast_period", 5),  # 快速均线周期
        ("slow_period", 20),  # 慢速均线周期
        ("printlog", False),
        ("print_signal", False),
    )
    
    def __init__(self):
        super().__init__()
        
        # 计算移动平均线
        self.sma_fast = bt.indicators.SimpleMovingAverage(
            self.datas[0].close,
            period=self.params.fast_period,
        )
        self.sma_slow = bt.indicators.SimpleMovingAverage(
            self.datas[0].close,
            period=self.params.slow_period,
        )
        
        # 交叉信号
        self.crossover = bt.indicators.CrossOver(
            self.sma_fast,
            self.sma_slow,
        )
        
        # 上一时刻状态
        self.was_above = False
        
        if self.params.printlog:
            self.log(
                f"MA策略初始化: fast={self.params.fast_period}, "
                f"slow={self.params.slow_period}"
            )
    
    def next(self):
        # 检查是否有挂单
        if self.order:
            return
        
        # 当前价格
        current_price = self.dataclose[0]
        
        # 获取当前持仓
        position = self.getposition(self.data0)
        
        # 金叉 - 买入信号
        if self.crossover > 0:
            self.log_signal("BUY", f"金叉: MA{self.params.fast_period} > MA{self.params.slow_period}")
            
            # 买入
            size = self.get_position_size(current_price, target_pct=1.0)
            if size > 0:
                self.order = self.buy()
        
        # 死叉 - 卖出信号
        elif self.crossover < 0 and position.size > 0:
            self.log_signal("SELL", f"死叉: MA{self.params.fast_period} < MA{self.params.slow_period}")
            
            # 卖出
            self.order = self.close()


class RSIStrategy(BaseStrategy):
    """
    RSI 策略
    
    RSI < 30 超卖买入，RSI > 70 超买卖出
    """
    
    params = (
        ("rsi_period", 14),  # RSI周期
        ("rsi_lower", 30),  # 超卖阈值
        ("rsi_upper", 70),  # 超买阈值
        ("printlog", False),
        ("print_signal", False),
    )
    
    def __init__(self):
        super().__init__()
        
        # 计算RSI
        self.rsi = bt.indicators.RSI(
            self.datas[0].close,
            period=self.params.rsi_period,
        )
        
        if self.params.printlog:
            self.log(
                f"RSI策略初始化: period={self.params.rsi_period}, "
                f"lower={self.params.rsi_lower}, upper={self.params.rsi_upper}"
            )
    
    def next(self):
        if self.order:
            return
        
        current_price = self.dataclose[0]
        position = self.getposition(self.data0)
        
        # RSI < 30 超卖 - 买入
        if self.rsi < self.params.rsi_lower and position.size == 0:
            self.log_signal("BUY", f"RSI={self.rsi[0]:.2f} < {self.params.rsi_lower} (超卖)")
            
            size = self.get_position_size(current_price, target_pct=1.0)
            if size > 0:
                self.order = self.buy()
        
        # RSI > 70 超买 - 卖出
        elif self.rsi > self.params.rsi_upper and position.size > 0:
            self.log_signal("SELL", f"RSI={self.rsi[0]:.2f} > {self.params.rsi_upper} (超买)")
            
            self.order = self.close()


class DualSignalStrategy(BaseStrategy):
    """
    双因子策略（MA + RSI）
    
    同时满足 MA 金叉和 RSI 超卖时买入
    同时满足 MA 死叉和 RSI 超买时卖出
    """
    
    params = (
        ("fast_period", 5),
        ("slow_period", 20),
        ("rsi_period", 14),
        ("rsi_lower", 30),
        ("rsi_upper", 70),
        ("printlog", False),
        ("print_signal", False),
    )
    
    def __init__(self):
        super().__init__()
        
        # MA 指标
        self.sma_fast = bt.indicators.SimpleMovingAverage(
            self.datas[0].close,
            period=self.params.fast_period,
        )
        self.sma_slow = bt.indicators.SimpleMovingAverage(
            self.datas[0].close,
            period=self.params.slow_period,
        )
        self.crossover = bt.indicators.CrossOver(
            self.sma_fast,
            self.sma_slow,
        )
        
        # RSI 指标
        self.rsi = bt.indicators.RSI(
            self.datas[0].close,
            period=self.params.rsi_period,
        )
    
    def next(self):
        if self.order:
            return
        
        current_price = self.dataclose[0]
        position = self.getposition(self.data0)
        
        # 买入条件：MA金叉 且 RSI < 30
        buy_signal = (self.crossover > 0) and (self.rsi < self.params.rsi_lower)
        
        # 卖出条件：MA死叉 且 RSI > 70
        sell_signal = (self.crossover < 0) and (self.rsi > self.params.rsi_upper)
        
        if buy_signal and position.size == 0:
            self.log_signal(
                "BUY",
                f"MA金叉 + RSI超卖(RSI={self.rsi[0]:.2f})"
            )
            size = self.get_position_size(current_price, target_pct=1.0)
            if size > 0:
                self.order = self.buy()
        
        elif sell_signal and position.size > 0:
            self.log_signal(
                "SELL",
                f"MA死叉 + RSI超买(RSI={self.rsi[0]:.2f})"
            )
            self.order = self.close()


class BreakoutStrategy(BaseStrategy):
    """
    突破策略
    
    价格突破N日高点买入，跌破N日低点卖出
    """
    
    params = (
        ("period", 20),  # 周期
        ("printlog", False),
        ("print_signal", False),
    )
    
    def __init__(self):
        super().__init__()
        
        # 最高价/最低价
        self.highest = bt.indicators.Highest(
            self.datas[0].high,
            period=self.params.period,
        )
        self.lowest = bt.indicators.Lowest(
            self.datas[0].low,
            period=self.params.period,
        )
        
        # 价格位置
        self.price_above_high = False
        self.price_below_low = False
    
    def next(self):
        if self.order:
            return
        
        current_price = self.dataclose[0]
        position = self.getposition(self.data0)
        
        # 突破最高价 - 买入
        if self.dataclose[0] > self.highest[-1] and not self.price_above_high:
            self.log_signal("BUY", f"突破{self.params.period}日高点")
            size = self.get_position_size(current_price, target_pct=1.0)
            if size > 0:
                self.order = self.buy()
            self.price_above_high = True
            self.price_below_low = False
        
        # 跌破最低价 - 卖出
        elif self.dataclose[0] < self.lowest[-1] and not self.price_below_low and position.size > 0:
            self.log_signal("SELL", f"跌破{self.params.period}日低点")
            self.order = self.close()
            self.price_below_low = True
            self.price_above_high = False


class MeanReversionStrategy(BaseStrategy):
    """
    均值回归策略
    
    价格偏离均线过多时反向交易
    """
    
    params = (
        ("period", 20),  # 均线周期
        ("deviation", 2.0),  # 标准差倍数
        ("printlog", False),
        ("print_signal", False),
    )
    
    def __init__(self):
        super().__init__()
        
        # 均线
        self.sma = bt.indicators.SimpleMovingAverage(
            self.datas[0].close,
            period=self.params.period,
        )
        
        # 标准差
        self.std = bt.indicators.StandardDeviation(
            self.datas[0].close,
            period=self.params.period,
        )
        
        # 布林带
        self.bollinger = bt.indicators.BollingerBands(
            self.datas[0].close,
            period=self.params.period,
            devfactor=self.params.deviation,
        )
    
    def next(self):
        if self.order:
            return
        
        current_price = self.dataclose[0]
        position = self.getposition(self.data0)
        
        # 下轨 - 买入
        if current_price < self.bollinger.lines.bot[0] and position.size == 0:
            self.log_signal("BUY", f"价格触布林下轨")
            size = self.get_position_size(current_price, target_pct=1.0)
            if size > 0:
                self.order = self.buy()
        
        # 上轨 - 卖出
        elif current_price > self.bollinger.lines.top[0] and position.size > 0:
            self.log_signal("SELL", f"价格触布林上轨")
            self.order = self.close()
        
        # 中轨 - 卖出（回归均值）
        elif position.size > 0 and current_price >= self.sma[0]:
            self.log_signal("SELL", f"价格回归均线")
            self.order = self.close()


class MultiSignalStrategy(BaseStrategy):
    """
    多信号综合策略
    
    综合多个指标信号，少数服从多数
    """
    
    params = (
        ("ma_short", 5),
        ("ma_long", 20),
        ("rsi_period", 14),
        ("rsi_lower", 30),
        ("rsi_upper", 70),
        ("macd_fast", 12),
        ("macd_slow", 26),
        ("macd_signal", 9),
        ("printlog", False),
        ("print_signal", False),
    )
    
    def __init__(self):
        super().__init__()
        
        # MA
        self.sma_fast = bt.indicators.SimpleMovingAverage(
            self.datas[0].close, period=self.params.ma_short
        )
        self.sma_slow = bt.indicators.SimpleMovingAverage(
            self.datas[0].close, period=self.params.ma_long
        )
        self.ma_cross = bt.indicators.CrossOver(
            self.sma_fast, self.sma_slow
        )
        
        # RSI
        self.rsi = bt.indicators.RSI(
            self.datas[0].close, period=self.params.rsi_period
        )
        
        # MACD
        self.macd = bt.indicators.MACD(
            self.datas[0].close,
            period_fast=self.params.macd_fast,
            period_slow=self.params.macd_slow,
            period_signal=self.params.macd_signal,
        )
        self.macd_cross = bt.indicators.CrossOver(
            self.macd.lines.macd, self.macd.lines.signal
        )
    
    def next(self):
        if self.order:
            return
        
        position = self.getposition(self.data0)
        
        # 统计买入信号
        buy_signals = 0
        sell_signals = 0
        
        # MA 信号
        if self.ma_cross > 0:
            buy_signals += 1
        elif self.ma_cross < 0:
            sell_signals += 1
        
        # RSI 信号
        if self.rsi < self.params.rsi_lower:
            buy_signals += 1
        elif self.rsi > self.params.rsi_upper:
            sell_signals += 1
        
        # MACD 信号
        if self.macd_cross > 0:
            buy_signals += 1
        elif self.macd_cross < 0:
            sell_signals += 1
        
        # 买入：至少2个买入信号
        if buy_signals >= 2 and position.size == 0:
            self.log_signal(
                "BUY",
                f"买入信号: {buy_signals}/3 (MA:{int(self.ma_cross>0)} RSI:{int(self.rsi<30)} MACD:{int(self.macd_cross>0)})"
            )
            self.order = self.buy()
        
        # 卖出：至少2个卖出信号
        elif sell_signals >= 2 and position.size > 0:
            self.log_signal(
                "SELL",
                f"卖出信号: {sell_signals}/3"
            )
            self.order = self.close()


# ---------- 因子驱动策略基类 ----------

class FactorStrategy(BaseStrategy):
    """
    因子驱动策略基类
    
    接收因子信号进行交易
    子类需要实现 compute_signal 方法
    """
    
    params = (
        ("factor_data", None),  # 因子数据 DataFrame
        ("factor_name", "factor"),  # 因子列名
        ("top_n", 10),  # 买入排名前N
        ("bottom_n", 0),  # 卖出排名后N（0表示不平仓）
        ("printlog", False),
        ("print_signal", False),
    )
    
    def __init__(self):
        super().__init__()
        
        self.factor_data = self.params.factor_data
        self.factor_name = self.params.factor_name
        
        # 当前日期索引
        self.current_idx = -1
        
        if self.params.printlog and self.factor_data is not None:
            self.log(
                f"因子策略初始化: 因子={self.factor_name}, "
                f"数据量={len(self.factor_data)}"
            )
    
    def compute_signal(self, current_date) -> int:
        """
        计算信号
        
        子类应重写此方法
        
        Args:
            current_date: 当前日期
            
        Returns:
            1 (买入), -1 (卖出), 0 (持有)
        """
        return 0
    
    def next(self):
        # 获取当前日期
        current_date = self.datas[0].datetime.date(0)
        
        # 计算信号
        signal = self.compute_signal(current_date)
        
        # 执行交易
        if self.order:
            return
        
        position = self.getposition(self.data0)
        current_price = self.dataclose[0]
        
        if signal > 0 and position.size == 0:
            self.log_signal("BUY", f"因子信号: {self.factor_name}")
            size = self.get_position_size(current_price, target_pct=1.0)
            if size > 0:
                self.order = self.buy()
        
        elif signal < 0 and position.size > 0:
            self.log_signal("SELL", f"因子信号: {self.factor_name}")
            self.order = self.close()


# ---------- 策略映射 ----------

# 策略名称到类的映射
STRATEGY_MAP = {
    "ma": MAStrategy,
    "ma_strategy": MAStrategy,
    "rsi": RSIStrategy,
    "rsi_strategy": RSIStrategy,
    "dual": DualSignalStrategy,
    "dual_signal": DualSignalStrategy,
    "breakout": BreakoutStrategy,
    "breakout_strategy": BreakoutStrategy,
    "mean_reversion": MeanReversionStrategy,
    "mean_reversion_strategy": MeanReversionStrategy,
    "multi": MultiSignalStrategy,
    "multi_signal": MultiSignalStrategy,
    "factor": FactorStrategy,
    "factor_strategy": FactorStrategy,
    "base": BaseStrategy,
    "base_strategy": BaseStrategy,
}


def get_strategy(name: str) -> type:
    """
    根据名称获取策略类
    
    Args:
        name: 策略名称
        
    Returns:
        策略类
        
    Raises:
        ValueError: 策略不存在
    """
    name_lower = name.lower()
    if name_lower not in STRATEGY_MAP:
        raise ValueError(f"未知策略: {name}, 可用策略: {list(STRATEGY_MAP.keys())}")
    return STRATEGY_MAP[name_lower]


# ---------- 便捷函数 ----------

def create_strategy(
    name: str,
    **params
) -> type:
    """
    创建策略类的便捷函数
    
    Args:
        name: 策略名称
        **params: 策略参数
        
    Returns:
        策略类（参数已绑定）
    """
    strategy_class = get_strategy(name)
    
    # 绑定参数
    class ConfiguredStrategy(strategy_class):
        pass
    
    # 设置参数
    for key, value in params.items():
        setattr(ConfiguredStrategy.params, key, value)
    
    return ConfiguredStrategy
