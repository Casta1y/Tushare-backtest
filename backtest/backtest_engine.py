"""
backtest_engine.py - 回测引擎框架

提供 BacktestEngine 类，用于运行回测并管理回测生命周期。
基于 Backtrader 框架实现。
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict

import backtrader as bt
import pandas as pd
import numpy as np

from .backtest_config import Config, BacktestConfig


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """交易记录"""
    datetime: str
    symbol: str
    action: str  # BUY / SELL
    price: float
    volume: int
    commission: float
    pnl: float = 0.0
    pnl_pct: float = 0.0


@dataclass
class PositionRecord:
    """持仓记录"""
    datetime: str
    symbol: str
    volume: int
    avg_price: float
    market_value: float
    unrealized_pnl: float = 0.0


@dataclass
class BacktestResult:
    """回测结果数据类"""
    # 基本信息
    strategy_name: str
    start_date: str
    end_date: str
    initial_cash: float
    final_value: float
    
    # 收益率指标
    total_return: float = 0.0
    total_return_pct: float = 0.0
    annual_return: float = 0.0
    annual_return_pct: float = 0.0
    
    # 风险指标
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    volatility: float = 0.0
    
    # 收益指标
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # 交易统计
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    
    # 持仓统计
    holding_days: int = 0
    max_position: float = 0.0
    
    # 详细记录
    trades: List[Dict] = field(default_factory=list)
    positions: List[Dict] = field(default_factory=list)
    equity_curve: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def summary(self) -> str:
        """生成结果摘要"""
        lines = [
            "=" * 50,
            f"回测结果摘要 - {self.strategy_name}",
            "=" * 50,
            f"回测区间: {self.start_date} ~ {self.end_date}",
            f"初始资金: ¥{self.initial_cash:,.2f}",
            f"最终资产: ¥{self.final_value:,.2f}",
            f"总收益率: {self.total_return_pct:.2f}%",
            f"年化收益率: {self.annual_return_pct:.2f}%",
            f"最大回撤: {self.max_drawdown_pct:.2f}%",
            f"夏普比率: {self.sharpe_ratio:.2f}",
            f"交易次数: {self.total_trades}",
            f"胜率: {self.win_rate:.2f}%",
            "=" * 50,
        ]
        return "\n".join(lines)


class BacktestEngine:
    """
    回测引擎类
    
    基于 Backtrader 框架，提供完整的回测功能：
    - 策略执行
    - 交易记录
    - 性能分析
    - 结果输出
    
    用法:
        engine = BacktestEngine(config)
        engine.set_data(dataframe)
        engine.set_strategy(MyStrategy)
        results = engine.run()
        engine.save_results("output.json")
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        initial_cash: float = 100000.0,
        strategy_name: str = "default_strategy",
        start_date: str = "20200101",
        end_date: str = "20231231",
        commission_rate: float = 0.0003,
        stamp_tax_rate: float = 0.001,
        slippage: float = 0.001,
        benchmark: str = "000300.SH",
    ):
        """
        初始化回测引擎
        
        Args:
            config: Config 对象，如果提供则使用其中的回测配置
            initial_cash: 初始资金
            strategy_name: 策略名称
            start_date: 回测开始日期 (YYYYMMDD)
            end_date: 回测结束日期 (YYYYMMDD)
            commission_rate: 佣金费率
            stamp_tax_rate: 印花税率（卖出时收取）
            slippage: 滑点
            benchmark: 基准代码
        """
        # 使用 Config 对象或手动参数
        if config is not None:
            self.config: Config = config
            bt_config: BacktestConfig = config.backtest
            self.initial_cash = bt_config.initial_cash
            self.strategy_name = bt_config.strategy_name
            self.start_date = bt_config.start_date
            self.end_date = bt_config.end_date
            self.commission_rate = bt_config.commission_rate
            self.stamp_tax_rate = bt_config.stamp_tax_rate
            self.slippage = bt_config.slippage
            self.benchmark = bt_config.benchmark
        else:
            self.config = None
            self.initial_cash = initial_cash
            self.strategy_name = strategy_name
            self.start_date = start_date
            self.end_date = end_date
            self.commission_rate = commission_rate
            self.stamp_tax_rate = stamp_tax_rate
            self.slippage = slippage
            self.benchmark = benchmark
        
        # 日志
        self.logger = logger
        
        # Backtrader 相关
        self.cerebro = bt.Cerebro()
        self._setup_cerebro()
        
        # 数据和策略
        self.data_feed = None
        self.strategy_class = None
        self.strategy_params = {}
        
        # 回测结果
        self.results: Optional[BacktestResult] = None
        self.trades: List[TradeRecord] = []
        self.equity_curve: List[Dict] = []
        
        # 日志
        self.logger = logger
        
        self.logger.info(
            f"[BacktestEngine] 初始化完成: 初始资金={self.initial_cash}, "
            f"策略={self.strategy_name}, 区间={self.start_date}~{self.end_date}"
        )
    
    def _setup_cerebro(self) -> None:
        """配置 Cerebro"""
        # 设置初始资金
        self.cerebro.broker.setcash(self.initial_cash)
        
        # 设置佣金（买入时收取）
        self.cerebro.broker.setcommission(
            commission=self.commission_rate,
        )
        
        # 设置滑点（通过自定义佣金实现）
        # 注意：Backtrader 的 slippage 实现有限，这里使用简化处理
        
        # 设置订单类型为市价单
        self.cerebro.broker.set_coc(True)  # Cheat on Close
        
        # 添加分析器
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        self.cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="time_return")
        self.cerebro.addanalyzer(bt.analyzers.SQN, _name="sqn")
        
        # 添加观察器（用于绘图）
        self.cerebro.addobserver(bt.observers.Value)
        self.cerebro.addobserver(bt.observers.DrawDown)
        
        self.logger.debug("[BacktestEngine] Cerebro 配置完成")
    
    def set_data(
        self,
        data: pd.DataFrame,
        dataname: str = "data",
        fromdate: Optional[datetime] = None,
        todate: Optional[datetime] = None,
    ) -> "BacktestEngine":
        """
        设置数据源
        
        Args:
            data: 包含 OHLCV 数据的 DataFrame
            dataname: 数据名称
            fromdate: 开始日期
            todate: 结束日期
            
        Returns:
            self
            
        Raises:
            ValueError: 数据格式不正确
        """
        # 验证数据格式
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [c for c in required_cols if c not in data.columns]
        if missing_cols:
            raise ValueError(f"数据缺少必要列: {missing_cols}")
        
        # 处理日期列
        df = data.copy()
        
        # 尝试识别日期列
        date_col = None
        for col in ["date", "trade_date", "datetime", "time"]:
            if col in df.columns:
                date_col = col
                break
        
        if date_col:
            df["datetime"] = pd.to_datetime(df[date_col])
        elif isinstance(df.index, pd.DatetimeIndex):
            df["datetime"] = df.index
        else:
            # 使用默认日期索引
            df["datetime"] = pd.date_range(start="2020-01-01", periods=len(df), freq="D")
        
        df = df.sort_values("datetime")
        df.set_index("datetime", inplace=True)
        
        # 创建 Backtrader 数据源
        data_feed = bt.feeds.PandasData(
            dataname=dataname,
            fromdate=fromdate or df.index.min(),
            todate=todate or df.index.max(),
            datetime=None,  # 使用索引
            open="open",
            high="high",
            low="low",
            close="close",
            volume="volume",
            openinterest=-1,
        )
        
        self.cerebro.adddata(data_feed)
        self.data_feed = data_feed
        
        self.logger.info(f"[BacktestEngine] 数据加载完成: {len(df)} 条记录")
        
        return self
    
    def set_strategy(
        self,
        strategy_class: type,
        **params
    ) -> "BacktestEngine":
        """
        设置策略
        
        Args:
            strategy_class: 策略类（继承 bt.Strategy）
            **params: 策略参数
            
        Returns:
            self
        """
        self.strategy_class = strategy_class
        self.strategy_params = params
        
        self.cerebro.addstrategy(strategy_class, **params)
        
        self.logger.info(
            f"[BacktestEngine] 策略设置完成: {strategy_class.__name__}, "
            f"参数: {params}"
        )
        
        return self
    
    def add_strategy(
        self,
        strategy_class: type,
        **params
    ) -> "BacktestEngine":
        """
        添加额外策略（多策略组合）
        
        Args:
            strategy_class: 策略类
            **params: 策略参数
            
        Returns:
            self
        """
        self.cerebro.addstrategy(strategy_class, **params)
        
        self.logger.info(
            f"[BacktestEngine] 添加额外策略: {strategy_class.__name__}"
        )
        
        return self
    
    def run(self) -> BacktestResult:
        """
        运行回测
        
        Returns:
            BacktestResult: 回测结果
        """
        self.logger.info("[BacktestEngine] 开始运行回测...")
        
        # 运行回测
        try:
            self.results = self.cerebro.run()
            self.strategy = self.results[0]  # 第一个策略的结果
            
            # 获取最终资产
            final_value = self.cerebro.broker.getvalue()
            
            # 提取分析结果
            self._extract_analysis()
            
            # 构建结果对象
            result = self._build_result(final_value)
            
            self.results = result
            
            self.logger.info(
                f"[BacktestEngine] 回测完成: 最终资产=¥{final_value:,.2f}, "
                f"收益率={(final_value / self.initial_cash - 1) * 100:.2f}%"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"[BacktestEngine] 回测运行失败: {e}")
            raise
    
    def _extract_analysis(self) -> None:
        """从策略中提取分析结果"""
        if not hasattr(self, 'strategy') or self.strategy is None:
            return
        
        # 可以在这里添加更多分析结果提取逻辑
        # 当前主要由 _build_result 统一处理
    
    def _build_result(self, final_value: float) -> BacktestResult:
        """
        构建回测结果对象
        
        Args:
            final_value: 最终资产值
            
        Returns:
            BacktestResult 对象
        """
        # 初始化默认值
        total_return = final_value - self.initial_cash
        total_return_pct = (final_value / self.initial_cash - 1) * 100
        
        # 计算年化收益率
        start_dt = datetime.strptime(self.start_date, "%Y%m%d")
        end_dt = datetime.strptime(self.end_date, "%Y%m%d")
        years = (end_dt - start_dt).days / 365.25
        years = max(years, 0.01)  # 避免除零
        annual_return_pct = ((final_value / self.initial_cash) ** (1 / years) - 1) * 100
        annual_return = self.initial_cash * annual_return_pct / 100
        
        # 默认风险指标（后续由 analyzer 计算）
        max_drawdown_pct = 0.0
        sharpe_ratio = 0.0
        volatility = 0.0
        win_rate = 0.0
        total_trades = 0
        
        # 尝试从分析器获取数据
        if hasattr(self, 'strategy') and self.strategy:
            # 获取 Returns 分析器
            if hasattr(self.strategy, "analyzers"):
                for name, analyzer in self.strategy.analyzers.items():
                    if name == "returns":
                        try:
                            ret = analyzer.get_analysis()
                            if ret:
                                if 'rtot' in ret:
                                    total_return_pct = ret['rtot'] * 100
                                if 'rnorm' in ret:
                                    annual_return_pct = ret['rnorm'] * 100
                        except:
                            pass
                    
                    elif name == "drawdown":
                        try:
                            dd = analyzer.get_analysis()
                            if dd and 'max' in dd:
                                max_drawdown_pct = dd['max'].get('drawdown', 0)
                        except:
                            pass
                    
                    elif name == "sharpe":
                        try:
                            sr = analyzer.get_analysis()
                            if sr and 'sharperatio' in sr:
                                sharpe_ratio = sr['sharperatio'] or 0.0
                        except:
                            pass
                    
                    elif name == "trades":
                        try:
                            ta = analyzer.get_analysis()
                            if ta:
                                # 获取交易统计
                                total_trades = ta.get('total', {}).get('total', 0)
                                won = ta.get('won', {}).get('total', 0)
                                lost = ta.get('lost', {}).get('total', 0)
                                if total_trades > 0:
                                    win_rate = won / total_trades * 100
                        except:
                            pass
        
        result = BacktestResult(
            strategy_name=self.strategy_name,
            start_date=self.start_date,
            end_date=self.end_date,
            initial_cash=self.initial_cash,
            final_value=final_value,
            total_return=total_return,
            total_return_pct=total_return_pct,
            annual_return=annual_return,
            annual_return_pct=annual_return_pct,
            max_drawdown=max_drawdown_pct,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            volatility=volatility,
            total_trades=total_trades,
            win_rate=win_rate,
        )
        
        return result
    
    def get_results(self) -> Optional[BacktestResult]:
        """
        获取回测结果
        
        Returns:
            BacktestResult 或 None
        """
        return self.results
    
    def save_results(
        self,
        filepath: str,
        include_details: bool = True,
    ) -> None:
        """
        保存回测结果到文件
        
        Args:
            filepath: 保存路径
            include_details: 是否包含详细交易记录
        """
        if self.results is None:
            self.logger.warning("[BacktestEngine] 没有可保存的结果")
            return
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        
        # 转换为字典
        result_dict = self.results.to_dict()
        
        # 根据扩展名选择格式
        if filepath.endswith(".json"):
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=2)
        elif filepath.endswith(".csv"):
            # 保存摘要信息
            df = pd.DataFrame([{
                k: v for k, v in result_dict.items()
                if not isinstance(v, (list, dict))
            }])
            df.to_csv(filepath, index=False, encoding="utf-8-sig")
        else:
            # 默认 JSON
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"[BacktestEngine] 结果已保存: {filepath}")
    
    def print_results(self) -> None:
        """打印回测结果摘要"""
        if self.results:
            print(self.results.summary())
        else:
            print("[BacktestEngine] 没有可显示的结果")
    
    def get_equity_curve(self) -> pd.DataFrame:
        """
        获取权益曲线
        
        Returns:
            包含日期和权益的 DataFrame
        """
        # TODO: 从观察器获取完整的权益曲线
        return pd.DataFrame(self.equity_curve)
    
    def get_trades(self) -> List[TradeRecord]:
        """
        获取交易记录
        
        Returns:
            交易记录列表
        """
        return self.trades
    
    def __repr__(self) -> str:
        return (
            f"BacktestEngine(strategy={self.strategy_name}, "
            f"cash={self.initial_cash}, "
            f"period={self.start_date}~{self.end_date})"
        )


# ---------- 便捷函数 ----------

def create_engine(
    config_path: Optional[str] = None,
    **kwargs
) -> BacktestEngine:
    """
    创建回测引擎的便捷函数
    
    Args:
        config_path: 配置文件路径
        **kwargs: 直接传递的回测参数
        
    Returns:
        BacktestEngine 实例
    """
    if config_path:
        config = Config(config_path)
        return BacktestEngine(config=config, **kwargs)
    else:
        return BacktestEngine(**kwargs)


def run_backtest(
    data: pd.DataFrame,
    strategy_class: type,
    config: Optional[Config] = None,
    **kwargs
) -> BacktestResult:
    """
    一键运行回测的便捷函数
    
    Args:
        data: 行情数据 DataFrame
        strategy_class: 策略类
        config: Config 对象
        **kwargs: 回测参数
        
    Returns:
        BacktestResult 对象
        
    用法:
        result = run_backtest(
            data=df,
            strategy_class=MyStrategy,
            initial_cash=100000
        )
    """
    engine = BacktestEngine(config=config, **kwargs)
    engine.set_data(data)
    engine.set_strategy(strategy_class)
    return engine.run()
