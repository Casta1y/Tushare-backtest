"""
backtest_config.py - 回测配置管理模块

提供 Config 类，用于加载和管理 YAML 配置文件，
以及 BacktestConfig 数据类封装回测相关参数。
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import yaml


@dataclass
class TushareConfig:
    """Tushare API 配置"""
    token: str = ""
    cache_dir: str = "./data/cache"
    retry_times: int = 3
    retry_delay: float = 1.0


@dataclass
class BacktestConfig:
    """回测运行参数"""
    initial_cash: float = 100000.0
    strategy_name: str = "default_strategy"
    start_date: str = "20200101"
    end_date: str = "20231231"
    benchmark: str = "000300.SH"
    commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.001
    slippage: float = 0.001
    min_volume: int = 100
    allow_short: bool = False
    stop_loss: float = 0.05
    take_profit: float = 0.10
    max_position: float = 0.3
    max_total_position: float = 1.0


@dataclass
class StrategyConfig:
    """策略参数配置"""
    ma_short: int = 5
    ma_long: int = 20
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    stop_loss: float = 0.05
    take_profit: float = 0.10
    max_position: float = 0.3
    max_total_position: float = 1.0


@dataclass
class DataConfig:
    """数据获取配置"""
    source: str = "tushare"
    data_type: str = "daily"
    fields: List[str] = field(default_factory=lambda: [
        "trade_date", "ts_code", "open", "high", "low", "close", "volume", "amount"
    ])
    fill_missing: str = "ffill"
    remove_outliers: bool = True
    outlier_std: float = 5.0


@dataclass
class OutputConfig:
    """输出/日志配置"""
    results_dir: str = "./output/results"
    plots_dir: str = "./output/plots"
    report_path: str = "./output/backtest_report.html"
    save_intermediate: bool = True
    log_level: str = "INFO"


class Config:
    """
    全局配置管理类
    
    负责从 YAML 文件加载配置，提供统一的配置访问接口。
    
    用法:
        config = Config("config/config.yaml")
        initial_cash = config.backtest.initial_cash
        tushare_token = config.tushare.token
    """
    
    # 默认配置文件路径
    DEFAULT_CONFIG_PATH = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        "config", "config.yaml"
    )
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: YAML 配置文件路径，默认为 config/config.yaml
        """
        self._config_path = config_path or self.DEFAULT_CONFIG_PATH
        self._raw_config: Dict[str, Any] = {}
        
        # 子配置对象
        self.tushare: TushareConfig = TushareConfig()
        self.backtest: BacktestConfig = BacktestConfig()
        self.strategy: StrategyConfig = StrategyConfig()
        self.data: DataConfig = DataConfig()
        self.output: OutputConfig = OutputConfig()
        
        self._load()
    
    def _load(self) -> None:
        """从 YAML 文件加载配置"""
        if not os.path.exists(self._config_path):
            print(f"[Config] 配置文件不存在 ({self._config_path})，使用默认配置")
            self._apply_defaults()
            return
        
        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                self._raw_config = yaml.safe_load(f) or {}
            
            self._parse_tushare()
            self._parse_backtest()
            self._parse_strategy()
            self._parse_data()
            self._parse_output()
            
            print(f"[Config] 配置加载成功: {self._config_path}")
            
        except yaml.YAMLError as e:
            print(f"[Config] YAML 解析错误: {e}，使用默认配置")
            self._apply_defaults()
        except Exception as e:
            print(f"[Config] 配置加载失败: {e}，使用默认配置")
            self._apply_defaults()
    
    def _apply_defaults(self) -> None:
        """应用所有默认配置"""
        self._parse_tushare()
        self._parse_backtest()
        self._parse_strategy()
        self._parse_data()
        self._parse_output()
    
    def _get(self, section: str, key: str, default: Any = None) -> Any:
        """安全获取配置值"""
        return self._raw_config.get(section, {}).get(key, default)
    
    def _parse_tushare(self) -> None:
        """解析 Tushare 配置"""
        sec = self._raw_config.get("tushare", {})
        self.tushare = TushareConfig(
            token=sec.get("token", ""),
            cache_dir=sec.get("cache_dir", "./data/cache"),
            retry_times=sec.get("retry_times", 3),
            retry_delay=sec.get("retry_delay", 1.0),
        )
    
    def _parse_backtest(self) -> None:
        """解析回测配置"""
        sec = self._raw_config.get("backtest", {})
        self.backtest = BacktestConfig(
            initial_cash=sec.get("initial_cash", 100000.0),
            strategy_name=sec.get("strategy_name", "default_strategy"),
            start_date=sec.get("start_date", "20200101"),
            end_date=sec.get("end_date", "20231231"),
            benchmark=sec.get("benchmark", "000300.SH"),
            commission_rate=sec.get("commission_rate", 0.0003),
            stamp_tax_rate=sec.get("stamp_tax_rate", 0.001),
            slippage=sec.get("slippage", 0.001),
            min_volume=sec.get("min_volume", 100),
            allow_short=sec.get("allow_short", False),
        )
    
    def _parse_strategy(self) -> None:
        """解析策略配置"""
        sec = self._raw_config.get("strategy", {})
        self.strategy = StrategyConfig(
            ma_short=sec.get("ma_short", 5),
            ma_long=sec.get("ma_long", 20),
            rsi_period=sec.get("rsi_period", 14),
            rsi_overbought=sec.get("rsi_overbought", 70.0),
            rsi_oversold=sec.get("rsi_oversold", 30.0),
            stop_loss=sec.get("stop_loss", 0.05),
            take_profit=sec.get("take_profit", 0.10),
            max_position=sec.get("max_position", 0.3),
            max_total_position=sec.get("max_total_position", 1.0),
        )
    
    def _parse_data(self) -> None:
        """解析数据配置"""
        sec = self._raw_config.get("data", {})
        fields = sec.get("fields", [
            "trade_date", "ts_code", "open", "high", "low", "close", "volume", "amount"
        ])
        pre = sec.get("preprocess", {})
        self.data = DataConfig(
            source=sec.get("source", "tushare"),
            data_type=sec.get("data_type", "daily"),
            fields=fields,
            fill_missing=pre.get("fill_missing", "ffill"),
            remove_outliers=pre.get("remove_outliers", True),
            outlier_std=pre.get("outlier_std", 5.0),
        )
    
    def _parse_output(self) -> None:
        """解析输出配置"""
        sec = self._raw_config.get("output", {})
        self.output = OutputConfig(
            results_dir=sec.get("results_dir", "./output/results"),
            plots_dir=sec.get("plots_dir", "./output/plots"),
            report_path=sec.get("report_path", "./output/backtest_report.html"),
            save_intermediate=sec.get("save_intermediate", True),
            log_level=sec.get("log_level", "INFO"),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置导出为字典"""
        return {
            "tushare": {
                "token": self.tushare.token,
                "cache_dir": self.tushare.cache_dir,
                "retry_times": self.tushare.retry_times,
                "retry_delay": self.tushare.retry_delay,
            },
            "backtest": {
                "initial_cash": self.backtest.initial_cash,
                "strategy_name": self.backtest.strategy_name,
                "start_date": self.backtest.start_date,
                "end_date": self.backtest.end_date,
                "benchmark": self.backtest.benchmark,
                "commission_rate": self.backtest.commission_rate,
                "stamp_tax_rate": self.backtest.stamp_tax_rate,
                "slippage": self.backtest.slippage,
                "min_volume": self.backtest.min_volume,
                "allow_short": self.backtest.allow_short,
            },
            "strategy": {
                "ma_short": self.strategy.ma_short,
                "ma_long": self.strategy.ma_long,
                "rsi_period": self.strategy.rsi_period,
                "rsi_overbought": self.strategy.rsi_overbought,
                "rsi_oversold": self.strategy.rsi_oversold,
                "stop_loss": self.strategy.stop_loss,
                "take_profit": self.strategy.take_profit,
                "max_position": self.strategy.max_position,
                "max_total_position": self.strategy.max_total_position,
            },
            "data": {
                "source": self.data.source,
                "data_type": self.data.data_type,
                "fields": self.data.fields,
                "fill_missing": self.data.fill_missing,
                "remove_outliers": self.data.remove_outliers,
                "outlier_std": self.data.outlier_std,
            },
            "output": {
                "results_dir": self.output.results_dir,
                "plots_dir": self.output.plots_dir,
                "report_path": self.output.report_path,
                "save_intermediate": self.output.save_intermediate,
                "log_level": self.output.log_level,
            },
        }
    
    def __repr__(self) -> str:
        return (
            f"Config(backtest={self.backtest.strategy_name}, "
            f"start={self.backtest.start_date}, end={self.backtest.end_date}, "
            f"cash={self.backtest.initial_cash})"
        )
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """
        便捷构造器：从 YAML 文件加载配置
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            Config 实例
        """
        return cls(config_path=config_path)


# ---------- 便捷函数 ----------

def load_config(config_path: Optional[str] = None) -> Config:
    """加载配置文件的快捷函数"""
    return Config(config_path=config_path)
