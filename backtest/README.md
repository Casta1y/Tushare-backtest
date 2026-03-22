# Backtest Module

本模块提供完整的回测框架，基于 Backtrader 实现。

## 核心文件

| 文件 | 功能 |
|------|------|
| `backtest_config.py` | 配置管理：Config 类负责加载 YAML 配置、参数验证 |
| `backtest_engine.py` | 回测引擎：BacktestEngine 核心执行回测 |
| `backtest_strategy.py` | 策略适配：继承 Backtrader 策略，适配自定义信号 |
| `performance_analyzer.py` | 性能分析：收益率、夏普比、最大回撤等 |
| `risk_analyzer.py` | 风险分析：波动率、Beta、索提诺比率等 |
| `result_visualizer.py` | 结果可视化：收益曲线、回撤图等 |
| `backtest_report.py` | 报告生成：输出 HTML 格式回测报告 |

## 快速使用

```python
from backtest import BacktestConfig, BacktestEngine

# 加载配置
config = BacktestConfig.from_yaml("config/config.yaml")

# 创建回测引擎
engine = BacktestEngine(config)

# 执行回测
results = engine.run(strategy=my_strategy, data=data)

# 获取结果
returns = results["returns"]
metrics = results["metrics"]
```

## 依赖

- `backtrader` - 回测框架
- `pandas` - 数据处理
- `matplotlib` - 可视化
- `jinja2` - 报告模板
- `pyyaml` - YAML 配置解析
