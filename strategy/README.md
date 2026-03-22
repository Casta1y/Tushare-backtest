# 策略层 (Strategy Layer)

## 概述

策略层负责因子计算、信号生成和多策略组合，是量化交易策略的核心引擎。

## 模块列表

### `factor_base.py`
- **FactorBase**: 所有因子的抽象基类
  - `calculate()`: 子类必须实现的因子计算方法
  - `validate_params()`: 参数验证
  - `get_name()`: 返回因子名称
  - 统一的 DataFrame 输入/输出格式

### `technical_factors.py`
- **技术因子类**：共 46 个因子

  使用方式：
  ```python
  from strategy.technical_factors import MAFactor, RSIFactor, MACDFactor
  
  # 创建因子实例
  ma = MAFactor({'period': 5})
  rsi = RSIFactor({'period': 14})
  
  # 计算因子值
  ma_value = ma.calculate(df)
  rsi_value = rsi.calculate(df)
  ```

  | 因子类 | 参数 | 说明 |
  |--------|------|------|
  | MAFactor | period | 简单移动平均 |
  | EMAFactor | period | 指数移动平均 |
  | MACDFactor | fast, slow, signal | MACD 指标 |
  | RSIFactor | period | 相对强弱指数 |
  | KDJFactor | period | KDJ 随机指标 |
  | BBandsFactor | period, std_dev | 布林带 |
  | ATRFactor | period | 平均真实波幅 |
  | ADXFactor | period | 趋向指标 |
  | ... | ... | 更多因子 |

### `fundamental_factors.py`
- **基本面因子类**：共 47 个因子

  使用方式：
  ```python
  from strategy.fundamental_factors import PEFactor, ROEFactor
  
  # 创建因子实例
  pe = PEFactor({})
  roe = ROEFactor({})
  
  # 计算因子值
  pe_value = pe.calculate(df)
  roe_value = roe.calculate(df)
  ```

  | 因子类 | 说明 |
  |--------|------|
  | PEFactor | 市盈率 |
  | PBFactor | 市净率 |
  | PSFactor | 市销率 |
  | ROEFactor | 净资产收益率 |
  | ROAFactor | 资产收益率 |
  | GrossMarginFactor | 毛利率 |
  | NetMarginFactor | 净利率 |
  | RevenueGrowthFactor | 营收增长率 |
  | ... | 更多因子 |
  | profit_growth | `profit_growth(data)` | 净利润增长率 |
  | asset_turnover | `asset_turnover(data)` | 资产周转率 |
  | debt_ratio | `debt_ratio(data)` | 资产负债率 |
  | current_ratio | `current_ratio(data)` | 流动比率 |
  | quick_ratio | `quick_ratio(data)` | 速动比率 |
  | cash_ratio | `cash_ratio(data)` | 现金比率 |
  | peg | `peg(data)` | PEG 比率 |
  | ... | ... | 更多因子持续补充 |

### `signal_generator.py`
- **SignalGenerator**: 交易信号生成器
  - `generate()`: 根据因子值和阈值生成交易信号
  - `get_signals()`: 批量获取信号
  - 输出格式：列 `signal`（1=买入, -1=卖出, 0=持有）

### `strategy_composer.py`
- **StrategyComposer**: 多策略组合器
  - `combine()`: 加权平均组合多个策略
  - `rank()`: 策略排序与选择
  - 支持因子等权、因子加权、IC 加权等多种组合方式

### `factor_library.py`
- **FactorLibrary**: 因子注册与管理
  - `register_factor()`: 注册自定义因子
  - `get_factor()`: 获取已注册因子
  - `list_factors()`: 列出所有因子
  - `factor_doc()`: 查看因子文档

## 信号格式约定

信号 DataFrame 包含以下列：

| 列名 | 类型 | 说明 |
|------|------|------|
| `ts_code` | str | 股票代码 |
| `trade_date` | str | 交易日期 |
| `signal` | int | 信号（1=买入, -1=卖出, 0=持有）|
| `strength` | float | 信号强度（0~1）|

## 使用示例

```python
from strategy.technical_factors import MAFactor, EMAFactor, RSIFactor, MACDFactor
from strategy.fundamental_factors import PEFactor, ROEFactor
from strategy.signal_generator import SignalGenerator
from strategy.strategy_composer import StrategyComposer
from strategy.factor_library import FactorLibrary
import pandas as pd

# 假设 df 是包含 OHLCV 数据的 DataFrame

# 计算技术因子
ma = MAFactor({'period': 5})
ma_result = ma.calculate(df)

rsi = RSIFactor({'period': 14})
rsi_result = rsi.calculate(df)

# 计算基本面因子
pe = PEFactor({})
pe_result = pe.calculate(df)

# 生成交易信号（基于简单金叉策略）
signals = pd.DataFrame(index=df.index)
signals['signal'] = 0
ma5 = ma_result
ma20 = MAFactor({'period': 20}).calculate(df)
signals.loc[ma5 > ma20, 'signal'] = 1
signals.loc[ma5 < ma20, 'signal'] = -1

# 因子库管理
library = FactorLibrary()
library.register_factor('MA5', ma)
```

## 与回测层的接口

```
SignalGenerator.generate() → pd.DataFrame → BacktestEngine.run()
```
