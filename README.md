# Tushare 量化回测框架

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Backtrader-3.0+-green.svg" alt="Backtrader">
  <img src="https://img.shields.io/badge/Tushare-API-orange.svg" alt="Tushare">
</p>

基于 Tushare 数据接口和 Backtrader 回测框架的量化交易回测系统，支持多因子策略、技术指标、回测分析和可视化报告。

---

## 📋 功能特性

| 功能 | 说明 |
|------|------|
| **数据获取** | Tushare API 封装，支持日线、分钟线、财务数据 |
| **数据清洗** | 缺失值处理、异常值检测、数据类型转换 |
| **缓存管理** | 内存+磁盘二级缓存，LRU 淘汰，TTL 过期 |
| **因子库** | 46 个技术因子 + 47 个基本面因子 |
| **信号生成** | 阈值、交叉、百分位、自定义信号 |
| **回测引擎** | Backtrader 集成，支持滑点、佣金设置 |
| **性能分析** | 收益率、夏普比率、最大回撤、胜率等 |
| **风险分析** | 波动率、Beta、Sortino、VaR/CVaR |
| **可视化** | 收益率曲线、回撤图、月度热力图 |
| **报告生成** | HTML/JSON/Markdown 格式报告 |

---

## 🏗️ 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                        示例层 (Examples)                      │
│  example_1_simple_ma.py  example_2_factor.py  example_3_multi.py │
└──────────────────────────┬──────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   数据层 (Data)  │ │  策略层 (Strategy)│ │  回测层 (Backtest)│
├─────────────────┤ ├─────────────────┤ ├─────────────────┤
│ tushare_client  │ │ factor_base     │ │ backtest_engine │
│ data_fetcher    │ │ technical_factor │ │ backtest_strategy│
│ data_cleaner    │ │ fundamental_fact │ │ performance_ana │
│ data_cache      │ │ signal_generator │ │ risk_analyzer   │
│ data_storage    │ │ strategy_composer│ │ result_visualiz │
│                 │ │ factor_library   │ │ backtest_report │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              ▼
                    ┌─────────────────┐
                    │   输出结果      │
                    │ results/charts │
                    └─────────────────┘
```

---

## 📁 项目结构

```
quant_backtest/
├── src/                           # 源代码
│   ├── data/                     # 数据层
│   │   ├── tushare_client.py    # Tushare API 封装
│   │   ├── data_fetcher.py      # 数据获取
│   │   ├── data_cleaner.py      # 数据清洗
│   │   ├── data_cache.py        # 缓存管理
│   │   └── data_storage.py      # 数据存储
│   ├── strategy/                 # 策略层
│   │   ├── factor_base.py       # 因子基类
│   │   ├── technical_factors.py # 技术因子 (46个)
│   │   ├── fundamental_factors.py # 基本面因子 (47个)
│   │   ├── signal_generator.py # 信号生成
│   │   ├── strategy_composer.py # 策略组合
│   │   └── factor_library.py    # 因子库管理
│   ├── backtest/                 # 回测层
│   │   ├── backtest_engine.py   # 回测引擎
│   │   ├── backtest_strategy.py # Backtrader 适配
│   │   ├── performance_analyzer.py # 性能分析
│   │   ├── risk_analyzer.py    # 风险分析
│   │   ├── result_visualizer.py # 可视化
│   │   └── backtest_report.py   # 报告生成
│   └── examples/                 # 示例代码
│       ├── example_1_simple_ma.py      # 双均线策略
│       ├── example_2_factor_strategy.py # 多因子策略
│       ├── example_3_multi_strategy.py  # 多策略组合
│       └── visualize_results.py          # 可视化脚本
├── config/
│   └── config.yaml              # 配置文件
└── README.md                    # 本文件
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install tushare backtrader pandas numpy matplotlib pyyaml jinja2
```

### 2. 配置 Tushare Token

编辑 `config/config.yaml`：

```yaml
tushare:
  token: "your_token_here"

backtest:
  initial_cash: 100000
  commission: 0.0003
  slippage: 0.001
```

### 3. 运行示例

```bash
cd src/examples/

# 示例1：双均线策略
python3 example_1_simple_ma.py

# 示例2：多因子选股策略
python3 example_2_factor_strategy.py

# 示例3：多策略组合
python3 example_3_multi_strategy.py

# 示例4：因子库完整示例（使用策略层因子）
python3 example_4_factor_library.py

# 示例5：完整回测示例（覆盖所有回测层模块）
python3 example_5_full_backtest.py

# 生成可视化图表
python3 visualize_results.py
```

### 4. API 使用方式

**统一设计**：所有分析器支持两种初始化方式

```python
# 方式1：初始化时传参
analyzer = PerformanceAnalyzer(equity_curve=equity)
returns = analyzer.calculate_returns()

# 方式2：后续设置数据
analyzer = PerformanceAnalyzer()
analyzer.set_data(equity)
returns = analyzer.calculate_returns()
```

---

## 📊 示例说明

| 示例 | 文件 | 策略说明 |
|------|------|----------|
| 1 | `example_1_simple_ma.py` | 双均线策略（MA5/MA20 金叉买，死叉卖） |
| 2 | `example_2_factor_strategy.py` | 多因子选股（PE、ROE、动量等因子综合评分） |
| 3 | `example_3_multi_strategy.py` | 多策略组合（MA + RSI + MACD 组合） |
| 4 | `example_4_factor_library.py` | 因子库完整示例（使用策略层 46+ 技术因子 + 47 基本面因子） |
| 5 | `example_5_full_backtest.py` | 完整回测示例（覆盖 PerformanceAnalyzer、RiskAnalyzer、ResultVisualizer、BacktestReport、DataStorage） |

---

## 📈 因子库

### 技术因子 (46个)

| 类别 | 因子数量 | 代表因子 |
|------|----------|----------|
| 移动平均类 | 10 | MA, EMA, SMA, WMA, VWMA, HMA, ALMA, DMA, TMA, VWAP |
| 趋势类 | 8 | MACD, ADX, AROON, ICHIMOKU, PLUS_DI, MINUS_DI |
| 动量类 | 11 | RSI, KDJ, ROC, CCI, Williams%R, Stochastic, CMO |
| 波动率类 | 8 | ATR, BBands, Keltner, Donchian, STD, HistVolatility |
| 成交量类 | 9 | OBV, VROC, CMF, MFI, VOL_MA, VOL_RATIO, TurnoverRate |

### 基本面因子 (47个)

| 类别 | 因子数量 | 代表因子 |
|------|----------|----------|
| 估值类 | 8 | PE, PB, PS, PCF, EV_EBITDA, PEG, PE_TTM, DividendYield |
| 盈利能力 | 8 | ROE, ROA, ROIC, GrossMargin, NetMargin, OperatingMargin |
| 成长类 | 8 | ProfitGrowth, RevenueGrowth, EPSGrowth, CAGR3Y |
| 财务结构 | 6 | DebtToAsset, DebtToEquity, CurrentRatio, QuickRatio |
| 运营效率 | 6 | InventoryTurnover, ReceivableTurnover, AssetTurnover |
| 综合因子 | 2 | AltmanZScore, PiotroskiFScore |

---

## 📂 输出结果

### 回测数据
```
src/output/results/
├── ma_signals.csv           # 双均线策略信号
├── factor_scores.csv         # 多因子评分
├── factor_equity.csv         # 多因子权益曲线
└── multi_strategy_equity.csv # 多策略组合权益
```

### 可视化图表
```
src/output/charts/
├── example1_ma_strategy.png           # 双均线策略图
├── example2_factor_strategy.png       # 多因子选股图
├── example3_multi_strategy.png        # 多策略组合图
├── comprehensive_comparison.png       # 综合对比图
├── drawdown_comparison.png            # 回撤对比图
└── report.md                          # Markdown 报告
```

---

## 🔧 核心模块

### 数据层 (Data Layer)

| 模块 | 功能 |
|------|------|
| `tushare_client.py` | Tushare API 封装，11 个 API 方法，含重试/日志/错误处理 |
| `data_fetcher.py` | 数据获取接口，支持日线/分钟线/财务数据，含批量获取、数据验证 |
| `data_cleaner.py` | 数据清洗，支持 6 种缺失值填充方法、异常值检测 |
| `data_cache.py` | 内存+磁盘双缓存、TTL 过期、LRU 淘汰、统计信息 |
| `data_storage.py` | CSV/Parquet 读写、压缩支持、分区存储 |

### 策略层 (Strategy Layer)

| 模块 | 功能 |
|------|------|
| `factor_base.py` | 因子基类，支持参数验证、DataFrame/Series 自动处理 |
| `technical_factors.py` | 46 个技术因子实现 |
| `fundamental_factors.py` | 47 个基本面因子实现 |
| `signal_generator.py` | 阈值/交叉/百分位/自定义四种信号生成方式 |
| `strategy_composer.py` | 6 种策略组合方法（等权/加权/排序/最大/最小/乘法） |
| `factor_library.py` | 因子注册、查询、分类、文档生成 |

### 回测层 (Backtest Layer)

| 模块 | 功能 |
|------|------|
| `backtest_engine.py` | 回测引擎，支持初始资金、佣金/印花税/滑点配置 |
| `backtest_strategy.py` | Backtrader 策略适配，提供 7 个内置策略 |
| `performance_analyzer.py` | 收益率、夏普比率、最大回撤、年化收益、胜率 |
| `risk_analyzer.py` | 波动率、Beta、Sortino、VaR/CVaR、信息比率 |
| `result_visualizer.py` | 收益率曲线、回撤曲线、分布图、月度热力图 |
| `backtest_report.py` | HTML/JSON/Markdown 报告生成 |

### API 使用示例

```python
from backtest import PerformanceAnalyzer, RiskAnalyzer, ResultVisualizer
from data import DataStorage
import pandas as pd

# 方式1：初始化时传参
analyzer = PerformanceAnalyzer(equity_curve=equity_df)

# 方式2：后续设置数据
analyzer = PerformanceAnalyzer()
analyzer.set_data(equity_df)

# 计算指标
returns = analyzer.calculate_returns()
sharpe = analyzer.calculate_sharpe_ratio()
max_dd = analyzer.calculate_max_drawdown()

# 保存结果
storage = DataStorage(base_dir="./output")
storage.save_to_csv(equity_df, "equity.csv")
```

---

## 📝 自定义策略

### 添加自定义因子

```python
def calculate_factors(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    
    # 示例：添加 RSI 因子
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    
    return df
```

### 添加因子权重

```python
factor_weights = {
    "rsi": (1, 0.10),  # (方向, 权重) 1=越大越好
    "pe": (-1, 0.15),  # -1=越小越好
}
```

---

## 📄 许可证

MIT License

---

**版本**: 1.0.0  
**日期**: 2026-03-22
