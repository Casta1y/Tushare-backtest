# 数据层 (Data Layer)

## 概述

数据层负责从 Tushare 获取、处理、缓存和存储市场数据，是量化回测框架的基础数据来源。

## 模块列表

### `tushare_client.py`
- **TushareClient**: Tushare API 客户端封装类
  - 接收 API Token 初始化
  - 统一封装 `pro_bar`、`daily`、`financial_data` 等常用接口
  - 内置重试机制和错误处理
  - 日志记录请求和响应

### `data_fetcher.py`
- **DataFetcher**: 数据获取接口
  - `get_daily()`: 获取日线数据
  - `get_minute_data()`: 获取分钟级数据
  - `get_financial_data()`: 获取财务数据
  - `fetch_all_stocks()`: 批量获取全市场股票列表
  - 数据完整性自动校验

### `data_cleaner.py`
- **DataCleaner**: 数据清洗工具
  - `clean_data()`: 统一清洗入口
  - 缺失值处理（向前填充、向后填充、均值填充、删除）
  - 异常值检测与处理（3σ 法则、IQR 法则）
  - 数据类型自动转换
  - 新股/停牌日数据过滤

### `data_cache.py`
- **DataCache**: 内存 + 磁盘二级缓存
  - `save()`: 写入缓存
  - `load()`: 读取缓存
  - TTL 过期机制
  - 缓存大小限制（LRU 淘汰）
  - 序列化方式支持（pickle / parquet）

### `data_storage.py`
- **DataStorage**: 持久化存储管理器
  - `save_to_csv()` / `save_to_parquet()`: 存储数据
  - `load_from_csv()` / `load_from_parquet()`: 读取数据
  - 目录自动创建
  - 数据压缩支持

## 使用示例

```python
from data import DataStorage
import pandas as pd

# 初始化存储管理器
storage = DataStorage(base_dir="./output")

# 保存数据为 CSV
storage.save_to_csv(df, "results/data.csv")

# 保存数据为 Parquet（需要 pyarrow）
storage.save_to_parquet(df, "results/data.parquet")

# 读取数据
df = storage.load_from_csv("results/data.csv")
```

## 数据字段约定

所有 DataFrame 均使用以下标准列名：

| 字段 | 类型 | 说明 |
|------|------|------|
| `ts_code` | str | 股票代码 |
| `trade_date` | str | 交易日期（YYYYMMDD） |
| `open` | float | 开盘价 |
| `high` | float | 最高价 |
| `low` | float | 最低价 |
| `close` | float | 收盘价 |
| `vol` | float | 成交量 |
| `amount` | float | 成交额 |

## 配置

在 `config/config.yaml` 中配置：

```yaml
data:
  tushare_token: "your_token_here"
  cache_dir: "./data/cache"
  storage_dir: "./data/storage"
  cache_ttl_days: 7
  retry_times: 3
```

## 与策略层的接口

```
DataFetcher.get_daily() → pd.DataFrame → FactorBase.calculate() → SignalGenerator
```
