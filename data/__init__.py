# -*- coding: utf-8 -*-
"""
数据层模块 (Data Layer)

提供 Tushare 数据接口封装、数据清洗、缓存和存储功能。

主要模块
--------
tushare_client : Tushare API 客户端封装
data_fetcher   : 数据获取接口
data_cleaner   : 数据清洗和预处理
data_cache     : 数据缓存管理
data_storage   : 数据持久化存储（CSV/Parquet）

使用示例
--------
>>> from src.data import TushareClient, DataFetcher
>>> client = TushareClient(token="your_token")
>>> fetcher = DataFetcher(client)
>>> df = fetcher.get_daily("000001.SZ", "20230101", "20231231")
"""

__version__ = "1.0.0"

from .tushare_client import TushareClient
from .data_fetcher import DataFetcher, DataValidationError
from .data_cleaner import DataCleaner
from .data_cache import DataCache
from .data_storage import DataStorage

__all__ = [
    "TushareClient",
    "DataFetcher",
    "DataValidationError",
    "DataCleaner",
    "DataCache",
    "DataStorage",
]
