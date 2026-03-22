# -*- coding: utf-8 -*-
"""
数据获取模块

提供统一的数据获取接口，支持日线、分钟线、财务数据获取和数据验证。
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union

import pandas as pd
import numpy as np

from .tushare_client import TushareClient
from .data_cache import DataCache

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """数据验证错误异常。"""
    pass


class DataFetcher:
    """
    数据获取器

    封装TushareClient，提供更高层次的数据获取接口，
    包含数据验证、批量获取、自动重试等功能。

    Parameters
    ----------
    tushare_client : TushareClient
        Tushare客户端实例。
    validate : bool, optional
        是否启用数据验证，默认True。
    batch_size : int, optional
        批量获取时的批次大小，默认50。

    Examples
    --------
    >>> client = TushareClient(token="your_token")
    >>> fetcher = DataFetcher(client)
    >>> df = fetcher.get_daily_data("000001.SZ", "20230101", "20231231")
    """

    # 数据字段要求定义
    REQUIRED_DAILY_FIELDS = ["ts_code", "trade_date", "open", "high", "low", "close", "vol"]
    REQUIRED_MINUTE_FIELDS = ["ts_code", "trade_time", "open", "high", "low", "close", "vol"]
    REQUIRED_FINANCIAL_FIELDS = ["ts_code", "ann_date", "report_type"]

    def __init__(
        self,
        tushare_client: TushareClient,
        validate: bool = True,
        batch_size: int = 50,
        cache_dir: str = "./data/cache",
        cache_ttl: int = 86400,
        enable_cache: bool = True,
    ):
        if not isinstance(tushare_client, TushareClient):
            raise TypeError("tushare_client must be a TushareClient instance.")

        self.client = tushare_client
        self.validate = validate
        self.batch_size = batch_size
        
        # 缓存配置
        self.enable_cache = enable_cache
        if self.enable_cache:
            self.cache = DataCache(
                cache_dir=cache_dir,
                default_ttl=cache_ttl,  # 默认24小时过期
            )
            logger.info(f"DataFetcher cache enabled: dir={cache_dir}, ttl={cache_ttl}s")
        else:
            self.cache = None
            logger.info("DataFetcher cache disabled")

    def get_daily_data(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        adj: str = "qfq",
        asset: str = "E",
        fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        获取日线数据。

        Parameters
        ----------
        ts_code : str
            股票代码，如 "000001.SZ"。
        start_date : str
            开始日期，格式 YYYYMMDD。
        end_date : str
            结束日期，格式 YYYYMMDD。
        adj : str, optional
            复权类型。qfq=前复权（默认），hfq=后复权，None=不复权。
        asset : str, optional
            资产类型。E=股票（默认），I=指数。
        fields : list, optional
            指定返回字段，为None时返回所有字段。

        Returns
        -------
        pd.DataFrame
            日线数据，包含字段：ts_code, trade_date, open, high, low, close, vol, amount。

        Raises
        ------
        DataValidationError
            数据验证失败时抛出。
        ValueError
            参数格式错误时抛出。
        """
        # 参数验证
        self._validate_date_range(start_date, end_date)
        self._validate_ts_code(ts_code)

        # 构建基础缓存键（不包含日期范围）
        cache_key = f"daily_{ts_code}_{adj}_{asset}"

        # 检查缓存中是否有完整或部分数据
        cached_data = None
        if self.enable_cache and self.cache:
            cached_data = self.cache.load(cache_key)
            if cached_data is not None:
                logger.info(f"Cache hit for daily data base: {ts_code}")

        logger.info(f"Fetching daily data for {ts_code} from {start_date} to {end_date}")

        try:
            # 智能合并：检查缓存中已有的数据
            cached_range_data = pd.DataFrame()
            if cached_data is not None:
                # 过滤出请求范围内的缓存数据（用于返回给用户）
                cached_range_data = self._merge_cached_data(cached_data, start_date, end_date)
                if len(cached_range_data) == len(cached_data) and len(cached_range_data) > 0:
                    # 缓存数据完全匹配请求范围
                    logger.info(f"Cache provides complete data for {start_date}~{end_date}")
                    # 返回筛选后的数据，但不需要更新缓存
                    return cached_range_data

            # 计算需要请求的缺失日期范围
            missing_start, missing_end = self._get_missing_date_range(cached_data, start_date, end_date)

            if missing_start and missing_end:
                logger.info(f"Requesting missing data: {missing_start} ~ {missing_end}")

                # 请求缺失的数据
                missing_df = self.client.daily(
                    ts_code=ts_code,
                    start_date=missing_start,
                    end_date=missing_end,
                    asset=asset,
                    adj=adj,
                )

                # 合并数据（用原始缓存数据 + 新数据）
                if cached_data is not None and not cached_data.empty:
                    # 合并：原始缓存 + 新数据
                    df = pd.concat([cached_data, missing_df], ignore_index=True)
                    # 按日期排序并去重
                    df = df.sort_values('trade_date').drop_duplicates(subset=['trade_date']).reset_index(drop=True)
                else:
                    df = missing_df
            else:
                logger.info(f"No missing data, using cached data: {start_date}~{end_date}")
                df = cached_range_data if not cached_range_data.empty else cached_data

            # 字段筛选
            if fields:
                available_fields = [f for f in fields if f in df.columns]
                df = df[available_fields]

            # 数据验证
            if self.validate:
                self._validate_daily_data(df)

            # 数据后处理
            df = self._post_process_daily(df)

            # 更新缓存（合并后）
            if self.enable_cache and self.cache:
                if cached_data is not None:
                    # 合并缓存
                    self._update_cached_data(cache_key, df)
                else:
                    # 首次缓存
                    self.cache.save(cache_key, df)
                    logger.info(f"Cache saved for daily data: {ts_code}")

            logger.info(f"Successfully fetched {len(df)} daily records for {ts_code}")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch daily data: {e}")
            raise

    def get_minute_data(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        freq: str = "60",
        adj: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        获取分钟级行情数据。

        Parameters
        ----------
        ts_code : str
            股票代码，如 "000001.SZ"。
        start_date : str
            开始时间，格式 YYYYMMDD HH:MM:SS。
        end_date : str
            结束时间，格式 YYYYMMDD HH:MM:SS。
        freq : str, optional
            频率。5/15/30/60（分钟），默认 60。
        adj : str, optional
            复权类型。默认None（不复权）。

        Returns
        -------
        pd.DataFrame
            分钟级数据，包含字段：ts_code, trade_time, open, high, low, close, vol。

        Raises
        ------
        DataValidationError
            数据验证失败时抛出。
        ValueError
            参数格式错误时抛出。
        """
        # 参数验证
        if freq not in ["5", "15", "30", "60"]:
            raise ValueError(f"Invalid freq: {freq}. Must be one of ['5', '15', '30', '60']")
        self._validate_ts_code(ts_code)

        logger.info(f"Fetching {freq}min data for {ts_code} from {start_date} to {end_date}")

        try:
            # 获取数据
            df = self.client.minute(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                freq=freq,
            )

            # 数据验证
            if self.validate:
                self._validate_minute_data(df)

            # 数据后处理
            df = self._post_process_minute(df)

            logger.info(f"Successfully fetched {len(df)} minute records for {ts_code}")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch minute data: {e}")
            raise

    def get_financial_data(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        report_type: str = "1",
    ) -> pd.DataFrame:
        """
        获取财务数据。

        Parameters
        ----------
        ts_code : str
            股票代码，如 "000001.SZ"。
        start_date : str
            开始报告期，格式 YYYYMMDD。
        end_date : str
            结束报告期，格式 YYYYMMDD。
        report_type : str, optional
            报告类型。1=合并报告（默认），2=单季合并。

        Returns
        -------
        pd.DataFrame
            财务数据，包含字段：ts_code, ann_date, f_ann_date, report_type, 
            total_revenue, net_profit, roe, roa 等。

        Raises
        ------
        DataValidationError
            数据验证失败时抛出。
        """
        self._validate_ts_code(ts_code)
        self._validate_date_range(start_date, end_date)

        logger.info(f"Fetching financial data for {ts_code} from {start_date} to {end_date}")

        try:
            # 获取数据
            df = self.client.financial_data(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
            )

            # 筛选报告类型
            if report_type and "report_type" in df.columns:
                df = df[df["report_type"].astype(str) == report_type]

            # 数据验证
            if self.validate:
                self._validate_financial_data(df)

            # 数据后处理
            df = self._post_process_financial(df)

            logger.info(f"Successfully fetched {len(df)} financial records for {ts_code}")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch financial data: {e}")
            raise

    def get_batch_daily_data(
        self,
        ts_codes: List[str],
        start_date: str,
        end_date: str,
        adj: str = "qfq",
    ) -> Dict[str, pd.DataFrame]:
        """
        批量获取多只股票的日线数据。

        Parameters
        ----------
        ts_codes : list
            股票代码列表。
        start_date : str
            开始日期。
        end_date : str
            结束日期。
        adj : str, optional
            复权类型。

        Returns
        -------
        dict
            股票代码到DataFrame的映射。

        Examples
        --------
        >>> fetcher = DataFetcher(client)
        >>> data = fetcher.get_batch_daily_data(
        ...     ["000001.SZ", "000002.SZ"],
        ...     "20230101",
        ...     "20231231"
        ... )
        """
        results = {}
        total = len(ts_codes)

        for idx, code in enumerate(ts_codes, 1):
            try:
                df = self.get_daily_data(code, start_date, end_date, adj)
                results[code] = df
                logger.info(f"[{idx}/{total}] Fetched {code}: {len(df)} records")
            except Exception as e:
                logger.warning(f"[{idx}/{total}] Failed to fetch {code}: {e}")
                results[code] = pd.DataFrame()

        return results

    def get_index_daily(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        获取指数日线数据。

        Parameters
        ----------
        ts_code : str
            指数代码，如 "000001.SH"（上证指数）。
        start_date : str
            开始日期。
        end_date : str
            结束日期。

        Returns
        -------
        pd.DataFrame
            指数日线数据。
        """
        self._validate_date_range(start_date, end_date)

        logger.info(f"Fetching index daily for {ts_code} from {start_date} to {end_date}")

        try:
            df = self.client.index_daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
            )

            if self.validate:
                self._validate_daily_data(df)

            df = self._post_process_daily(df)

            logger.info(f"Successfully fetched {len(df)} index records for {ts_code}")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch index daily: {e}")
            raise

    # ─────────────────────────────────────────────
    # 数据验证方法
    # ─────────────────────────────────────────────
    # 缓存智能合并辅助方法
    # ─────────────────────────────────────────────

    def _merge_cached_data(self, cached_data: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从缓存数据中筛选出请求范围内的数据。

        Parameters
        ----------
        cached_data : pd.DataFrame
            缓存的数据。
        start_date : str
            请求的开始日期。
        end_date : str
            请求的结束日期。

        Returns
        -------
        pd.DataFrame
            在请求范围内的数据。
        """
        if cached_data is None or cached_data.empty:
            return pd.DataFrame()

        if 'trade_date' not in cached_data.columns:
            return pd.DataFrame()

        # 将 trade_date 转为字符串以便比较
        cached_data = cached_data.copy()
        cached_data['trade_date'] = cached_data['trade_date'].astype(str)

        # 筛选日期范围内的数据
        mask = (cached_data['trade_date'] >= start_date) & (cached_data['trade_date'] <= end_date)
        return cached_data[mask]

    def _get_missing_date_range(self, existing_df: pd.DataFrame, start_date: str, end_date: str) -> tuple:
        """
        计算缺失的日期范围。

        Parameters
        ----------
        existing_df : pd.DataFrame
            已有的数据。
        start_date : str
            请求的开始日期。
        end_date : str
            请求的结束日期。

        Returns
        -------
        tuple
            (missing_start, missing_end) 如果有缺失数据，否则返回 (None, None)。
        """
        if existing_df is None or existing_df.empty:
            return start_date, end_date

        if 'trade_date' not in existing_df.columns:
            return start_date, end_date

        # 获取已有数据的日期范围
        existing_dates = set(existing_df['trade_date'].astype(str).tolist())

        # 获取已有数据的实际范围（用于判断是否需要请求）
        existing_start = min(existing_dates)
        existing_end = max(existing_dates)

        # 如果请求范围完全在已有数据范围内，不需要请求
        if start_date >= existing_start and end_date <= existing_end:
            # 进一步检查：请求范围内的所有交易日是否都在缓存中
            existing_in_range = set(d for d in existing_dates if start_date <= d <= end_date)
            if len(existing_in_range) > 0:
                # 有数据在请求范围内，不需要再请求
                return None, None

        # 如果请求范围完全不在已有数据范围内，需要请求整个范围
        if end_date < existing_start or start_date > existing_end:
            return start_date, end_date

        # 请求范围与已有数据有重叠
        # 需要请求已有数据之前的部分
        if start_date < existing_start:
            return start_date, existing_start
        # 需要请求已有数据之后的部分
        if end_date > existing_end:
            return existing_end, end_date

        return None, None

    def _update_cached_data(self, cache_key: str, new_df: pd.DataFrame) -> None:
        """
        更新缓存数据。

        Parameters
        ----------
        cache_key : str
            缓存键。
        new_df : pd.DataFrame
            新的数据（包含所有需要缓存的数据）。
        """
        if new_df is None or new_df.empty:
            return

        self.cache.save(cache_key, new_df)
        logger.info(f"Cache updated for key: {cache_key}")

    # ─────────────────────────────────────────────

    def _validate_date_range(self, start_date: str, end_date: str) -> None:
        """验证日期范围格式。"""
        try:
            start = datetime.strptime(start_date, "%Y%m%d")
            end = datetime.strptime(end_date, "%Y%m%d")
            if start > end:
                raise ValueError(f"start_date ({start_date}) must be <= end_date ({end_date})")
        except ValueError as e:
            logger.error(f"Invalid date format: {e}")
            raise ValueError(f"Date must be in YYYYMMDD format: {e}")

    def _validate_ts_code(self, ts_code: str) -> None:
        """验证股票代码格式。"""
        if not ts_code or not isinstance(ts_code, str):
            raise ValueError("ts_code must be a non-empty string")
        if not (ts_code.endswith(".SH") or ts_code.endswith(".SZ")):
            raise ValueError(f"Invalid ts_code format: {ts_code}. Must end with .SH or .SZ")

    def _validate_daily_data(self, df: pd.DataFrame) -> None:
        """
        验证日线数据完整性。

        Raises
        ------
        DataValidationError
            验证失败时抛出。
        """
        if df is None or df.empty:
            raise DataValidationError("DataFrame is empty")

        # 检查必需字段
        missing_fields = [f for f in self.REQUIRED_DAILY_FIELDS if f not in df.columns]
        if missing_fields:
            raise DataValidationError(f"Missing required fields: {missing_fields}")

        # 检查关键字段的空值
        if df["close"].isna().any():
            logger.warning("Found null values in 'close' column")

        # 检查价格合理性
        if (df["high"] < df["low"]).any():
            raise DataValidationError("high < low found in data (invalid prices)")

        if (df["open"] <= 0).any():
            logger.warning("Found non-positive open prices")

    def _validate_minute_data(self, df: pd.DataFrame) -> None:
        """验证分钟数据完整性。"""
        if df is None or df.empty:
            raise DataValidationError("DataFrame is empty")

        missing_fields = [f for f in self.REQUIRED_MINUTE_FIELDS if f not in df.columns]
        if missing_fields:
            raise DataValidationError(f"Missing required fields: {missing_fields}")

    def _validate_financial_data(self, df: pd.DataFrame) -> None:
        """验证财务数据完整性。"""
        if df is None or df.empty:
            raise DataValidationError("DataFrame is empty")

        missing_fields = [f for f in self.REQUIRED_FINANCIAL_FIELDS if f not in df.columns]
        if missing_fields:
            raise DataValidationError(f"Missing required fields: {missing_fields}")

    # ─────────────────────────────────────────────
    # 数据后处理方法
    # ─────────────────────────────────────────────

    def _post_process_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """日线数据后处理。"""
        if df.empty:
            return df

        # 按日期排序
        if "trade_date" in df.columns:
            df = df.sort_values("trade_date").reset_index(drop=True)

        # 确保日期格式正确
        if "trade_date" in df.columns and df["trade_date"].dtype == "int64":
            df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str), format="%Y%m%d")

        return df

    def _post_process_minute(self, df: pd.DataFrame) -> pd.DataFrame:
        """分钟数据后处理。"""
        if df.empty:
            return df

        # 按时间排序
        if "trade_time" in df.columns:
            df = df.sort_values("trade_time").reset_index(drop=True)

        return df

    def _post_process_financial(self, df: pd.DataFrame) -> pd.DataFrame:
        """财务数据后处理。"""
        if df.empty:
            return df

        # 按公告日期排序
        if "ann_date" in df.columns:
            df = df.sort_values("ann_date").reset_index(drop=True)

        return df

    def __repr__(self) -> str:
        return f"DataFetcher(validate={self.validate}, batch_size={self.batch_size})"
