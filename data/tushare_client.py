# -*- coding: utf-8 -*-
"""
Tushare API 客户端封装

提供统一的 Tushare Pro API 调用接口，支持重试、日志记录和错误处理。
"""

import logging
import time
from typing import Optional, Dict, Any, List

import pandas as pd
import tushare as ts

logger = logging.getLogger(__name__)


class TushareClient:
    """
    Tushare Pro API 客户端封装类。

    Parameters
    ----------
    token : str
        Tushare Pro API Token。
    max_retry : int, optional
        失败重试次数，默认 3 次。
    retry_delay : float, optional
        重试间隔（秒），默认 1.0 秒。
    timeout : int, optional
        单次请求超时时间（秒），默认 30 秒。

    Attributes
    ----------
    pro : tushare.pro.Api
        Tushare Pro API 实例。
    max_retry : int
        最大重试次数。
    retry_delay : float
        重试延迟。
    timeout : int
        请求超时。

    Examples
    --------
    >>> client = TushareClient(token="your_token_here")
    >>> df = client.daily("000001.SZ", "20230101", "20231231")
    >>> df = client.pro_bar("000001.SZ", "20230101", "20231231")
    """

    def __init__(
        self,
        token: str,
        max_retry: int = 3,
        retry_delay: float = 1.0,
        timeout: int = 30,
    ):
        if not token or not isinstance(token, str):
            raise ValueError("Tushare token must be a non-empty string.")

        self.token = token
        self.max_retry = max_retry
        self.retry_delay = retry_delay
        self.timeout = timeout

        # 初始化 Tushare Pro API
        self._api: Optional[Any] = None
        self._init_api()

    def _init_api(self) -> None:
        """初始化 Tushare Pro API 实例。"""
        try:
            ts.set_token(self.token)
            self._api = ts.pro_api()
            logger.info("Tushare API initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Tushare API: {e}")
            raise ConnectionError(f"Tushare API initialization failed: {e}")

    def _call_with_retry(self, func, *args, **kwargs) -> Any:
        """
        带重试机制的 API 调用。

        Parameters
        ----------
        func : callable
            要调用的 API 方法。
        *args : positional arguments
            位置参数。
        **kwargs : keyword arguments
            关键字参数。

        Returns
        -------
        DataFrame or dict
            API 返回结果。

        Raises
        ------
        RuntimeError
            超过最大重试次数后抛出。
        """
        last_error = None
        for attempt in range(1, self.max_retry + 1):
            try:
                result = func(*args, **kwargs)
                if attempt > 1:
                    logger.info(f"Retry succeeded on attempt {attempt}.")
                return result
            except Exception as e:
                last_error = e
                logger.warning(
                    f"API call failed (attempt {attempt}/{self.max_retry}): {e}"
                )
                if attempt < self.max_retry:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(
                        f"API call failed after {self.max_retry} attempts."
                    )
        raise RuntimeError(
            f"API call failed after {self.max_retry} attempts. "
            f"Last error: {last_error}"
        )

    # ─────────────────────────────────────────────
    # 股票日线/周线/月线
    # ─────────────────────────────────────────────

    def daily(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        asset: str = "E",
        adj: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        获取股票日线数据（前复权/后复权）。

        Parameters
        ----------
        ts_code : str
            股票代码，如 "000001.SZ"。
        start_date : str
            开始日期，格式 YYYYMMDD，如 "20230101"。
        end_date : str
            结束日期，格式 YYYYMMDD，如 "20231231"。
        asset : str, optional
            资产类型。E=股票（默认），I=指数。
        adj : str, optional
            复权类型。None=不复权，qfq=前复权，hfq=后复权。

        Returns
        -------
        pd.DataFrame
            包含字段：ts_code, trade_date, open, high, low, close, vol, amount。
        """
        # 直接使用 daily 接口，不使用 pro_bar
        # 注意：Tushare 的 daily 接口返回的是未复权数据
        # 如果需要复权数据，可以使用 ts.pro_bar() 或自行计算
        return self._call_with_retry(
            self._api.daily,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
        )

    def pro_bar(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        asset: str = "E",
        adj: str = "qfq",
        freq: str = "D",
    ) -> pd.DataFrame:
        """
        获取行情数据（K 线），支持分钟级数据。

        Parameters
        ----------
        ts_code : str
            股票代码。
        start_date : str
            开始日期/时间。
        end_date : str
            结束日期/时间。
        asset : str, optional
            资产类型。E=股票（默认），I=指数。
        adj : str, optional
            复权类型。qfq=前复权（默认），hfq=后复权，None=不复权。
        freq : str, optional
            K 线频率。D=日线（默认），W=周线，M=月线，
            5/15/30/60=分钟级。

        Returns
        -------
        pd.DataFrame
            包含字段：ts_code, trade_date, open, high, low, close, vol, amount。
        """
        return self._call_with_retry(
            self._api.pro_bar,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            asset=asset,
            adj=adj,
            freq=freq,
        )

    def weekly(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取周线数据（前复权）。"""
        return self.pro_bar(ts_code, start_date, end_date, freq="W")

    def monthly(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取月线数据（前复权）。"""
        return self.pro_bar(ts_code, start_date, end_date, freq="M")

    def minute(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        freq: str = "60",
    ) -> pd.DataFrame:
        """
        获取分钟级行情数据。

        Parameters
        ----------
        ts_code : str
            股票代码。
        start_date : str
            开始时间，格式 YYYYMMDD HH:MM:SS。
        end_date : str
            结束时间，格式 YYYMMDD HH:MM:SS。
        freq : str, optional
            频率。5/15/30/60（分钟），默认 60。

        Returns
        -------
        pd.DataFrame
            包含字段：ts_code, trade_time, open, high, low, close, vol。
        """
        return self.pro_bar(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            asset="E",
            adj=None,
            freq=freq,
        )

    # ─────────────────────────────────────────────
    # 交易日历
    # ─────────────────────────────────────────────

    def trade_cal(
        self, start_date: str, end_date: str, is_open: Optional[int] = None
    ) -> pd.DataFrame:
        """
        获取交易日历。

        Parameters
        ----------
        start_date : str
            开始日期。
        end_date : str
            结束日期。
        is_open : int, optional
            是否交易。1=是，0=否。

        Returns
        -------
        pd.DataFrame
            包含字段：cal_date, is_open。
        """
        kwargs: Dict[str, Any] = {"start_date": start_date, "end_date": end_date}
        if is_open is not None:
            kwargs["is_open"] = is_open
        return self._call_with_retry(self._api.trade_cal, **kwargs)

    # ─────────────────────────────────────────────
    # 股票基础信息
    # ─────────────────────────────────────────────

    def stock_basic(
        self,
        ts_code: Optional[str] = None,
        name: Optional[str] = None,
        list_status: str = "L",
    ) -> pd.DataFrame:
        """
        获取股票基本信息。

        Parameters
        ----------
        ts_code : str, optional
            股票代码。
        name : str, optional
            股票名称（模糊搜索）。
        list_status : str, optional
            上市状态。L=上市，D=退市，P=暂停上市，默认 L。

        Returns
        -------
        pd.DataFrame
            包含字段：ts_code, symbol, name, area, industry, list_date。
        """
        kwargs: Dict[str, Any] = {"list_status": list_status}
        if ts_code:
            kwargs["ts_code"] = ts_code
        if name:
            kwargs["name"] = name
        return self._call_with_retry(self._api.stock_basic, **kwargs)

    # ─────────────────────────────────────────────
    # 财务数据
    # ─────────────────────────────────────────────

    def financial_data(
        self, ts_code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        获取财务数据（利润表、资产负债表、现金流量表合并）。

        Parameters
        ----------
        ts_code : str
            股票代码。
        start_date : str
            开始报告期，格式 YYYYMMDD，如 "20200101"。
        end_date : str
            结束报告期，格式 YYYYMMDD。

        Returns
        -------
        pd.DataFrame
            包含字段：ts_code, ann_date, f_ann_date, report_type, etc.
        """
        return self._call_with_retry(
            self._api.fina_indicator,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
        )

    def income(
        self, ts_code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """获取利润表数据。"""
        return self._call_with_retry(
            self._api.income, ts_code=ts_code, start_date=start_date, end_date=end_date
        )

    def balance_sheet(
        self, ts_code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """获取资产负债表数据。"""
        return self._call_with_retry(
            self._api.balancesheet,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
        )

    def cashflow(
        self, ts_code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """获取现金流量表数据。"""
        return self._call_with_retry(
            self._api.cashflow, ts_code=ts_code, start_date=start_date, end_date=end_date
        )

    # ─────────────────────────────────────────────
    # 指数数据
    # ─────────────────────────────────────────────

    def index_daily(
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
            包含字段：ts_code, trade_date, close, open, high, low, vol, amount。
        """
        return self._call_with_retry(
            self._api.index_daily,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
        )

    # ─────────────────────────────────────────────
    # 通用查询
    # ─────────────────────────────────────────────

    def query(
        self,
        api_name: str,
        params: Optional[Dict[str, Any]] = None,
        fields: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        通用查询接口。

        Parameters
        ----------
        api_name : str
            Tushare API 名称，如 "daily"。
        params : dict, optional
            查询参数字典。
        fields : str, optional
            返回字段列表，用逗号分隔。

        Returns
        -------
        pd.DataFrame
            查询结果。
        """
        params = params or {}
        if fields:
            params["fields"] = fields
        return self._call_with_retry(self._api.query, api_name, **params)

    def __repr__(self) -> str:
        return (
            f"TushareClient(token='{self.token[:6]}***', "
            f"max_retry={self.max_retry}, timeout={self.timeout})"
        )
