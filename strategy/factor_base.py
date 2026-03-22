# -*- coding: utf-8 -*-
"""
因子基类 (Factor Base)

定义所有因子的抽象基类，提供统一的接口规范和通用功能。
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FactorBase(ABC):
    """
    所有量化因子的抽象基类。

    子类需要：
    1. 定义 ``name`` 类属性（因子名称）
    2. 实现 ``_calculate_single()`` 方法（核心计算逻辑）
    3. 可选重写 ``_validate_params()`` 方法（参数验证）

    Parameters
    ----------
    params : dict, optional
        因子参数字典，如 {"period": 20}。

    Attributes
    ----------
    name : str
        因子名称（子类必须定义）。
    params : dict
        因子参数。
    required_cols : list
        必需输入列名（默认 ["close"]，子类可覆盖）。

    Examples
    --------
    >>> class MA5Factor(FactorBase):
    ...     name = "MA5"
    ...     required_cols = ["close"]
    ...     def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
    ...         return df["close"].rolling(window=5).mean()
    >>> factor = MA5Factor({"period": 5})
    >>> result = factor.calculate(df)
    """

    # 子类必须定义
    name: str = "FactorBase"
    required_cols: List[str] = ["close"]

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        初始化因子。

        Parameters
        ----------
        params : dict, optional
            因子参数字典。
        """
        self.params: Dict[str, Any] = params or {}
        self._validate_params()

    # ─────────────────────────────────────────────
    # 子类必须实现
    # ─────────────────────────────────────────────

    @abstractmethod
    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        """
        单个因子的核心计算逻辑（子类必须实现）。

        Parameters
        ----------
        df : pd.DataFrame
            标准化后的输入数据，包含所有 ``required_cols`` 列。

        Returns
        -------
        pd.Series
            计算结果（索引与 df 相同）。
        """
        ...

    # ─────────────────────────────────────────────
    # 参数验证（子类可覆盖）
    # ─────────────────────────────────────────────

    def _validate_params(self) -> None:
        """
        验证因子参数字典。

        默认实现检查参数类型，可以被子类重写以添加自定义验证逻辑。

        Raises
        ------
        ValueError
            参数值不合法时抛出。
        """
        for key, value in self.params.items():
            if value is None:
                raise ValueError(f"Parameter '{key}' cannot be None.")
            if not isinstance(value, (int, float, str, bool, list, tuple, type(None))):
                raise ValueError(
                    f"Parameter '{key}' has unsupported type: {type(value).__name__}"
                )

        # 默认检查：如果 params 中包含 period，确保为正整数
        if "period" in self.params:
            period = self.params["period"]
            if not isinstance(period, (int, float)):
                raise ValueError(
                    f"Parameter 'period' must be numeric, got {type(period).__name__}"
                )
            if isinstance(period, float) and period != int(period):
                raise ValueError(f"Parameter 'period' must be integer, got {period}")
            if period <= 0:
                raise ValueError(f"Parameter 'period' must be positive, got {period}")

    # ─────────────────────────────────────────────
    # 公开接口
    # ─────────────────────────────────────────────

    def calculate(
        self,
        data: Union[pd.DataFrame, pd.Series],
        col: Optional[str] = None,
    ) -> pd.Series:
        """
        计算因子值（统一入口）。

        自动处理 DataFrame / Series 两种输入格式，并验证必需列。

        Parameters
        ----------
        data : pd.DataFrame or pd.Series
            输入数据。
            - DataFrame：必须包含 ``required_cols`` 中的所有列。
            - Series：直接作为 ``close`` 列使用。
        col : str, optional
            当 data 为 Series 时，指定列名（默认使用 Series 的 name）。

        Returns
        -------
        pd.Series
            因子值序列，索引与输入数据一致。

        Raises
        ------
        ValueError
            输入数据为空或不包含必需列时抛出。
        """
        # ── 处理输入格式 ──
        df = self._normalize_input(data, col)

        # ── 验证数据 ──
        self._validate_data(df)

        # ── 计算 ──
        logger.debug(f"Calculating factor '{self.name}' with params={self.params}")
        result = self._calculate_single(df)
        result.name = self.name

        # ── 同步索引 ──
        result = result.reset_index(drop=True)
        if "trade_date" in df.columns:
            result.index = df["trade_date"].values

        return result

    def get_name(self) -> str:
        """返回因子名称。"""
        return self.name

    def get_params(self) -> Dict[str, Any]:
        """返回当前参数字典。"""
        return self.params.copy()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', params={self.params})"

    # ─────────────────────────────────────────────
    # 内部工具方法
    # ─────────────────────────────────────────────

    def _normalize_input(
        self,
        data: Union[pd.DataFrame, pd.Series],
        col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        将 Series 或 DataFrame 规范化为标准 DataFrame 格式。

        Parameters
        ----------
        data : pd.DataFrame or pd.Series
            输入数据。
        col : str, optional
            Series 列名（默认使用 Series.name）。

        Returns
        -------
        pd.DataFrame
            至少包含 required_cols 的 DataFrame。
        """
        if isinstance(data, pd.Series):
            series_name = col or data.name or "value"
            return pd.DataFrame({series_name: data.values}, index=data.index)
        elif isinstance(data, pd.DataFrame):
            return data.copy()
        else:
            raise TypeError(
                f"Input data must be pd.DataFrame or pd.Series, "
                f"got {type(data).__name__}"
            )

    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        验证输入数据的必要列和非空性。

        Parameters
        ----------
        df : pd.DataFrame

        Raises
        ------
        ValueError
            缺少必需列或数据为空时抛出。
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        missing_cols = set(self.required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Missing required columns for factor '{self.name}': {missing_cols}. "
                f"Required: {self.required_cols}"
            )


class CompositeFactor(FactorBase):
    """
    复合因子基类，支持组合多个子因子。

    子类只需实现 ``_build_components()`` 返回子因子列表。

    Examples
    --------
    >>> class DualMAFactor(CompositeFactor):
    ...     name = "DualMA"
    ...     required_cols = ["close"]
    ...     def _build_components(self) -> List[FactorBase]:
    ...         return [MA5Factor(), MA20Factor()]
    """

    name: str = "CompositeFactor"
    required_cols: List[str] = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        components = self._build_components()
        results = []
        for factor in components:
            result = factor.calculate(df)
            results.append(result)
        return pd.concat(results, axis=1).mean(axis=1)

    @abstractmethod
    def _build_components(self) -> List[FactorBase]:
        """返回组成该复合因子的子因子列表。"""
        ...
