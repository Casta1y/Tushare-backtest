# -*- coding: utf-8 -*-
"""
数据清洗模块

提供数据清洗功能，包括缺失值处理、异常值处理、数据类型转换等。
"""

import logging
from typing import Optional, List, Dict, Any, Union, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    数据清洗器

    提供完整的数据清洗流程，支持缺失值处理、异常值检测与处理、
    数据类型转换等常用功能。

    Parameters
    ----------
    strict_mode : bool, optional
        严格模式，异常值直接删除而非标记，默认False。
    fill_method : str, optional
        缺失值填充方法，默认"ffill"。
        可选：'ffill', 'bfill', 'mean', 'median', 'zero', 'interpolate'

    Examples
    --------
    >>> cleaner = DataCleaner(strict_mode=False, fill_method="ffill")
    >>> df_clean = cleaner.clean_data(df)
    """

    # 常用的价格和交易量字段
    PRICE_COLUMNS = ["open", "high", "low", "close", "pre_close"]
    VOLUME_COLUMNS = ["vol", "volume", "amount"]
    DATE_COLUMNS = ["trade_date", "ann_date", "f_ann_date", "report_date"]

    def __init__(
        self,
        strict_mode: bool = False,
        fill_method: str = "ffill",
    ):
        self.strict_mode = strict_mode
        self.fill_method = fill_method
        self._validation_methods = {
            "price": self._validate_price,
            "volume": self._validate_volume,
            "date": self._validate_date,
            "percentage": self._validate_percentage,
        }

    def clean_data(
        self,
        df: pd.DataFrame,
        drop_duplicates: bool = True,
        sort_by: Optional[str] = None,
        ascending: bool = True,
    ) -> pd.DataFrame:
        """
        完整的数据清洗流程。

        Parameters
        ----------
        df : pd.DataFrame
            原始数据。
        drop_duplicates : bool, optional
            是否删除重复行，默认True。
        sort_by : str, optional
            排序字段。
        ascending : bool, optional
            升序还是降序，默认True。

        Returns
        -------
        pd.DataFrame
            清洗后的数据。

        Examples
        --------
        >>> df = cleaner.clean_data(df, drop_duplicates=True, sort_by="trade_date")
        """
        if df is None or df.empty:
            logger.warning("Input DataFrame is empty, returning as is")
            return df

        logger.info(f"Starting data cleaning for {len(df)} records")
        df = df.copy()

        # 1. 删除重复行
        if drop_duplicates:
            before = len(df)
            df = df.drop_duplicates()
            if len(df) < before:
                logger.info(f"Removed {before - len(df)} duplicate rows")

        # 2. 排序
        if sort_by and sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)

        # 3. 处理缺失值
        df = self.handle_missing_values(df)

        # 4. 处理异常值
        df = self.handle_outliers(df)

        # 5. 数据类型转换
        df = self.convert_data_types(df)

        logger.info(f"Data cleaning completed, {len(df)} records remaining")
        return df

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        fill_method: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        处理缺失值。

        Parameters
        ----------
        df : pd.DataFrame
            输入数据。
        columns : list, optional
            指定需要处理的列，为None时处理所有列。
        fill_method : str, optional
            填充方法，默认使用实例的fill_method。
            可选：'ffill', 'bfill', 'mean', 'median', 'zero', 'interpolate'

        Returns
        -------
        pd.DataFrame
            处理后的数据。
        """
        if df.empty:
            return df

        df = df.copy()
        fill_method = fill_method or self.fill_method

        # 确定要处理的列
        if columns is None:
            columns = df.columns.tolist()

        for col in columns:
            if col not in df.columns:
                continue

            # 检查缺失值数量
            missing_count = df[col].isna().sum()
            if missing_count == 0:
                continue

            logger.debug(f"Processing column '{col}': {missing_count} missing values")

            # 根据数据类型选择填充方法
            if df[col].dtype in ["float64", "int64", "float32", "int32"]:
                # 数值列
                if fill_method == "ffill":
                    df[col] = df[col].fillna(method="ffill")
                elif fill_method == "bfill":
                    df[col] = df[col].fillna(method="bfill")
                elif fill_method == "mean":
                    df[col] = df[col].fillna(df[col].mean())
                elif fill_method == "median":
                    df[col] = df[col].fillna(df[col].median())
                elif fill_method == "zero":
                    df[col] = df[col].fillna(0)
                elif fill_method == "interpolate":
                    df[col] = df[col].interpolate(method="linear")
                else:
                    df[col] = df[col].fillna(method="ffill")
            else:
                # 非数值列，使用前向填充
                df[col] = df[col].fillna(method="ffill")

        return df

    def handle_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = "iqr",
        threshold: float = 3.0,
        action: str = "clip",
    ) -> pd.DataFrame:
        """
        处理异常值。

        Parameters
        ----------
        df : pd.DataFrame
            输入数据。
        columns : list, optional
            指定需要处理的列，为None时处理价格和交易量列。
        method : str, optional
            异常值检测方法，默认"iqr"。
            可选：'iqr'（四分位距），'zscore'（Z分数）
        threshold : float, optional
            阈值，默认3.0。
        action : str, optional
            处理方式，默认"clip"（裁剪）。
            可选：'clip'（裁剪到边界），'nan'（替换为NaN），'mean'（替换为均值）

        Returns
        -------
        pd.DataFrame
            处理后的数据。
        """
        if df.empty:
            return df

        df = df.copy()

        # 默认处理价格和交易量列
        if columns is None:
            columns = [c for c in self.PRICE_COLUMNS + self.VOLUME_COLUMNS if c in df.columns]

        for col in columns:
            if col not in df.columns:
                continue

            if not pd.api.types.is_numeric_dtype(df[col]):
                continue

            if method == "iqr":
                df = self._handle_outliers_iqr(df, col, action)
            elif method == "zscore":
                df = self._handle_outliers_zscore(df, col, threshold, action)
            else:
                logger.warning(f"Unknown outlier method: {method}")

        return df

    def _handle_outliers_iqr(
        self,
        df: pd.DataFrame,
        column: str,
        action: str = "clip",
    ) -> pd.DataFrame:
        """使用IQR方法处理异常值。"""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
        outlier_count = outlier_mask.sum()

        if outlier_count > 0:
            logger.debug(f"Column '{column}': {outlier_count} outliers detected (IQR method)")

            if action == "clip":
                df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
            elif action == "nan":
                df.loc[outlier_mask, column] = np.nan
            elif action == "mean":
                df.loc[outlier_mask, column] = df[column].mean()

        return df

    def _handle_outliers_zscore(
        self,
        df: pd.DataFrame,
        column: str,
        threshold: float = 3.0,
        action: str = "clip",
    ) -> pd.DataFrame:
        """使用Z-score方法处理异常值。"""
        mean = df[column].mean()
        std = df[column].std()

        if std == 0:
            return df

        z_scores = np.abs((df[column] - mean) / std)
        outlier_mask = z_scores > threshold
        outlier_count = outlier_mask.sum()

        if outlier_count > 0:
            logger.debug(f"Column '{column}': {outlier_count} outliers detected (Z-score method)")

            if action == "clip":
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
                df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
            elif action == "nan":
                df.loc[outlier_mask, column] = np.nan
            elif action == "mean":
                df.loc[outlier_mask, column] = mean

        return df

    def convert_data_types(
        self,
        df: pd.DataFrame,
        date_columns: Optional[List[str]] = None,
        numeric_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        数据类型转换。

        Parameters
        ----------
        df : pd.DataFrame
            输入数据。
        date_columns : list, optional
            日期列列表，为None时自动识别。
        numeric_columns : list, optional
            数值列列表，为None时自动处理。

        Returns
        -------
        pd.DataFrame
            转换后的数据。
        """
        if df.empty:
            return df

        df = df.copy()

        # 日期列转换
        if date_columns is None:
            date_columns = [c for c in self.DATE_COLUMNS if c in df.columns]

        for col in date_columns:
            if col in df.columns:
                df[col] = self._convert_to_date(df[col])

        # 数值列转换
        if numeric_columns is None:
            numeric_columns = [c for c in self.PRICE_COLUMNS + self.VOLUME_COLUMNS if c in df.columns]

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def _convert_to_date(self, series: pd.Series) -> pd.Series:
        """将序列转换为日期类型。"""
        if series.dtype == "object" or series.dtype == "int64":
            # 尝试多种日期格式
            try:
                # YYYYMMDD 格式
                return pd.to_datetime(series.astype(str), format="%Y%m%d")
            except:
                try:
                    # 标准日期格式
                    return pd.to_datetime(series)
                except:
                    return series
        return series

    # ─────────────────────────────────────────────
    # 数据验证方法
    # ─────────────────────────────────────────────

    def validate_data(
        self,
        df: pd.DataFrame,
        checks: Optional[List[str]] = None,
    ) -> Tuple[bool, List[str]]:
        """
        验证数据质量。

        Parameters
        ----------
        df : pd.DataFrame
            待验证的数据。
        checks : list, optional
            检查项列表。
            可选：'price', 'volume', 'date', 'percentage', 'all'

        Returns
        -------
        tuple
            (是否通过验证, 错误信息列表)。
        """
        if df.empty:
            return True, []

        errors = []
        checks = checks or ["price", "volume"]

        for check_type in checks:
            if check_type == "all":
                checks = ["price", "volume", "date", "percentage"]
                break

            validator = self._validation_methods.get(check_type)
            if validator:
                is_valid, error_msg = validator(df)
                if not is_valid:
                    errors.append(error_msg)

        return len(errors) == 0, errors

    def _validate_price(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """验证价格数据的合理性。"""
        for col in self.PRICE_COLUMNS:
            if col not in df.columns:
                continue

            # 检查负值
            if (df[col] < 0).any():
                return False, f"Found negative prices in column '{col}'"

            # 检查高估（收盘价远高于开盘价50%）
            if "close" in df.columns and "open" in df.columns:
                if ((df["close"] / df["open"] - 1) > 0.5).any():
                    return False, "Found suspiciously high price changes (>50%)"

        return True, ""

    def _validate_volume(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """验证交易量数据的合理性。"""
        for col in self.VOLUME_COLUMNS:
            if col not in df.columns:
                continue

            if (df[col] < 0).any():
                return False, f"Found negative volume in column '{col}'"

        return True, ""

    def _validate_date(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """验证日期数据的合法性。"""
        for col in self.DATE_COLUMNS:
            if col not in df.columns:
                continue

            if df[col].dtype == "object":
                try:
                    pd.to_datetime(df[col], format="%Y%m%d")
                except:
                    return False, f"Invalid date format in column '{col}'"

        return True, ""

    def _validate_percentage(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """验证百分比数据的合理性。"""
        pct_cols = [c for c in df.columns if c.endswith("_pct") or c.endswith("_rate")]
        for col in pct_cols:
            if col not in df.columns:
                continue
            # 检查极端值
            if ((df[col] > 100) | (df[col] < -100)).any():
                logger.warning(f"Found extreme percentage values in column '{col}'")

        return True, ""

    # ─────────────────────────────────────────────
    # 便捷方法
    # ─────────────────────────────────────────────

    def remove_missing_rows(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        threshold: float = 1.0,
    ) -> pd.DataFrame:
        """
        删除缺失值过多的行。

        Parameters
        ----------
        df : pd.DataFrame
            输入数据。
        columns : list, optional
            检查缺失值的列，为None时检查所有列。
        threshold : float, optional
            缺失比例阈值，超过该比例的行被删除，默认1.0（全部为空才删除）。

        Returns
        -------
        pd.DataFrame
            处理后的数据。
        """
        if df.empty or threshold <= 0:
            return df

        df = df.copy()

        if columns is None:
            columns = df.columns.tolist()

        # 计算每行的缺失比例
        missing_ratio = df[columns].isna().sum(axis=1) / len(columns)
        df = df[missing_ratio < threshold].reset_index(drop=True)

        removed = len(missing_ratio[missing_ratio >= threshold])
        if removed > 0:
            logger.info(f"Removed {removed} rows with excessive missing values")

        return df

    def fill_forward(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        前向填充指定列。

        Parameters
        ----------
        df : pd.DataFrame
            输入数据。
        columns : list
            需要前向填充的列。

        Returns
        -------
        pd.DataFrame
            处理后的数据。
        """
        df = df.copy()
        for col in columns:
            if col in df.columns:
                df[col] = df[col].fillna(method="ffill")
        return df

    def __repr__(self) -> str:
        return f"DataCleaner(strict_mode={self.strict_mode}, fill_method='{self.fill_method}')"
