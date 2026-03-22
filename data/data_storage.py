# -*- coding: utf-8 -*-
"""
数据存储模块

提供数据持久化功能，支持CSV和Parquet格式的读写操作。
"""

import logging
import os
from pathlib import Path
from typing import Optional, List, Union, Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)


class DataStorage:
    """
    数据存储器

    提供统一的数据存储接口，支持CSV和Parquet格式，
    包含自动目录创建、压缩支持、分区存储等功能。

    Parameters
    ----------
    base_dir : str, optional
        基础存储目录，默认为当前目录下的 "data"。
    compression : str, optional
        压缩格式，默认"snappy"。
        CSV支持：'gzip', 'bz2', 'zip', 'xz'
        Parquet支持：'snappy', 'gzip', 'brotli', None
    chunk_size : int, optional
        分块写入大小，用于大文件分块处理，默认10000。

    Examples
    --------
    >>> storage = DataStorage(base_dir="./data", compression="snappy")
    >>> storage.save_to_csv(df, "daily/stock_000001.csv")
    >>> df = storage.load_from_csv("daily/stock_000001.csv")
    """

    def __init__(
        self,
        base_dir: Optional[str] = None,
        compression: str = "snappy",
        chunk_size: int = 10000,
    ):
        # 基础目录
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            self.base_dir = Path("data")

        self.base_dir.mkdir(parents=True, exist_ok=True)

        # 配置
        self.compression = compression
        self.chunk_size = chunk_size

        logger.info(f"DataStorage initialized: base_dir={self.base_dir}, compression={compression}")

    # ─────────────────────────────────────────────
    # CSV 存储方法
    # ─────────────────────────────────────────────

    def save_to_csv(
        self,
        df: pd.DataFrame,
        relative_path: str,
        index: bool = False,
        encoding: str = "utf-8-sig",
        compression: Optional[str] = None,
    ) -> str:
        """
        保存数据到CSV文件。

        Parameters
        ----------
        df : pd.DataFrame
            要保存的数据。
        relative_path : str
            相对于base_dir的路径，如 "daily/stock_000001.csv"。
        index : bool, optional
            是否保存索引，默认False。
        encoding : str, optional
            文件编码，默认"utf-8-sig"（支持中文）。
        compression : str, optional
            压缩格式，为None时使用实例默认值。

        Returns
        -------
        str
            完整的文件路径。

        Raises
        ------
        ValueError
            数据为空时抛出。
        IOError
            写入失败时抛出。
        """
        if df is None or df.empty:
            raise ValueError("Cannot save empty DataFrame")

        # 构建完整路径
        full_path = self.base_dir / relative_path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # 确定压缩格式
        compression = compression or self._get_csv_compression(str(full_path))

        try:
            df.to_csv(
                full_path,
                index=index,
                encoding=encoding,
                compression=compression,
            )
            logger.info(f"Saved {len(df)} rows to {full_path}")
            return str(full_path)

        except Exception as e:
            logger.error(f"Failed to save CSV to {full_path}: {e}")
            raise IOError(f"Failed to save CSV: {e}")

    def load_from_csv(
        self,
        relative_path: str,
        encoding: str = "utf-8-sig",
        parse_dates: Optional[List[str]] = None,
        columns: Optional[List[str]] = None,
        nrows: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        从CSV文件加载数据。

        Parameters
        ----------
        relative_path : str
            相对于base_dir的路径。
        encoding : str, optional
            文件编码，默认"utf-8-sig"。
        parse_dates : list, optional
            需要解析为日期的列。
        columns : list, optional
            需要读取的列。
        nrows : int, optional
            读取行数限制。

        Returns
        -------
        pd.DataFrame
            加载的数据。

        Raises
        ------
        FileNotFoundError
            文件不存在时抛出。
        """
        full_path = self.base_dir / relative_path

        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {full_path}")

        try:
            df = pd.read_csv(
                full_path,
                encoding=encoding,
                parse_dates=parse_dates,
                usecols=columns,
                nrows=nrows,
            )
            logger.info(f"Loaded {len(df)} rows from {full_path}")
            return df

        except Exception as e:
            logger.error(f"Failed to load CSV from {full_path}: {e}")
            raise IOError(f"Failed to load CSV: {e}")

    def append_to_csv(
        self,
        df: pd.DataFrame,
        relative_path: str,
        index: bool = False,
        encoding: str = "utf-8-sig",
    ) -> str:
        """
        追加数据到CSV文件。

        Parameters
        ----------
        df : pd.DataFrame
            要追加的数据。
        relative_path : str
            相对于base_dir的路径。
        index : bool, optional
            是否保存索引，默认False。
        encoding : str, optional
            文件编码。

        Returns
        -------
        str
            完整的文件路径。
        """
        full_path = self.base_dir / relative_path

        # 如果文件不存在，直接保存
        if not full_path.exists():
            return self.save_to_csv(df, relative_path, index, encoding)

        # 追加模式
        try:
            # 读取现有数据
            existing_df = pd.read_csv(full_path, encoding=encoding)

            # 合并数据
            combined_df = pd.concat([existing_df, df], ignore_index=True)

            # 去重（可选）
            combined_df = combined_df.drop_duplicates()

            # 写回
            combined_df.to_csv(full_path, index=index, encoding=encoding)

            logger.info(f"Appended {len(df)} rows to {full_path}, total {len(combined_df)}")
            return str(full_path)

        except Exception as e:
            logger.error(f"Failed to append to CSV: {e}")
            raise

    # ─────────────────────────────────────────────
    # Parquet 存储方法
    # ─────────────────────────────────────────────

    def save_to_parquet(
        self,
        df: pd.DataFrame,
        relative_path: str,
        index: bool = False,
        compression: Optional[str] = None,
        engine: str = "pyarrow",
        partition_cols: Optional[List[str]] = None,
    ) -> str:
        """
        保存数据到Parquet文件。

        Parameters
        ----------
        df : pd.DataFrame
            要保存的数据。
        relative_path : str
            相对于base_dir的路径，如 "daily/stock_000001.parquet"。
        index : bool, optional
            是否保存索引，默认False。
        compression : str, optional
            压缩格式，为None时使用实例默认值（snappy）。
        engine : str, optional
            引擎类型，默认"pyarrow"，也可选"fastparquet"。
        partition_cols : list, optional
            分区列，如 ["year", "month"]。

        Returns
        -------
        str
            完整的文件路径或目录路径（分区存储时）。

        Raises
        ------
        ValueError
            数据为空时抛出。
        """
        if df is None or df.empty:
            raise ValueError("Cannot save empty DataFrame")

        # 构建完整路径
        full_path = self.base_dir / relative_path

        # 移除扩展名（如果存在）
        if full_path.suffix == ".parquet":
            full_path = full_path.with_suffix("")

        # 分区存储
        if partition_cols:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(
                full_path,
                index=index,
                compression=compression or self.compression,
                engine=engine,
                partition_cols=partition_cols,
            )
            logger.info(f"Saved {len(df)} rows to partitioned parquet at {full_path}")
            return str(full_path)

        # 普通存储
        full_path = Path(str(full_path) + ".parquet")
        full_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            df.to_parquet(
                full_path,
                index=index,
                compression=compression or self.compression,
                engine=engine,
            )
            logger.info(f"Saved {len(df)} rows to {full_path}")
            return str(full_path)

        except Exception as e:
            logger.error(f"Failed to save Parquet to {full_path}: {e}")
            raise IOError(f"Failed to save Parquet: {e}")

    def load_from_parquet(
        self,
        relative_path: str,
        columns: Optional[List[str]] = None,
        filters: Optional[List[tuple]] = None,
    ) -> pd.DataFrame:
        """
        从Parquet文件加载数据。

        Parameters
        ----------
        relative_path : str
            相对于base_dir的路径。
        columns : list, optional
            需要读取的列。
        filters : list, optional
            分区过滤条件，如 [("year", "=", 2024)]。

        Returns
        -------
        pd.DataFrame
            加载的数据。

        Raises
        ------
        FileNotFoundError
            文件不存在时抛出。
        """
        full_path = self.base_dir / relative_path

        # 处理扩展名
        if not full_path.suffix == ".parquet":
            full_path = Path(str(full_path) + ".parquet")

        if not full_path.exists():
            # 可能是分区目录
            if not Path(str(full_path).replace(".parquet", "")).exists():
                raise FileNotFoundError(f"File not found: {full_path}")

        try:
            df = pd.read_parquet(
                full_path,
                columns=columns,
                filters=filters,
            )
            logger.info(f"Loaded {len(df)} rows from {full_path}")
            return df

        except Exception as e:
            logger.error(f"Failed to load Parquet from {full_path}: {e}")
            raise IOError(f"Failed to load Parquet: {e}")

    # ─────────────────────────────────────────────
    # 通用方法
    # ─────────────────────────────────────────────

    def save(
        self,
        df: pd.DataFrame,
        relative_path: str,
        format: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        通用保存方法，自动识别格式。

        Parameters
        ----------
        df : pd.DataFrame
            要保存的数据。
        relative_path : str
            相对路径。
        format : str, optional
            格式类型，'csv' 或 'parquet'，为None时根据扩展名自动判断。
        **kwargs
            其他参数传递给具体的保存方法。

        Returns
        -------
        str
            保存的完整路径。
        """
        if format is None:
            format = self._detect_format(relative_path)

        if format == "csv":
            return self.save_to_csv(df, relative_path, **kwargs)
        elif format == "parquet":
            return self.save_to_parquet(df, relative_path, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def load(
        self,
        relative_path: str,
        format: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        通用加载方法，自动识别格式。

        Parameters
        ----------
        relative_path : str
            相对路径。
        format : str, optional
            格式类型，'csv' 或 'parquet'，为None时根据扩展名自动判断。
        **kwargs
            其他参数传递给具体的加载方法。

        Returns
        -------
        pd.DataFrame
            加载的数据。
        """
        if format is None:
            format = self._detect_format(relative_path)

        if format == "csv":
            return self.load_from_csv(relative_path, **kwargs)
        elif format == "parquet":
            return self.load_from_parquet(relative_path, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def delete(self, relative_path: str) -> bool:
        """
        删除指定文件。

        Parameters
        ----------
        relative_path : str
            相对路径。

        Returns
        -------
        bool
            是否成功删除。
        """
        full_path = self.base_dir / relative_path

        # 处理Parquet路径
        if not full_path.suffix:
            parquet_path = Path(str(full_path) + ".parquet")
            if parquet_path.exists():
                if parquet_path.is_dir():
                    shutil.rmtree(parquet_path)
                else:
                    parquet_path.unlink()
                return True

        if not full_path.exists():
            return False

        try:
            if full_path.is_dir():
                shutil.rmtree(full_path)
            else:
                full_path.unlink()
            logger.info(f"Deleted: {full_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete {full_path}: {e}")
            return False

    def list_files(
        self,
        relative_dir: str = "",
        pattern: str = "*",
        recursive: bool = False,
    ) -> List[str]:
        """
        列出缓存目录中的文件。

        Parameters
        ----------
        relative_dir : str, optional
            相对目录。
        pattern : str, optional
            文件名模式，默认"*"。
        recursive : bool, optional
            是否递归搜索，默认False。

        Returns
        -------
        list
            文件路径列表（相对于base_dir）。
        """
        full_dir = self.base_dir / relative_dir

        if not full_dir.exists():
            return []

        if recursive:
            files = full_dir.rglob(pattern)
        else:
            files = full_dir.glob(pattern)

        return [str(f.relative_to(self.base_dir)) for f in files if f.is_file()]

    def get_file_info(self, relative_path: str) -> Dict[str, Any]:
        """
        获取文件信息。

        Parameters
        ----------
        relative_path : str
            相对路径。

        Returns
        -------
        dict
            包含文件大小、修改时间等信息。
        """
        full_path = self.base_dir / relative_path

        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {full_path}")

        stat = full_path.stat()
        return {
            "path": str(full_path),
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / 1024 / 1024, 2),
            "modified_time": pd.Timestamp(stat.st_mtime, unit="s"),
            "is_dir": full_path.is_dir(),
        }

    # ─────────────────────────────────────────────
    # 辅助方法
    # ─────────────────────────────────────────────

    def _detect_format(self, path: str) -> str:
        """根据文件扩展名检测格式。"""
        ext = Path(path).suffix.lower()
        if ext in [".csv", ".txt", ".tsv"]:
            return "csv"
        elif ext in [".parquet", ".pq"]:
            return "parquet"
        else:
            # 默认尝试CSV
            return "csv"

    def _get_csv_compression(self, filepath: str) -> Optional[str]:
        """根据文件扩展名确定CSV压缩格式。"""
        ext = Path(filepath).suffix.lower()
        compression_map = {
            ".gz": "gzip",
            ".gzip": "gzip",
            ".bz2": "bz2",
            ".zip": "zip",
            ".xz": "xz",
        }
        return compression_map.get(ext)

    def __repr__(self) -> str:
        return f"DataStorage(base_dir='{self.base_dir}', compression='{self.compression}')"
