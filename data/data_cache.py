# -*- coding: utf-8 -*-
"""
数据缓存模块

提供数据缓存功能，支持内存缓存和磁盘缓存，包含过期机制。
"""

import hashlib
import json
import logging
import os
import pickle
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Union, Dict

import pandas as pd

logger = logging.getLogger(__name__)


class DataCache:
    """
    数据缓存器

    提供两级缓存机制：内存缓存（快速访问）和磁盘缓存（持久化）。
    支持缓存过期、容量管理、压缩存储等功能。

    Parameters
    ----------
    cache_dir : str, optional
        磁盘缓存目录，默认为当前目录下的 ".cache"。
    max_memory_items : int, optional
        内存缓存最大条目数，默认100。
    default_ttl : int, optional
        默认过期时间（秒），默认86400（24小时）。
    compression : bool, optional
        是否启用压缩，默认True。

    Examples
    --------
    >>> cache = DataCache(cache_dir="./cache", default_ttl=3600)
    >>> cache.save("daily_000001", df)
    >>> df = cache.load("daily_000001")
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_memory_items: int = 100,
        default_ttl: int = 86400,
        compression: bool = True,
    ):
        # 缓存目录设置
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(".cache")

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 内存缓存
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self.max_memory_items = max_memory_items

        # 缓存配置
        self.default_ttl = default_ttl
        self.compression = compression

        # 初始化统计信息
        self._stats = {
            "hits": 0,
            "misses": 0,
            "saves": 0,
            "loads": 0,
            "evictions": 0,
        }

        logger.info(f"DataCache initialized: dir={self.cache_dir}, ttl={default_ttl}s")

    def save(
        self,
        key: str,
        data: Any,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        保存数据到缓存。

        Parameters
        ----------
        key : str
            缓存键。
        data : Any
            要缓存的数据，通常为pandas.DataFrame。
        ttl : int, optional
            过期时间（秒），为None时使用默认TTL。
        metadata : dict, optional
            附加的元数据信息。

        Returns
        -------
        bool
            是否保存成功。
        """
        try:
            ttl = ttl if ttl is not None else self.default_ttl
            expire_time = datetime.now() + timedelta(seconds=ttl)

            # 构建缓存项
            cache_item = {
                "data": data,
                "created_at": datetime.now(),
                "expire_at": expire_time,
                "metadata": metadata or {},
            }

            # 保存到内存缓存
            self._save_to_memory(key, cache_item)

            # 保存到磁盘缓存
            self._save_to_disk(key, data, expire_time, metadata)

            self._stats["saves"] += 1
            logger.debug(f"Cached data with key '{key}', TTL={ttl}s")

            return True

        except Exception as e:
            logger.error(f"Failed to cache data with key '{key}': {e}")
            return False

    def load(
        self,
        key: str,
        default: Any = None,
        check_expired: bool = True,
    ) -> Any:
        """
        从缓存加载数据。

        Parameters
        ----------
        key : str
            缓存键。
        default : Any, optional
            缓存未命中时的默认返回值。
        check_expired : bool, optional
            是否检查过期，默认True。

        Returns
        -------
        Any
            缓存的数据，未命中时返回default。
        """
        # 先尝试从内存缓存加载
        memory_data = self._load_from_memory(key, check_expired)
        if memory_data is not None:
            self._stats["hits"] += 1
            self._stats["loads"] += 1
            logger.debug(f"Memory cache hit for key '{key}'")
            return memory_data

        # 再尝试从磁盘缓存加载
        disk_data = self._load_from_disk(key, check_expired)
        if disk_data is not None:
            # 更新内存缓存
            expire_time = datetime.now() + timedelta(seconds=self.default_ttl)
            self._save_to_memory(key, {
                "data": disk_data,
                "created_at": datetime.now(),
                "expire_at": expire_time,
                "metadata": {},
            })
            self._stats["hits"] += 1
            self._stats["loads"] += 1
            logger.debug(f"Disk cache hit for key '{key}'")
            return disk_data

        # 缓存未命中
        self._stats["misses"] += 1
        logger.debug(f"Cache miss for key '{key}'")
        return default

    def _save_to_memory(self, key: str, cache_item: Dict[str, Any]) -> None:
        """保存到内存缓存。"""
        # 如果缓存已满，执行LRU淘汰
        if len(self._memory_cache) >= self.max_memory_items:
            self._evict_lru()

        self._memory_cache[key] = cache_item

    def _load_from_memory(
        self,
        key: str,
        check_expired: bool = True,
    ) -> Optional[Any]:
        """从内存缓存加载。"""
        if key not in self._memory_cache:
            return None

        cache_item = self._memory_cache[key]

        # 检查过期
        if check_expired and self._is_expired(cache_item):
            del self._memory_cache[key]
            return None

        return cache_item["data"]

    def _save_to_disk(
        self,
        key: str,
        data: Any,
        expire_time: datetime,
        metadata: Optional[Dict] = None,
    ) -> None:
        """保存到磁盘缓存。"""
        # 生成安全的文件名
        filename = self._get_cache_filename(key)
        filepath = self.cache_dir / filename

        # 元数据
        meta = {
            "key": key,
            "created_at": datetime.now().isoformat(),
            "expire_at": expire_time.isoformat(),
            "metadata": metadata or {},
        }

        try:
            # 使用pickle保存数据
            with open(filepath, "wb") as f:
                pickle.dump(data, f)

            # 保存元数据
            meta_filepath = self.cache_dir / f"{filename}.meta"
            with open(meta_filepath, "w") as f:
                json.dump(meta, f, indent=2, default=str)

        except Exception as e:
            logger.warning(f"Failed to save to disk cache: {e}")

    def _load_from_disk(
        self,
        key: str,
        check_expired: bool = True,
    ) -> Optional[Any]:
        """从磁盘缓存加载。"""
        filename = self._get_cache_filename(key)
        filepath = self.cache_dir / filename

        if not filepath.exists():
            return None

        # 检查元数据和过期时间
        if check_expired:
            meta_filepath = self.cache_dir / f"{filename}.meta"
            if meta_filepath.exists():
                try:
                    with open(meta_filepath, "r") as f:
                        meta = json.load(f)
                    expire_at = datetime.fromisoformat(meta["expire_at"])
                    if datetime.now() > expire_at:
                        # 已过期，删除文件
                        self.delete(key)
                        return None
                except Exception as e:
                    logger.warning(f"Failed to check cache expiry: {e}")

        # 加载数据
        try:
            with open(filepath, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load from disk cache: {e}")
            return None

    def _is_expired(self, cache_item: Dict[str, Any]) -> bool:
        """检查缓存项是否过期。"""
        expire_at = cache_item.get("expire_at")
        if expire_at is None:
            return False
        if isinstance(expire_at, datetime):
            return datetime.now() > expire_at
        return False

    def _evict_lru(self) -> None:
        """淘汰最久未使用的缓存项。"""
        if not self._memory_cache:
            return

        # 找到最早创建的项
        oldest_key = min(
            self._memory_cache.keys(),
            key=lambda k: self._memory_cache[k].get("created_at", datetime.min)
        )
        del self._memory_cache[oldest_key]
        self._stats["evictions"] += 1
        logger.debug(f"Evicted LRU cache entry: {oldest_key}")

    def _get_cache_filename(self, key: str) -> str:
        """根据key生成安全的文件名。"""
        # 使用MD5哈希确保文件名安全
        hash_obj = hashlib.md5(key.encode())
        return f"cache_{hash_obj.hexdigest()}.pkl"

    def delete(self, key: str) -> bool:
        """
        删除指定缓存项。

        Parameters
        ----------
        key : str
            缓存键。

        Returns
        -------
        bool
            是否成功删除。
        """
        # 从内存删除
        if key in self._memory_cache:
            del self._memory_cache[key]

        # 从磁盘删除
        filename = self._get_cache_filename(key)
        filepath = self.cache_dir / filename
        meta_filepath = self.cache_dir / f"{filename}.meta"

        deleted = False
        if filepath.exists():
            filepath.unlink()
            deleted = True
        if meta_filepath.exists():
            meta_filepath.unlink()
            deleted = True

        if deleted:
            logger.debug(f"Deleted cache entry: {key}")

        return deleted

    def clear(self, memory_only: bool = False) -> int:
        """
        清空缓存。

        Parameters
        ----------
        memory_only : bool, optional
            是否只清空内存缓存，默认False（同时清空磁盘）。

        Returns
        -------
        int
            清空的缓存项数量。
        """
        count = len(self._memory_cache)
        self._memory_cache.clear()

        if not memory_only:
            # 清空磁盘缓存
            if self.cache_dir.exists():
                for item in self.cache_dir.iterdir():
                    if item.is_file():
                        item.unlink()

        logger.info(f"Cleared cache: {count} memory items, disk={not memory_only}")
        return count

    def exists(self, key: str, check_expired: bool = True) -> bool:
        """
        检查缓存键是否存在。

        Parameters
        ----------
        key : str
            缓存键。
        check_expired : bool, optional
            是否检查过期，默认True。

        Returns
        -------
        bool
            是否存在（且未过期）。
        """
        # 检查内存
        if key in self._memory_cache:
            if not check_expired or not self._is_expired(self._memory_cache[key]):
                return True

        # 检查磁盘
        filename = self._get_cache_filename(key)
        filepath = self.cache_dir / filename

        if not filepath.exists():
            return False

        if check_expired:
            meta_filepath = self.cache_dir / f"{filename}.meta"
            if meta_filepath.exists():
                try:
                    with open(meta_filepath, "r") as f:
                        meta = json.load(f)
                    expire_at = datetime.fromisoformat(meta["expire_at"])
                    return datetime.now() <= expire_at
                except:
                    return False

        return True

    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息。

        Returns
        -------
        dict
            包含命中数、未命中数、保存数等统计信息。
        """
        stats = self._stats.copy()
        total = stats["hits"] + stats["misses"]
        if total > 0:
            stats["hit_rate"] = stats["hits"] / total
        else:
            stats["hit_rate"] = 0.0
        stats["memory_items"] = len(self._memory_cache)
        stats["disk_items"] = len(list(self.cache_dir.glob("cache_*.pkl")))
        return stats

    def cleanup_expired(self) -> int:
        """
        清理所有过期的磁盘缓存。

        Returns
        -------
        int
            清理的过期项数量。
        """
        count = 0

        for filepath in self.cache_dir.glob("cache_*.pkl"):
            meta_filepath = Path(str(filepath) + ".meta")
            if not meta_filepath.exists():
                continue

            try:
                with open(meta_filepath, "r") as f:
                    meta = json.load(f)
                expire_at = datetime.fromisoformat(meta["expire_at"])

                if datetime.now() > expire_at:
                    filepath.unlink()
                    meta_filepath.unlink()
                    count += 1
            except Exception as e:
                logger.warning(f"Failed to check cache file {filepath}: {e}")

        # 清理内存中的过期项
        expired_keys = [
            k for k, v in self._memory_cache.items()
            if self._is_expired(v)
        ]
        for k in expired_keys:
            del self._memory_cache[k]
            count += 1

        if count > 0:
            logger.info(f"Cleaned up {count} expired cache items")

        return count

    def get_cache_size(self) -> Dict[str, int]:
        """
        获取缓存大小统计。

        Returns
        -------
        dict
            内存和磁盘缓存的大小（字节）。
        """
        # 内存大小（估算）
        memory_size = 0
        for item in self._memory_cache.values():
            try:
                memory_size += len(pickle.dumps(item["data"]))
            except:
                pass

        # 磁盘大小
        disk_size = 0
        for item in self.cache_dir.iterdir():
            if item.is_file():
                disk_size += item.stat().st_size

        return {
            "memory_bytes": memory_size,
            "disk_bytes": disk_size,
            "memory_mb": round(memory_size / 1024 / 1024),
            "disk_mb": round(disk_size / 1024 / 1024),
        }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"DataCache(hits={stats['hits']}, misses={stats['misses']}, "
            f"hit_rate={stats['hit_rate']:.2%}, "
            f"memory={stats['memory_items']}, disk={stats['disk_items']})"
        )
