# -*- coding: utf-8 -*-
"""
因子库管理模块 (Factor Library)

提供因子的注册、查询、文档管理等功能。

主要功能：
- 因子注册和注销
- 因子查询和获取
- 因子分类管理
- 因子文档生成
- 因子依赖管理
"""

import logging
from typing import Optional, Dict, Any, List, Union, Type, Callable
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FactorInfo:
    """因子信息数据类"""
    name: str
    factor_class: Type  # 因子类
    category: str       # 因子类别
    description: str    # 因子描述
    required_cols: List[str] = field(default_factory=list)
    params_schema: Dict[str, Any] = field(default_factory=dict)
    author: str = ""
    version: str = "1.0.0"
    created_at: str = ""
    tags: List[str] = field(default_factory=list)
    documentation: str = ""
    examples: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class FactorLibrary:
    """
    因子库管理器
    
    提供因子的注册、查询和管理功能。
    
    Parameters
    ----------
    name : str, optional
        因子库名称
    
    Examples
    --------
    >>> library = FactorLibrary(name="my_library")
    >>> library.register_factor(MAFactor, category="moving_average")
    >>> library.get_factor("MA")
    """

    def __init__(self, name: str = "default"):
        """
        初始化因子库。
        
        Parameters
        ----------
        name : str
            因子库名称
        """
        self.name = name
        self.factors: Dict[str, FactorInfo] = {}
        self.categories: Dict[str, List[str]] = {}
        self.factor_classes: Dict[str, Type] = {}
        
        # 注册默认因子
        self._register_default_factors()

    def _register_default_factors(self) -> None:
        """注册默认因子"""
        try:
            from .technical_factors import (
                MAFactor, EMAFactor, SMAFactor, WMAFactor, VWMAFactor,
                MACDFactor, RSIFactor, KDJFactor, ATRFactor, BBandsFactor,
                OBVFactor, ADXFactor, CCRFactor, ROCFactor,
                MACDDiffFactor, MACDDeaFactor,
                KFactor, DFactor,
            )
            
            # 技术因子
            self.register_factor(MAFactor, category="技术因子")
            self.register_factor(EMAFactor, category="技术因子")
            self.register_factor(SMAFactor, category="技术因子")
            self.register_factor(WMAFactor, category="技术因子")
            self.register_factor(VWMAFactor, category="技术因子")
            self.register_factor(MACDFactor, category="技术因子")
            self.register_factor(MACDDiffFactor, category="技术因子")
            self.register_factor(MACDDeaFactor, category="技术因子")
            self.register_factor(RSIFactor, category="技术因子")
            self.register_factor(KDJFactor, category="技术因子")
            self.register_factor(KFactor, category="技术因子")
            self.register_factor(DFactor, category="技术因子")
            self.register_factor(ATRFactor, category="技术因子")
            self.register_factor(BBandsFactor, category="技术因子")
            self.register_factor(OBVFactor, category="技术因子")
            self.register_factor(ADXFactor, category="技术因子")
            self.register_factor(CCRFactor, category="技术因子")
            self.register_factor(ROCFactor, category="技术因子")
            
        except ImportError as e:
            logger.warning(f"Could not import default factors: {e}")

    def register_factor(
        self,
        factor_class: Type,
        name: str = None,
        category: str = "未分类",
        description: str = "",
        tags: List[str] = None,
    ) -> FactorInfo:
        """
        注册因子。
        
        Parameters
        ----------
        factor_class : Type
            因子类（必须继承自FactorBase）
        name : str, optional
            因子名称，默认使用类的name属性
        category : str, optional
            因子类别
        description : str, optional
            因子描述
        tags : list, optional
            因子标签
        
        Returns
        -------
        FactorInfo
            因子信息
        
        Raises
        ------
        TypeError
            因子类不合法时抛出
        """
        # 获取因子名称
        factor_name = name or getattr(factor_class, "name", None)
        if not factor_name:
            factor_name = factor_class.__name__
        
        # 获取必需列
        required_cols = getattr(factor_class, "required_cols", ["close"])
        
        # 创建因子信息
        info = FactorInfo(
            name=factor_name,
            factor_class=factor_class,
            category=category,
            description=description or f"{factor_name}因子",
            required_cols=required_cols,
            tags=tags or [],
        )
        
        # 存储
        self.factors[factor_name] = info
        self.factor_classes[factor_name] = factor_class
        
        # 更新类别索引
        if category not in self.categories:
            self.categories[category] = []
        if factor_name not in self.categories[category]:
            self.categories[category].append(factor_name)
        
        logger.info(f"Registered factor: {factor_name} (category: {category})")
        
        return info

    def unregister_factor(self, name: str) -> bool:
        """
        注销因子。
        
        Parameters
        ----------
        name : str
            因子名称
        
        Returns
        -------
        bool
            是否成功注销
        """
        if name not in self.factors:
            logger.warning(f"Factor '{name}' not found in library.")
            return False
        
        info = self.factors[name]
        
        # 从类别中移除
        category = info.category
        if category in self.categories and name in self.categories[category]:
            self.categories[category].remove(name)
        
        # 删除
        del self.factors[name]
        if name in self.factor_classes:
            del self.factor_classes[name]
        
        logger.info(f"Unregistered factor: {name}")
        
        return True

    def get_factor(
        self,
        name: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        获取因子实例。
        
        Parameters
        ----------
        name : str
            因子名称
        params : dict, optional
            因子参数
        
        Returns
        -------
        FactorBase
            因子实例
        
        Raises
        ------
        ValueError
            因子不存在时抛出
        """
        if name not in self.factor_classes:
            raise ValueError(f"Factor '{name}' not found in library.")
        
        factor_class = self.factor_classes[name]
        
        return factor_class(params)

    def get_factor_info(self, name: str) -> Optional[FactorInfo]:
        """
        获取因子信息。
        
        Parameters
        ----------
        name : str
            因子名称
        
        Returns
        -------
        FactorInfo or None
            因子信息
        """
        return self.factors.get(name)

    def list_factors(
        self,
        category: str = None,
        tag: str = None,
    ) -> List[str]:
        """
        列出因子。
        
        Parameters
        ----------
        category : str, optional
            按类别筛选
        tag : str, optional
            按标签筛选
        
        Returns
        -------
        list
            因子名称列表
        """
        if category:
            return self.categories.get(category, [])
        
        if tag:
            result = []
            for name, info in self.factors.items():
                if tag in info.tags:
                    result.append(name)
            return result
        
        return list(self.factors.keys())

    def list_categories(self) -> List[str]:
        """
        列出所有因子类别。
        
        Returns
        -------
        list
            类别名称列表
        """
        return list(self.categories.keys())

    def search_factors(self, keyword: str) -> List[str]:
        """
        搜索因子。
        
        Parameters
        ----------
        keyword : str
            关键词
        
        Returns
        -------
        list
            匹配的因子名称列表
        """
        keyword = keyword.lower()
        result = []
        
        for name, info in self.factors.items():
            if keyword in name.lower():
                result.append(name)
            elif keyword in info.description.lower():
                result.append(name)
            elif keyword in info.category.lower():
                result.append(name)
        
        return result

    def get_factors_by_category(self, category: str) -> Dict[str, FactorInfo]:
        """
        按类别获取因子信息。
        
        Parameters
        ----------
        category : str
            类别名称
        
        Returns
        -------
        dict
            因子名称到信息的字典
        """
        factor_names = self.categories.get(category, [])
        return {name: self.factors[name] for name in factor_names}

    def generate_documentation(self) -> str:
        """
        生成因子库文档。
        
        Returns
        -------
        str
            Markdown格式的文档
        """
        lines = [
            f"# {self.name} 因子库",
            "",
            f"共注册 {len(self.factors)} 个因子，分为 {len(self.categories)} 个类别。",
            "",
            "## 目录",
            "",
        ]
        
        # 目录
        for category in sorted(self.categories.keys()):
            lines.append(f"- [{category}](#{category})")
        
        lines.append("")
        
        # 各类别
        for category in sorted(self.categories.keys()):
            lines.append(f"## {category}")
            lines.append("")
            
            factor_names = sorted(self.categories[category])
            for name in factor_names:
                info = self.factors[name]
                lines.append(f"### {name}")
                lines.append("")
                lines.append(f"**描述**: {info.description}")
                lines.append("")
                lines.append(f"**必需列**: {', '.join(info.required_cols)}")
                lines.append("")
                
                if info.tags:
                    lines.append(f"**标签**: {', '.join(info.tags)}")
                    lines.append("")
                
                lines.append("---")
                lines.append("")
        
        return "\n".join(lines)

    def export_config(self) -> Dict[str, Any]:
        """
        导出因子库配置。
        
        Returns
        -------
        dict
            配置字典
        """
        config = {
            "name": self.name,
            "factors": {},
            "categories": self.categories.copy(),
        }
        
        for name, info in self.factors.items():
            config["factors"][name] = {
                "category": info.category,
                "description": info.description,
                "required_cols": info.required_cols,
                "tags": info.tags,
                "version": info.version,
            }
        
        return config

    def import_config(self, config: Dict[str, Any]) -> None:
        """
        从配置导入因子库。
        
        Parameters
        ----------
        config : dict
            配置字典
        """
        self.name = config.get("name", self.name)
        
        for name, factor_config in config.get("factors", {}).items():
            if name in self.factor_classes:
                self.register_factor(
                    self.factor_classes[name],
                    name=name,
                    category=factor_config.get("category", "未分类"),
                    description=factor_config.get("description", ""),
                    tags=factor_config.get("tags", []),
                )

    def __repr__(self) -> str:
        return f"FactorLibrary(name='{self.name}', factors={len(self.factors)}, categories={len(self.categories)})"


class FactorCalculator:
    """
    因子计算器
    
    批量计算因子值。
    
    Parameters
    ----------
    library : FactorLibrary, optional
        因子库实例
    """
    
    def __init__(self, library: Optional[FactorLibrary] = None):
        self.library = library or FactorLibrary()
        self.calculators = {}

    def calculate(
        self,
        data: pd.DataFrame,
        factor_names: List[str],
        params: Optional[Dict[str, Dict[str, Any]]] = None,
        drop_na: bool = True,
    ) -> pd.DataFrame:
        """
        批量计算因子。
        
        Parameters
        ----------
        data : pd.DataFrame
            原始数据
        factor_names : list
            要计算的因子名称列表
        params : dict, optional
            各因子的参数字典
        drop_na : bool, optional
            是否删除包含NA的行
        
        Returns
        -------
        pd.DataFrame
            因子值DataFrame
        """
        params = params or {}
        results = pd.DataFrame(index=data.index)
        
        for name in factor_names:
            try:
                factor_params = params.get(name, {})
                factor = self.library.get_factor(name, factor_params)
                
                result = factor.calculate(data)
                results[name] = result
                
            except Exception as e:
                logger.error(f"Error calculating factor '{name}': {e}")
                results[name] = np.nan
        
        if drop_na:
            results = results.dropna()
        
        return results

    def calculate_all(
        self,
        data: pd.DataFrame,
        category: str = None,
        drop_na: bool = True,
    ) -> pd.DataFrame:
        """
        计算所有因子（或某类别所有因子）。
        
        Parameters
        ----------
        data : pd.DataFrame
            原始数据
        category : str, optional
            类别筛选
        drop_na : bool, optional
            是否删除包含NA的行
        
        Returns
        -------
        pd.DataFrame
            因子值DataFrame
        """
        factor_names = self.library.list_factors(category=category)
        return self.calculate(data, factor_names, drop_na=drop_na)


# 全局默认因子库
_default_library = None


def get_default_library() -> FactorLibrary:
    """
    获取默认因子库（单例）。
    
    Returns
    -------
    FactorLibrary
        默认因子库
    """
    global _default_library
    if _default_library is None:
        _default_library = FactorLibrary(name="default")
    return _default_library


def register_default_factor(
    factor_class: Type,
    name: str = None,
    category: str = "未分类",
    description: str = "",
    tags: List[str] = None,
) -> FactorInfo:
    """
    注册到默认因子库。
    
    Parameters
    ----------
    factor_class : Type
        因子类
    name : str, optional
        因子名称
    category : str, optional
        类别
    description : str, optional
        描述
    tags : list, optional
        标签
    
    Returns
    -------
    FactorInfo
        因子信息
    """
    library = get_default_library()
    return library.register_factor(factor_class, name, category, description, tags)


def get_factor(name: str, params: Dict[str, Any] = None) -> Any:
    """
    从默认因子库获取因子。
    
    Parameters
    ----------
    name : str
        因子名称
    params : dict, optional
        参数
    
    Returns
    -------
    FactorBase
        因子实例
    """
    library = get_default_library()
    return library.get_factor(name, params)


def list_all_factors() -> List[str]:
    """列出默认因子库中所有因子"""
    library = get_default_library()
    return library.list_factors()


# ============================================================
# 因子工厂函数
# ============================================================

def create_factor(
    factor_name: str,
    params: Optional[Dict[str, Any]] = None,
    library: Optional[FactorLibrary] = None,
) -> Any:
    """
    因子工厂函数。
    
    Parameters
    ----------
    factor_name : str
        因子名称
    params : dict, optional
        因子参数
    library : FactorLibrary, optional
        因子库
    
    Returns
    -------
    FactorBase
        因子实例
    """
    lib = library or get_default_library()
    return lib.get_factor(factor_name, params)


def create_factors(
    factor_configs: List[Dict[str, Any]],
    library: Optional[FactorLibrary] = None,
) -> Dict[str, Any]:
    """
    批量创建因子。
    
    Parameters
    ----------
    factor_configs : list
        因子配置列表，每项包含 name 和可选的 params
    library : FactorLibrary, optional
        因子库
    
    Returns
    -------
    dict
        因子名称到实例的字典
    """
    lib = library or get_default_library()
    result = {}
    
    for config in factor_configs:
        name = config.get("name")
        params = config.get("params", {})
        
        if name:
            result[name] = lib.get_factor(name, params)
    
    return result


# ============================================================
# 预定义因子集
# ============================================================

TECHNICAL_FACTORS = [
    "MA", "EMA", "SMA", "WMA", "VWMA",
    "MACD", "MACD_DIFF", "MACD_DEA",
    "RSI", "KDJ", "K", "D",
    "ATR", "BBANDS",
    "OBV", "ADX", "CCI", "ROC",
]

FUNDAMENTAL_FACTORS = [
    "PE", "PB", "PS", "PCF",
    "ROE", "ROA", "ROIC",
    "GROSS_MARGIN", "NET_MARGIN",
    "PROFIT_GROWTH", "REVENUE_GROWTH",
    "DEBT_TO_ASSET", "CURRENT_RATIO",
]

MOMENTUM_FACTORS = [
    "RSI", "KDJ", "ROC", "MOMENTUM",
    "CCI", "WILLIAMS_R", "STOCH",
]

TREND_FACTORS = [
    "MA", "EMA", "MACD", "ADX", "AROON",
]

VOLATILITY_FACTORS = [
    "ATR", "BBANDS", "STD", "HIST_VOLATILITY",
]

VOLUME_FACTORS = [
    "OBV", "VROC", "MFI", "CMF", "VOL_MA",
]
