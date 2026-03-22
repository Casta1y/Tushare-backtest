# -*- coding: utf-8 -*-
"""
基本面因子模块 (Fundamental Factors)

提供20+个基本面分析因子，包括：
- 估值类因子 (PE, PB, PS, PCF, EV/EBITDA等)
- 盈利能力因子 (ROE, ROA, ROIC, 毛利率, 净利率等)
- 成长类因子 (净利润增长率, 营收增长率, 毛利增长率等)
- 财务结构因子 (资产负债率, 流动比率, 速动比率等)
- 运营效率因子 (存货周转率, 应收账款周转率, 总资产周转率等)
- 股息类因子 (股息率, 派息率等)

所有因子继承自FactorBase基类。
"""

import logging
from typing import Optional, Dict, Any, List, Union
import warnings

import numpy as np
import pandas as pd

from .factor_base import FactorBase, CompositeFactor

logger = logging.getLogger(__name__)


# ============================================================
# 估值类因子 (Valuation Factors)
# ============================================================

class PEFactor(FactorBase):
    """
    PE (Price to Earnings Ratio)
    
    市盈率 = 股价 / 每股收益
    """
    name = "PE"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        # 需要财务数据
        if "eps" in df.columns:
            return df["close"] / df["eps"]
        return pd.Series(np.nan, index=df.index)


class PBFactor(FactorBase):
    """
    PB (Price to Book Ratio)
    
    市净率 = 股价 / 每股净资产
    """
    name = "PB"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        if "bps" in df.columns:
            return df["close"] / df["bps"]
        return pd.Series(np.nan, index=df.index)


class PSFactor(FactorBase):
    """
    PS (Price to Sales Ratio)
    
    市销率 = 股价 / 每股营收
    """
    name = "PS"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        if "sales_per_share" in df.columns:
            return df["close"] / df["sales_per_share"]
        return pd.Series(np.nan, index=df.index)


class PCFFactor(FactorBase):
    """
    PCF (Price to Cash Flow Ratio)
    
    市现率 = 股价 / 每股现金流
    """
    name = "PCF"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        if "cfps" in df.columns:
            return df["close"] / df["cfps"]
        return pd.Series(np.nan, index=df.index)


class EVEBITDAMFactor(FactorBase):
    """
    EV/EBITDA (Enterprise Value to EBITDA)
    
    企业价值倍数
    """
    name = "EV_EBITDA"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        if "market_cap" in df.columns and "ebitda" in df.columns:
            return df["market_cap"] / df["ebitda"]
        return pd.Series(np.nan, index=df.index)


class PEGFactor(FactorBase):
    """
    PEG (Price/Earnings to Growth Ratio)
    
    市盈率相对盈利增长比率 = PE / 盈利增长率
    """
    name = "PEG"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        if "pe" in df.columns and "profit_growth" in df.columns:
            pe = df["pe"]
            growth = df["profit_growth"]
            return pe / growth
        return pd.Series(np.nan, index=df.index)


class PEYearlyFactor(FactorBase):
    """
    PE TTM (滚动市盈率)
    
    基于过去12个月净利润计算的市盈率。
    """
    name = "PE_TTM"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        if "eps_ttm" in df.columns:
            return df["close"] / df["eps_ttm"]
        return pd.Series(np.nan, index=df.index)


class DividendYieldFactor(FactorBase):
    """
    Dividend Yield (股息率)
    
    股息率 = 每股股息 / 股价
    """
    name = "DIVIDEND_YIELD"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        if "dividend_per_share" in df.columns:
            return df["dividend_per_share"] / df["close"] * 100
        return pd.Series(np.nan, index=df.index)


# ============================================================
# 盈利能力因子 (Profitability Factors)
# ============================================================

class ROEFactor(FactorBase):
    """
    ROE (Return on Equity)
    
    净资产收益率 = 净利润 / 股东权益
    """
    name = "ROE"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        if "roe" in df.columns:
            return df["roe"] * 100  # 转为百分比
        return pd.Series(np.nan, index=df.index)


class ROAFactor(FactorBase):
    """
    ROA (Return on Assets)
    
    总资产收益率 = 净利润 / 总资产
    """
    name = "ROA"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        if "roa" in df.columns:
            return df["roa"] * 100
        return pd.Series(np.nan, index=df.index)


class ROICFactor(FactorBase):
    """
    ROIC (Return on Invested Capital)
    
    投入资本回报率 = NOPAT / 投入资本
    """
    name = "ROIC"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        if "roic" in df.columns:
            return df["roic"] * 100
        return pd.Series(np.nan, index=df.index)


class GrossMarginFactor(FactorBase):
    """
    Gross Margin (毛利率)
    
    毛利率 = (营收 - 成本) / 营收
    """
    name = "GROSS_MARGIN"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        if "gross_margin" in df.columns:
            return df["gross_margin"] * 100
        return pd.Series(np.nan, index=df.index)


class NetMarginFactor(FactorBase):
    """
    Net Profit Margin (净利率)
    
    净利率 = 净利润 / 营收
    """
    name = "NET_MARGIN"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        if "net_margin" in df.columns:
            return df["net_margin"] * 100
        return pd.Series(np.nan, index=df.index)


class OperatingMarginFactor(FactorBase):
    """
    Operating Margin (营业利润率)
    
    营业利润率 = 营业利润 / 营收
    """
    name = "OPERATING_MARGIN"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        if "operating_margin" in df.columns:
            return df["operating_margin"] * 100
        return pd.Series(np.nan, index=df.index)


class EBITDAMarginFactor(FactorBase):
    """
    EBITDA Margin (EBITDA利润率)
    
    EBITDA利润率 = EBITDA / 营收
    """
    name = "EBITDA_MARGIN"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        if "ebitda_margin" in df.columns:
            return df["ebitda_margin"] * 100
        return pd.Series(np.nan, index=df.index)


class EBITMarginFactor(FactorBase):
    """
    EBIT Margin (EBIT利润率)
    
    EBIT利润率 = EBIT / 营收
    """
    name = "EBIT_MARGIN"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        if "ebit_margin" in df.columns:
            return df["ebit_margin"] * 100
        return pd.Series(np.nan, index=df.index)


class ROEYearlyFactor(FactorBase):
    """
    ROE Yearly (年度净资产收益率)
    
    基于年度数据的ROE。
    """
    name = "ROE_YEARLY"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        if "roe_yearly" in df.columns:
            return df["roe_yearly"] * 100
        return pd.Series(np.nan, index=df.index)


# ============================================================
# 成长类因子 (Growth Factors)
# ============================================================

class ProfitGrowthFactor(FactorBase):
    """
    Net Profit Growth (净利润增长率)
    
    净利润增长率 = (本期净利润 - 上期净利润) / 上期净利润
    """
    name = "PROFIT_GROWTH"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        if "net_profit" in df.columns:
            return df["net_profit"].pct_change() * 100
        return pd.Series(np.nan, index=df.index)


class RevenueGrowthFactor(FactorBase):
    """
    Revenue Growth (营收增长率)
    
    营收增长率 = (本期营收 - 上期营收) / 上期营收
    """
    name = "REVENUE_GROWTH"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        if "revenue" in df.columns:
            return df["revenue"].pct_change() * 100
        return pd.Series(np.nan, index=df.index)


class GrossProfitGrowthFactor(FactorBase):
    """
    Gross Profit Growth (毛利增长率)
    
    毛利增长率 = (本期毛利 - 上期毛利) / 上期毛利
    """
    name = "GROSS_PROFIT_GROWTH"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        if "gross_profit" in df.columns:
            return df["gross_profit"].pct_change() * 100
        return pd.Series(np.nan, index=df.index)


class OperatingProfitGrowthFactor(FactorBase):
    """
    Operating Profit Growth (营业利润增长率)
    """
    name = "OPERATING_PROFIT_GROWTH"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        if "operating_profit" in df.columns:
            return df["operating_profit"].pct_change() * 100
        return pd.Series(np.nan, index=df.index)


class EPSGrowthFactor(FactorBase):
    """
    EPS Growth (每股收益增长率)
    
    每股收益增长率 = (本期EPS - 上期EPS) / 上期EPS
    """
    name = "EPS_GROWTH"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        if "eps" in df.columns:
            return df["eps"].pct_change() * 100
        return pd.Series(np.nan, index=df.index)


class BPSGrowthFactor(FactorBase):
    """
    BPS Growth (每股净资产增长率)
    """
    name = "BPS_GROWTH"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        if "bps" in df.columns:
            return df["bps"].pct_change() * 100
        return pd.Series(np.nan, index=df.index)


class RevenueGrowth3YFactor(FactorBase):
    """
    3-Year Revenue CAGR (营收3年复合增长率)
    """
    name = "REVENUE_CAGR_3Y"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        if "revenue" in df.columns:
            # 3年复合增长率
            return ((df["revenue"] / df["revenue"].shift(3)) ** (1/3) - 1) * 100
        return pd.Series(np.nan, index=df.index)


class ProfitGrowth3YFactor(FactorBase):
    """
    3-Year Net Profit CAGR (净利润3年复合增长率)
    """
    name = "PROFIT_CAGR_3Y"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        if "net_profit" in df.columns:
            return ((df["net_profit"] / df["net_profit"].shift(3)) ** (1/3) - 1) * 100
        return pd.Series(np.nan, index=df.index)


# ============================================================
# 财务结构因子 (Financial Structure Factors)
# ============================================================

class DebtToAssetFactor(FactorBase):
    """
    Debt to Asset Ratio (资产负债率)
    
    资产负债率 = 总负债 / 总资产
    """
    name = "DEBT_TO_ASSET"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        if "debt_to_asset" in df.columns:
            return df["debt_to_asset"] * 100
        return pd.Series(np.nan, index=df.index)


class DebtToEquityFactor(FactorBase):
    """
    Debt to Equity Ratio (产权比率)
    
    产权比率 = 总负债 / 股东权益
    """
    name = "DEBT_TO_EQUITY"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        if "debt_to_equity" in df.columns:
            return df["debt_to_equity"]
        return pd.Series(np.nan, index=df.index)


class CurrentRatioFactor(FactorBase):
    """
    Current Ratio (流动比率)
    
    流动比率 = 流动资产 / 流动负债
    """
    name = "CURRENT_RATIO"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        if "current_ratio" in df.columns:
            return df["current_ratio"]
        return pd.Series(np.nan, index=df.index)


class QuickRatioFactor(FactorBase):
    """
    Quick Ratio (速动比率)
    
    速动比率 = (流动资产 - 存货) / 流动负债
    """
    name = "QUICK_RATIO"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        if "quick_ratio" in df.columns:
            return df["quick_ratio"]
        return pd.Series(np.nan, index=df.index)


class CashRatioFactor(FactorBase):
    """
    Cash Ratio (现金比率)
    
    现金比率 = 现金及现金等价物 / 流动负债
    """
    name = "CASH_RATIO"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        if "cash_ratio" in df.columns:
            return df["cash_ratio"]
        return pd.Series(np.nan, index=df.index)


class EquityToAssetFactor(FactorBase):
    """
    Equity to Asset Ratio (股东权益比率)
    
    股东权益比率 = 股东权益 / 总资产
    """
    name = "EQUITY_TO_ASSET"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        if "equity_to_asset" in df.columns:
            return df["equity_to_asset"] * 100
        return pd.Series(np.nan, index=df.index)


# ============================================================
# 运营效率因子 (Operational Efficiency Factors)
# ============================================================

class InventoryTurnoverFactor(FactorBase):
    """
    Inventory Turnover (存货周转率)
    
    存货周转率 = 营业成本 / 平均存货
    """
    name = "INVENTORY_TURNOVER"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        if "inventory_turnover" in df.columns:
            return df["inventory_turnover"]
        return pd.Series(np.nan, index=df.index)


class InventoryDaysFactor(FactorBase):
    """
    Inventory Days (存货周转天数)
    
    存货周转天数 = 365 / 存货周转率
    """
    name = "INVENTORY_DAYS"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        if "inventory_turnover" in df.columns:
            return 365 / df["inventory_turnover"]
        return pd.Series(np.nan, index=df.index)


class ReceivableTurnoverFactor(FactorBase):
    """
    Receivable Turnover (应收账款周转率)
    
    应收账款周转率 = 营收 / 平均应收账款
    """
    name = "RECEIVABLE_TURNOVER"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        if "receivable_turnover" in df.columns:
            return df["receivable_turnover"]
        return pd.Series(np.nan, index=df.index)


class ReceivableDaysFactor(FactorBase):
    """
    Receivable Days (应收账款周转天数)
    
    应收账款周转天数 = 365 / 应收账款周转率
    """
    name = "RECEIVABLE_DAYS"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        if "receivable_turnover" in df.columns:
            return 365 / df["receivable_turnover"]
        return pd.Series(np.nan, index=df.index)


class AssetTurnoverFactor(FactorBase):
    """
    Total Asset Turnover (总资产周转率)
    
    总资产周转率 = 营收 / 平均总资产
    """
    name = "ASSET_TURNOVER"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        if "asset_turnover" in df.columns:
            return df["asset_turnover"]
        return pd.Series(np.nan, index=df.index)


class EquityTurnoverFactor(FactorBase):
    """
    Equity Turnover (股东权益周转率)
    
    股东权益周转率 = 营收 / 平均股东权益
    """
    name = "EQUITY_TURNOVER"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        if "equity_turnover" in df.columns:
            return df["equity_turnover"]
        return pd.Series(np.nan, index=df.index)


# ============================================================
# 股息类因子 (Dividend Factors)
# ============================================================

class DividendPayoutRatioFactor(FactorBase):
    """
    Dividend Payout Ratio (派息率/分红率)
    
    派息率 = 每股股息 / 每股收益
    """
    name = "DIVIDEND_PAYOUT_RATIO"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        if "dividend_per_share" in df.columns and "eps" in df.columns:
            return (df["dividend_per_share"] / df["eps"]) * 100
        return pd.Series(np.nan, index=df.index)


class DividendCoverFactor(FactorBase):
    """
    Dividend Cover (股息保障倍数)
    
    股息保障倍数 = 每股收益 / 每股股息
    """
    name = "DIVIDEND_COVER"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        if "dividend_per_share" in df.columns and "eps" in df.columns:
            return df["eps"] / df["dividend_per_share"]
        return pd.Series(np.nan, index=df.index)


class CashDividendYieldFactor(FactorBase):
    """
    Cash Dividend Yield (现金股息率)
    
    现金股息率 = 每股现金分红 / 股价
    """
    name = "CASH_DIVIDEND_YIELD"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        if "cash_dividend" in df.columns:
            return (df["cash_dividend"] / df["close"]) * 100
        return pd.Series(np.nan, index=df.index)


# ============================================================
# 其他基本面因子
# ============================================================

class OperatingIncomeFactor(FactorBase):
    """
    Operating Income (营业收入)
    
    衡量企业规模
    """
    name = "OPERATING_INCOME"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        if "operating_income" in df.columns:
            return df["operating_income"]
        return pd.Series(np.nan, index=df.index)


class TotalAssetsFactor(FactorBase):
    """
    Total Assets (总资产)
    """
    name = "TOTAL_ASSETS"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        if "total_assets" in df.columns:
            return df["total_assets"]
        return pd.Series(np.nan, index=df.index)


class TotalEquityFactor(FactorBase):
    """
    Total Equity (股东权益)
    """
    name = "TOTAL_EQUITY"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        if "total_equity" in df.columns:
            return df["total_equity"]
        return pd.Series(np.nan, index=df.index)


class MarketCapFactor(FactorBase):
    """
    Market Capitalization (市值)
    """
    name = "MARKET_CAP"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        if "market_cap" in df.columns:
            return df["market_cap"]
        return pd.Series(np.nan, index=df.index)


class CashToAssetFactor(FactorBase):
    """
    Cash to Asset Ratio (现金资产比)
    
    现金资产比 = 现金及现金等价物 / 总资产
    """
    name = "CASH_TO_ASSET"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        if "cash" in df.columns and "total_assets" in df.columns:
            return (df["cash"] / df["total_assets"]) * 100
        return pd.Series(np.nan, index=df.index)


class WorkingCapitalFactor(FactorBase):
    """
    Working Capital (营运资本)
    
    营运资本 = 流动资产 - 流动负债
    """
    name = "WORKING_CAPITAL"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        if "working_capital" in df.columns:
            return df["working_capital"]
        return pd.Series(np.nan, index=df.index)


# ============================================================
# 综合因子 (Composite Fundamental Factors)
# ============================================================

class AltmanZScoreFactor(FactorBase):
    """
    Altman Z-Score ( Altman Z分数)
    
    Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
    
    X1 = 营运资本/总资产
    X2 = 留存收益/总资产
    X3 = EBIT/总资产
    X4 = 股权市场价值/总负债
    X5 = 销售额/总资产
    """
    name = "ALTMAN_Z_SCORE"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        # 需要多个财务指标
        if all(col in df.columns for col in ["working_capital", "total_assets", 
                                               "retained_earnings", "ebit", "market_cap", 
                                               "total_liability", "revenue"]):
            x1 = df["working_capital"] / df["total_assets"]
            x2 = df["retained_earnings"] / df["total_assets"]
            x3 = df["ebit"] / df["total_assets"]
            x4 = df["market_cap"] / df["total_liability"]
            x5 = df["revenue"] / df["total_assets"]
            return 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5
        return pd.Series(np.nan, index=df.index)


class PiotroskiFScoreFactor(FactorBase):
    """
    Piotroski F-Score (皮奥罗特斯基F分数)
    
    9个财务指标打分，用于判断公司基本面质量
    """
    name = "PIOTROSKI_F_SCORE"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        # 简化的Piotroski分数
        score = pd.Series(0, index=df.index)
        
        if "roa" in df.columns:
            score += (df["roa"] > 0).astype(int)
        
        if "operating_cash_flow" in df.columns:
            score += (df["operating_cash_flow"] > 0).astype(int)
        
        if "roe" in df.columns:
            score += (df["roe"] > df["roe"].shift(1)).astype(int)
        
        if "gross_margin" in df.columns:
            score += (df["gross_margin"] > df["gross_margin"].shift(1)).astype(int)
        
        if "asset_turnover" in df.columns:
            score += (df["asset_turnover"] > df["asset_turnover"].shift(1)).astype(int)
        
        if "current_ratio" in df.columns:
            score += (df["current_ratio"] > df["current_ratio"].shift(1)).astype(int)
        
        if "debt_to_asset" in df.columns:
            score += (df["debt_to_asset"] < df["debt_to_asset"].shift(1)).astype(int)
        
        if "gross_margin" in df.columns:
            score += (df["gross_margin"] > 0).astype(int)
        
        if "shares" in df.columns:
            score += (df["shares"] <= df["shares"].shift(1)).astype(int)
        
        return score
