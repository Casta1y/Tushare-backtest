# -*- coding: utf-8 -*-
"""
技术因子模块 (Technical Factors)

提供50+个技术分析因子，包括：
- 移动平均类因子 (MA, EMA, SMA, WMA, VWMA, HMA, ALMA等)
- 趋势类因子 (MACD, ADX, Aroon, Ichimoku等)
- 动量类因子 (RSI, KDJ, ROC, CCI, Williams %R, Stochastic等)
- 波动率类因子 (ATR, Bollinger Bands, Keltner Channel, Donchian Channel等)
- 成交量类因子 (OBV, VROC, VWAP, CMF, MFI等)
- 周期类因子 (Hilbert Transform, Sine Wave等)

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
# 移动平均类因子 (Moving Average Factors)
# ============================================================

class MAFactor(FactorBase):
    """
    简单移动平均 (Moving Average)
    
    计算N日简单移动平均。
    """
    name = "MA"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        period = self.params.get("period", 5)
        return df["close"].rolling(window=int(period)).mean()


class EMAFactor(FactorBase):
    """
    指数移动平均 (Exponential Moving Average)
    
    计算N日指数移动平均。
    """
    name = "EMA"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        period = self.params.get("period", 12)
        return df["close"].ewm(span=int(period), adjust=False).mean()


class SMAFactor(FactorBase):
    """
    简单移动平均 (Simple Moving Average)
    
    与MA相同，保留作为别名。
    """
    name = "SMA"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        period = self.params.get("period", 10)
        return df["close"].rolling(window=int(period)).mean()


class WMAFactor(FactorBase):
    """
    加权移动平均 (Weighted Moving Average)
    
    计算N日加权移动平均，权重线性递减。
    """
    name = "WMA"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        period = int(self.params.get("period", 10))
        weights = np.arange(1, period + 1)
        return df["close"].rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )


class VWMAFactor(FactorBase):
    """
    成交量加权移动平均 (Volume Weighted Moving Average)
    
    根据成交量加权的移动平均。
    """
    name = "VWMA"
    required_cols = ["close", "vol"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        period = int(self.params.get("period", 20))
        close = df["close"]
        vol = df["vol"]
        
        def vwma_calc(x):
            if len(x) < period:
                return np.nan
            return np.sum(x * vol.iloc[x.index[0]:x.index[0]+len(x)]) / np.sum(vol.iloc[x.index[0]:x.index[0]+len(x)])
        
        return close.rolling(window=period).apply(vwma_calc, raw=True)


class VWAPFactor(FactorBase):
    """
    成交量加权平均价格 (Volume Weighted Average Price)
    
    典型价格 = (High + Low + Close) / 3
    VWAP = Σ(典型价格 × 成交量) / Σ成交量
    """
    name = "VWAP"
    required_cols = ["high", "low", "close", "vol"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        cumvol = df["vol"].cumsum()
        cumtyp = (typical_price * df["vol"]).cumsum()
        return cumtyp / cumvol


class HMAFactor(FactorBase):
    """
    Hull移动平均 (Hull Moving Average)
    
    一种高性能的移动平均，减少滞后。
    """
    name = "HMA"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        period = int(self.params.get("period", 14))
        
        # WMA of half period
        half_period = int(period / 2)
        
        def wma_calc(x):
            if len(x) < 2:
                return np.nan
            w = np.arange(1, len(x) + 1)
            return np.dot(x, w) / w.sum()
        
        wma_half = df["close"].rolling(window=half_period).apply(wma_calc, raw=True)
        
        # WMA of full period
        wma_full = df["close"].rolling(window=period).apply(wma_calc, raw=True)
        
        # HMA = WMA of (2*WMA_half - WMA_full)
        hma_raw = 2 * wma_half - wma_full
        sqrt_period = int(np.sqrt(period))
        hma = hma_raw.rolling(window=sqrt_period).apply(wma_calc, raw=True)
        
        return hma


class ALMAFactor(FactorBase):
    """
    Arnaud Legoux移动平均 (Arnaud Legoux Moving Average)
    
    可调滞后和平滑度的移动平均。
    """
    name = "ALMA"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        period = int(self.params.get("period", 20))
        offset = self.params.get("offset", 0.85)
        sigma = self.params.get("sigma", 6)
        
        window = df["close"].rolling(window=period)
        
        def alma_calc(x):
            if len(x) < period:
                return np.nan
            k = np.arange(period)
            w = np.exp(-((k - offset * (period - 1)) ** 2) / (2 * sigma ** 2))
            w = w / w.sum()
            return np.dot(x[-period:], w)
        
        return window.apply(alma_calc, raw=True)


class DMAFactor(FactorBase):
    """
    差值移动平均 (Difference Moving Average)
    
    当前价格与N日前移动平均的差值。
    """
    name = "DMA"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        period = int(self.params.get("period", 10))
        ma = df["close"].rolling(window=period).mean()
        return df["close"] - ma


class TMAFactor(FactorBase):
    """
    三角形移动平均 (Triangular Moving Average)
    
    对移动平均再进行移动平均。
    """
    name = "TMA"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        period = int(self.params.get("period", 20))
        first_ma = df["close"].rolling(window=int(period/2)).mean()
        return first_ma.rolling(window=int(period/2)).mean()


# ============================================================
# 趋势类因子 (Trend Factors)
# ============================================================

class MACDFactor(FactorBase):
    """
    MACD (Moving Average Convergence Divergence)
    
    - DIF = EMA12 - EMA26
    - DEA = EMA9(DIF)
    - MACD = 2 * (DIF - DEA)
    
    Parameters
    ----------
    fast_period : int, 默认12
        快速EMA周期
    slow_period : int, 默认26
        慢速EMA周期  
    signal_period : int, 默认9
        信号线周期
    """
    name = "MACD"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        fast = int(self.params.get("fast_period", 12))
        slow = int(self.params.get("slow_period", 26))
        signal = int(self.params.get("signal_period", 9))
        
        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=signal, adjust=False).mean()
        macd = 2 * (dif - dea)
        return macd


class MACDDiffFactor(FactorBase):
    """
    MACD DIF (MACD Fast Line)
    
    DIF = EMA12 - EMA26
    """
    name = "MACD_DIFF"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        fast = int(self.params.get("fast_period", 12))
        slow = int(self.params.get("slow_period", 26))
        
        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
        return ema_fast - ema_slow


class MACDDeaFactor(FactorBase):
    """
    MACD DEA (MACD Signal Line)
    
    DEA = EMA9(DIF)
    """
    name = "MACD_DEA"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        fast = int(self.params.get("fast_period", 12))
        slow = int(self.params.get("slow_period", 26))
        signal = int(self.params.get("signal_period", 9))
        
        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
        dif = ema_fast - ema_slow
        return dif.ewm(span=signal, adjust=False).mean()


class ADXFactor(FactorBase):
    """
    ADX (Average Directional Index)
    
    平均趋向指数，衡量趋势强度。
    """
    name = "ADX"
    required_cols = ["high", "low", "close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        period = int(self.params.get("period", 14))
        
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Directional Movement
        up = high - high.shift(1)
        down = low.shift(1) - low
        
        plus_dm = np.where((up > down) & (up > 0), up, 0)
        minus_dm = np.where((down > up) & (down > 0), down, 0)
        
        plus_dm = pd.Series(plus_dm, index=high.index)
        minus_dm = pd.Series(minus_dm, index=low.index)
        
        # Smoothed DM
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # DX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # ADX
        adx = dx.rolling(window=period).mean()
        return adx


class PlusDIFactor(FactorBase):
    """
    +DI (Positive Directional Indicator)
    
    上升方向指标。
    """
    name = "PLUS_DI"
    required_cols = ["high", "low", "close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        period = int(self.params.get("period", 14))
        
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        up = high - high.shift(1)
        down = low.shift(1) - low
        
        plus_dm = np.where((up > down) & (up > 0), up, 0)
        plus_dm = pd.Series(plus_dm, index=high.index)
        
        return 100 * (plus_dm.rolling(window=period).mean() / atr)


class MinusDIFactor(FactorBase):
    """
    -DI (Negative Directional Indicator)
    
    下降方向指标。
    """
    name = "MINUS_DI"
    required_cols = ["high", "low", "close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        period = int(self.params.get("period", 14))
        
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        up = high - high.shift(1)
        down = low.shift(1) - low
        
        minus_dm = np.where((down > up) & (down > 0), down, 0)
        minus_dm = pd.Series(minus_dm, index=low.index)
        
        return 100 * (minus_dm.rolling(window=period).mean() / atr)


class AroonFactor(FactorBase):
    """
    Aroon指标
    
    识别趋势变化和强度。
    """
    name = "AROON"
    required_cols = ["high", "low"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        period = int(self.params.get("period", 25))
        
        high = df["high"]
        low = df["low"]
        
        # Aroon Up
        def rolling_max_idx(x):
            if len(x) == 0:
                return np.nan
            return (len(x) - 1 - np.argmax(x[::-1])) / (len(x) - 1) * 100
        
        # Aroon Down
        def rolling_min_idx(x):
            if len(x) == 0:
                return np.nan
            return (len(x) - 1 - np.argmin(x[::-1])) / (len(x) - 1) * 100
        
        aroon_up = high.rolling(window=period).apply(rolling_max_idx, raw=True)
        aroon_down = low.rolling(window=period).apply(rolling_min_idx, raw=True)
        
        # Aroon Oscillator
        return aroon_up - aroon_down


class IchimokuFactor(FactorBase):
    """
    Ichimoku Cloud (一目均衡表)
    
    综合趋势指标。
    """
    name = "ICHIMOKU"
    required_cols = ["high", "low", "close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        # Tenkan-sen (Conversion Line)
        high = df["high"]
        low = df["low"]
        
        period9 = int(self.params.get("period9", 9))
        period26 = int(self.params.get("period26", 26))
        
        tenkan_sen = (high.rolling(window=period9).max() + low.rolling(window=period9).min()) / 2
        
        # Kijun-sen (Base Line)
        kijun_sen = (high.rolling(window=period26).max() + low.rolling(window=period26).min()) / 2
        
        # Tenkan-sen - Kijun-sen (Cloud)
        return tenkan_sen - kijun_sen


# ============================================================
# 动量类因子 (Momentum Factors)
# ============================================================

class RSIFactor(FactorBase):
    """
    RSI (Relative Strength Index)
    
    相对强弱指标，衡量价格变动的速度和幅度。
    """
    name = "RSI"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        period = int(self.params.get("period", 14))
        
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class KDJFactor(FactorBase):
    """
    KDJ随机指标
    
    - K = 2/3 * prev_K + 1/3 * RSV
    - D = 2/3 * prev_D + 1/3 * K
    - J = 3*K - 2*D
    """
    name = "KDJ"
    required_cols = ["high", "low", "close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        period = int(self.params.get("period", 9))
        k_period = int(self.params.get("k_period", 3))
        d_period = int(self.params.get("d_period", 3))
        
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        
        rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
        rsv = rsv.fillna(50)
        
        # K and D calculation
        k = rsv.ewm(alpha=1/k_period, adjust=False).mean()
        d = k.ewm(alpha=1/d_period, adjust=False).mean()
        j = 3 * k - 2 * d
        
        return j


class KFactor(FactorBase):
    """KDJ中的K线"""
    name = "K"
    required_cols = ["high", "low", "close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        period = int(self.params.get("period", 9))
        k_period = int(self.params.get("k_period", 3))
        
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        
        rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
        rsv = rsv.fillna(50)
        
        return rsv.ewm(alpha=1/k_period, adjust=False).mean()


class DFactor(FactorBase):
    """KDJ中的D线"""
    name = "D"
    required_cols = ["high", "low", "close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        period = int(self.params.get("period", 9))
        d_period = int(self.params.get("d_period", 3))
        
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        
        rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
        rsv = rsv.fillna(50)
        
        k = rsv.ewm(alpha=1/d_period, adjust=False).mean()
        return k.ewm(alpha=1/d_period, adjust=False).mean()


class ROCFactor(FactorBase):
    """
    ROC (Rate of Change)
    
    价格变化率。
    """
    name = "ROC"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        period = int(self.params.get("period", 12))
        
        roc = ((df["close"] - df["close"].shift(period)) / df["close"].shift(period)) * 100
        return roc


class CCRFactor(FactorBase):
    """
    CCI (Commodity Channel Index)
    
    商品通道指数。
    """
    name = "CCI"
    required_cols = ["high", "low", "close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        period = int(self.params.get("period", 14))
        
        tp = (df["high"] + df["low"] + df["close"]) / 3
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        
        cci = (tp - sma) / (0.015 * mad)
        return cci


class WilliamsRFactor(FactorBase):
    """
    Williams %R
    
    威廉指标。
    """
    name = "WILLIAMS_R"
    required_cols = ["high", "low", "close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        period = int(self.params.get("period", 14))
        
        highest_high = df["high"].rolling(window=period).max()
        lowest_low = df["low"].rolling(window=period).min()
        
        wr = -100 * (highest_high - df["close"]) / (highest_high - lowest_low)
        return wr


class StochasticFactor(FactorBase):
    """
    Stochastic Oscillator (随机震荡指标)
    
    %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
    %D = SMA(%K)
    """
    name = "STOCH"
    required_cols = ["high", "low", "close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        period = int(self.params.get("period", 14))
        smooth_k = int(self.params.get("smooth_k", 3))
        
        lowest_low = df["low"].rolling(window=period).min()
        highest_high = df["high"].rolling(window=period).max()
        
        k = 100 * (df["close"] - lowest_low) / (highest_high - lowest_low)
        return k.rolling(window=smooth_k).mean()


class MomentumFactor(FactorBase):
    """
    Momentum (动量指标)
    
    价格变化的动量。
    """
    name = "MOMENTUM"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        period = int(self.params.get("period", 10))
        
        return df["close"] - df["close"].shift(period)


class CMOFactor(FactorBase):
    """
    CMO (Chande Momentum Oscillator)
    
    钱德动量振荡器。
    """
    name = "CMO"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        period = int(self.params.get("period", 14))
        
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        sum_gain = gain.rolling(window=period).sum()
        sum_loss = loss.rolling(window=period).sum()
        
        cmo = 100 * (sum_gain - sum_loss) / (sum_gain + sum_loss)
        return cmo


class UltimateOscFactor(FactorBase):
    """
    Ultimate Oscillator
    
    最终振荡器。
    """
    name = "ULTIMATE_OSC"
    required_cols = ["high", "low", "close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        period1 = int(self.params.get("period1", 7))
        period2 = int(self.params.get("period2", 14))
        period3 = int(self.params.get("period3", 28))
        
        close = df["close"]
        low = df["low"]
        high = df["high"]
        
        prev_close = close.shift(1)
        
        # Buying Pressure
        bp = close - pd.concat([low, prev_close], axis=1).min(axis=1)
        
        # True Range
        tr = pd.concat([
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        ], axis=1).max(axis=1)
        
        avg1 = bp.rolling(window=period1).sum() / tr.rolling(window=period1).sum()
        avg2 = bp.rolling(window=period2).sum() / tr.rolling(window=period2).sum()
        avg3 = bp.rolling(window=period3).sum() / tr.rolling(window=period3).sum()
        
        uo = 100 * (4*avg1 + 2*avg2 + avg3) / (4 + 2 + 1)
        return uo


# ============================================================
# 波动率类因子 (Volatility Factors)
# ============================================================

class ATRFactor(FactorBase):
    """
    ATR (Average True Range)
    
    平均真实波幅，衡量波动率。
    """
    name = "ATR"
    required_cols = ["high", "low", "close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        period = int(self.params.get("period", 14))
        
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()


class BBandsFactor(FactorBase):
    """
    Bollinger Bands (布林带)
    
    - Upper Band = MA + K * STD
    - Middle Band = MA
    - Lower Band = MA - K * STD
    
    Parameters
    ----------
    period : int, 默认20
        移动平均周期
    std_dev : float, 默认2.0
        标准差倍数
    """
    name = "BBANDS"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        period = int(self.params.get("period", 20))
        std_dev = self.params.get("std_dev", 2.0)
        
        middle = df["close"].rolling(window=period).mean()
        std = df["close"].rolling(window=period).std()
        
        # Return bandwidth (Upper - Lower) / Middle
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        
        return (upper - lower) / middle


class BBandsWidthFactor(FactorBase):
    """布林带宽度"""
    name = "BBANDS_WIDTH"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        period = int(self.params.get("period", 20))
        std_dev = self.params.get("std_dev", 2.0)
        
        middle = df["close"].rolling(window=period).mean()
        std = df["close"].rolling(window=period).std()
        
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        
        return upper - lower


class BBandsPositionFactor(FactorBase):
    """布林带位置 (%B)"""
    name = "BBANDS_POSITION"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        period = int(self.params.get("period", 20))
        std_dev = self.params.get("std_dev", 2.0)
        
        middle = df["close"].rolling(window=period).mean()
        std = df["close"].rolling(window=period).std()
        
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        
        return (df["close"] - lower) / (upper - lower)


class KeltnerChannelFactor(FactorBase):
    """
    Keltner Channel (肯特纳通道)
    
    基于ATR的波动率通道。
    """
    name = "KELTNER"
    required_cols = ["high", "low", "close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        period = int(self.params.get("period", 20))
        mult = self.params.get("multiplier", 2.0)
        
        middle = df["close"].ewm(span=period).mean()
        
        # Calculate ATR
        tr1 = df["high"] - df["low"]
        tr2 = abs(df["high"] - df["close"].shift(1))
        tr3 = abs(df["low"] - df["close"].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Return upper - lower bandwidth
        upper = middle + mult * atr
        lower = middle - mult * atr
        
        return (upper - lower) / middle


class DonchianChannelFactor(FactorBase):
    """
    Donchian Channel (唐奇安通道)
    
    最高价和最低价形成的通道。
    """
    name = "DONCHIAN"
    required_cols = ["high", "low"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        period = int(self.params.get("period", 20))
        
        upper = df["high"].rolling(window=period).max()
        lower = df["low"].rolling(window=period).min()
        
        return (upper - lower) / upper


class StandardDeviationFactor(FactorBase):
    """
    Standard Deviation (标准差)
    
    价格波动的标准差。
    """
    name = "STD"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        period = int(self.params.get("period", 20))
        
        return df["close"].rolling(window=period).std()


class HistoricalVolatilityFactor(FactorBase):
    """
    Historical Volatility (历史波动率)
    
    收益率的对数标准差。
    """
    name = "HIST_VOLATILITY"
    required_cols = ["close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        period = int(self.params.get("period", 20))
        
        log_returns = np.log(df["close"] / df["close"].shift(1))
        return log_returns.rolling(window=period).std() * np.sqrt(252)


# ============================================================
# 成交量类因子 (Volume Factors)
# ============================================================

class OBVFactor(FactorBase):
    """
    OBV (On Balance Volume)
    
    能量潮，成交量累计。
    """
    name = "OBV"
    required_cols = ["close", "vol"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        vol = df["vol"]
        
        direction = np.sign(close.diff())
        direction = direction.fillna(0)
        
        obv = (direction * vol).cumsum()
        return obv


class VROCFactor(FactorBase):
    """
    VROC (Volume Rate of Change)
    
    成交量变化率。
    """
    name = "VROC"
    required_cols = ["vol"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        period = int(self.params.get("period", 12))
        
        vroc = ((df["vol"] - df["vol"].shift(period)) / df["vol"].shift(period)) * 100
        return vroc


class VBSFactor(FactorBase):
    """
    Volume Bollinger Bands (成交量布林带)
    
    成交量布林带分析。
    """
    name = "VBBANDS"
    required_cols = ["vol"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        period = int(self.params.get("period", 20))
        std_dev = self.params.get("std_dev", 2.0)
        
        middle = df["vol"].rolling(window=period).mean()
        std = df["vol"].rolling(window=period).std()
        
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        
        return (upper - lower) / middle


class CMFFactor(FactorBase):
    """
    CMF (Chaikin Money Flow)
    
    资金流量。
    """
    name = "CMF"
    required_cols = ["high", "low", "close", "vol"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        period = int(self.params.get("period", 20))
        
        high = df["high"]
        low = df["low"]
        close = df["close"]
        vol = df["vol"]
        
        money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
        money_flow_multiplier = money_flow_multiplier.fillna(0)
        
        money_flow_volume = money_flow_multiplier * vol
        
        cmf = money_flow_volume.rolling(window=period).sum() / vol.rolling(window=period).sum()
        return cmf


class MFIFactor(FactorBase):
    """
    MFI (Money Flow Index)
    
    资金流量指标。
    """
    name = "MFI"
    required_cols = ["high", "low", "close", "vol"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        period = int(self.params.get("period", 14))
        
        tp = (df["high"] + df["low"] + df["close"]) / 3
        raw_money_flow = tp * df["vol"]
        
        positive_flow = tp.diff().apply(lambda x: x if x > 0 else 0)
        negative_flow = tp.diff().apply(lambda x: -x if x < 0 else 0)
        
        avg_gain = positive_flow.rolling(window=period).mean()
        avg_loss = negative_flow.rolling(window=period).mean()
        
        money_ratio = avg_gain / avg_loss
        mfi = 100 - (100 / (1 + money_ratio))
        return mfi


class VWMAVolumeFactor(FactorBase):
    """
    VWMA Volume (成交量加权移动平均成交量)
    
    成交量加权平均。
    """
    name = "VWMA_VOL"
    required_cols = ["vol", "close"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        period = int(self.params.get("period", 20))
        
        vwma = df["vol"] * df["close"]
        return vwma.rolling(window=period).mean()


class VolumeMAFactor(FactorBase):
    """
    Volume MA (成交量移动平均)
    
    成交量的移动平均。
    """
    name = "VOL_MA"
    required_cols = ["vol"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        period = int(self.params.get("period", 5))
        
        return df["vol"].rolling(window=period).mean()


class VolumeRatioFactor(FactorBase):
    """
    Volume Ratio (量比)
    
    当前成交量与历史平均成交量的比值。
    """
    name = "VOL_RATIO"
    required_cols = ["vol"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        period = int(self.params.get("period", 5))
        
        vol_ma = df["vol"].rolling(window=period).mean()
        return df["vol"] / vol_ma


class TurnoverRateFactor(FactorBase):
    """
    Turnover Rate (换手率)
    
    成交量/流通股本。
    """
    name = "TURNOVER_RATE"
    required_cols = ["vol"]

    def _calculate_single(self, df: pd.DataFrame) -> pd.Series:
        # 假设有流通股本字段，如果没有则使用成交量代替
        if "float_share" in df.columns:
            return df["vol"] / df["float_share"] * 100
