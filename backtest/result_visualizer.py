"""
result_visualizer.py - 结果可视化模块

提供回测结果的可视化功能，包括收益率曲线、回撤图、因子分布图、滚动指标图等。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, List, Optional, Tuple, Union

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置默认样式
plt.style.use('seaborn-v0_8-whitegrid')


class ResultVisualizer:
    """
    结果可视化器
    
    用于可视化回测结果：
    - 收益率曲线
    - 回撤曲线
    - 因子分布
    - 滚动指标
    
    用法:
        viz = ResultVisualizer(results)
        viz.plot_returns()
        viz.plot_drawdown()
        viz.save_all("output/")
    """
    
    def __init__(
        self,
        equity_curve: Union[pd.DataFrame, pd.Series, np.ndarray, None] = None,
        benchmark: Optional[Union[pd.Series, np.ndarray]] = None,
        title: str = "Backtest Result",
        output_dir: str = "./output",
    ):
        """
        初始化可视化器
        
        Args:
            equity_curve: 权益曲线（可选，后续可通过 set_data 设置）
            benchmark: 基准收益率序列 (可选)
            title: 图表标题
            output_dir: 输出目录
        """
        self.title = title
        self.output_dir = output_dir
        self.benchmark = benchmark
        self.equity = None
        
        # 如果传入了数据，则处理
        if equity_curve is not None:
            self.set_data(equity_curve)
    
    def set_data(
        self,
        equity_curve: Union[pd.DataFrame, pd.Series, np.ndarray],
        benchmark: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> None:
        """
        设置数据
        
        Args:
            equity_curve: 权益曲线
            benchmark: 基准收益率序列 (可选)
        """
        # 处理 equity_curve
        if isinstance(equity_curve, pd.DataFrame):
            if 'value' in equity_curve.columns:
                self.equity = equity_curve['value'].values
                if 'date' in equity_curve.columns:
                    self.dates = pd.to_datetime(equity_curve['date'])
                else:
                    self.dates = pd.RangeIndex(start=0, stop=len(self.equity))
            elif len(equity_curve.columns) >= 2:
                self.dates = pd.to_datetime(equity_curve.iloc[:, 0])
                self.equity = equity_curve.iloc[:, 1].values
            else:
                self.equity = equity_curve.iloc[:, 0].values
                self.dates = pd.RangeIndex(start=0, stop=len(self.equity))
        elif isinstance(equity_curve, pd.Series):
            self.equity = equity_curve.values
            self.dates = equity_curve.index if hasattr(equity_curve, 'index') else pd.RangeIndex(start=0, stop=len(self.equity))
        elif isinstance(equity_curve, (list, np.ndarray)):
            self.equity = np.array(equity_curve)
            self.dates = pd.RangeIndex(start=0, stop=len(self.equity))
        else:
            raise ValueError("equity_curve 类型不支持")
        
        # 转换为DataFrame便于处理
        if not isinstance(self.dates, pd.DatetimeIndex):
            self.dates = pd.to_datetime(self.dates)
        
        self.df = pd.DataFrame({
            'date': self.dates,
            'equity': self.equity
        })
        self.df.set_index('date', inplace=True)
        
        # 计算收益率
        self.df['returns'] = self.df['equity'].pct_change().fillna(0)
        self.df['cum_returns'] = (1 + self.df['returns']).cumprod() - 1
        
        # 处理基准
        if benchmark is not None:
            if isinstance(benchmark, pd.Series):
                self.benchmark_returns = benchmark.values
            else:
                self.benchmark_returns = np.array(benchmark)
            self.df['benchmark_returns'] = self.benchmark_returns[:len(self.df)]
            self.df['benchmark_cum'] = (1 + self.df['benchmark_returns'].fillna(0)).cumprod() - 1
        else:
            self.benchmark_returns = None
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
    
    def plot_returns(
        self,
        figsize: Tuple[int, int] = (12, 6),
        show: bool = True,
        save: bool = True,
    ) -> plt.Figure:
        """
        绘制收益率曲线
        
        Args:
            figsize: 图表大小
            show: 是否显示
            save: 是否保存
            
        Returns:
            matplotlib Figure对象
        """
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # 上图：权益曲线
        ax1 = axes[0]
        ax1.plot(self.df.index, self.df['equity'], label='Strategy', linewidth=2, color='#2E86AB')
        
        if self.benchmark_returns is not None and 'benchmark_cum' in self.df.columns:
            ax1.plot(self.df.index, self.df['equity'].iloc[0] * (1 + self.df['benchmark_cum']), 
                    label='Benchmark', linewidth=1.5, color='#A23B72', alpha=0.7)
        
        ax1.set_ylabel('Portfolio Value', fontsize=12)
        ax1.set_title(f'{self.title} - Equity Curve', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 添加资金标注
        final_value = self.df['equity'].iloc[-1]
        ax1.axhline(y=final_value, color='gray', linestyle='--', alpha=0.5)
        ax1.annotate(f'¥{final_value:,.0f}', 
                    xy=(self.df.index[-1], final_value),
                    xytext=(10, 0), textcoords='offset points',
                    fontsize=10, color='gray')
        
        # 下图：累计收益率
        ax2 = axes[1]
        ax2.fill_between(self.df.index, 0, self.df['cum_returns'] * 100, 
                        alpha=0.3, color='#2E86AB', label='Strategy Return')
        ax2.plot(self.df.index, self.df['cum_returns'] * 100, 
                linewidth=2, color='#2E86AB')
        
        if self.benchmark_returns is not None and 'benchmark_cum' in self.df.columns:
            ax2.plot(self.df.index, self.df['benchmark_cum'] * 100,
                    linewidth=1.5, color='#A23B72', alpha=0.7, label='Benchmark Return')
        
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax2.set_title('Cumulative Returns', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 添加零线
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'returns.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"收益曲线已保存: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_drawdown(
        self,
        figsize: Tuple[int, int] = (12, 4),
        show: bool = True,
        save: bool = True,
    ) -> plt.Figure:
        """
        绘制回撤曲线
        
        Args:
            figsize: 图表大小
            show: 是否显示
            save: 是否保存
            
        Returns:
            matplotlib Figure对象
        """
        # 计算回撤
        self.df['cummax'] = self.df['equity'].cummax()
        self.df['drawdown'] = (self.df['equity'] - self.df['cummax']) / self.df['cummax']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 填充回撤区域
        ax.fill_between(self.df.index, 0, self.df['drawdown'] * 100,
                       color='#E74C3C', alpha=0.4)
        ax.plot(self.df.index, self.df['drawdown'] * 100,
               color='#E74C3C', linewidth=1.5)
        
        # 标注最大回撤
        max_dd = self.df['drawdown'].min()
        max_dd_date = self.df['drawdown'].idxmin()
        
        ax.scatter([max_dd_date], [max_dd * 100], color='#E74C3C', s=100, zorder=5)
        ax.annotate(f'Max DD: {max_dd*100:.1f}%',
                   xy=(max_dd_date, max_dd * 100),
                   xytext=(10, -20), textcoords='offset points',
                   fontsize=11, color='#E74C3C', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1))
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.set_title(f'{self.title} - Drawdown', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'drawdown.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"回撤曲线已保存: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_factor_distribution(
        self,
        returns_col: str = 'returns',
        figsize: Tuple[int, int] = (12, 6),
        show: bool = True,
        save: bool = True,
    ) -> plt.Figure:
        """
        绘制收益率分布图
        
        Args:
            returns_col: 收益率列名
            figsize: 图表大小
            show: 是否显示
            save: 是否保存
            
        Returns:
            matplotlib Figure对象
        """
        returns = self.df[returns_col].dropna()
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 左图：直方图
        ax1 = axes[0]
        n, bins, patches = ax1.hist(returns * 100, bins=50, density=True,
                                   alpha=0.7, color='#2E86AB', edgecolor='white')
        
        # 添加正态分布曲线
        mu, sigma = returns.mean() * 100, returns.std() * 100
        x = np.linspace(returns.min() * 100, returns.max() * 100, 100)
        from scipy import stats
        ax1.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2,
                label=f'Normal: μ={mu:.2f}%, σ={sigma:.2f}%')
        
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax1.set_xlabel('Daily Return (%)', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title('Return Distribution', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 右图：箱线图
        ax2 = axes[1]
        
        # 准备数据
        data = [
            returns[returns > 0] * 100,  # 盈利
            returns[returns < 0] * 100,  # 亏损
            returns * 100,  # 全部
        ]
        labels = ['Win Days', 'Loss Days', 'All Days']
        colors = ['#27AE60', '#E74C3C', '#2E86AB']
        
        bp = ax2.boxplot(data, labels=labels, patch_artist=True, notch=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.set_ylabel('Daily Return (%)', fontsize=12)
        ax2.set_title('Return Box Plot', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 添加统计信息
        stats_text = f"Win Rate: {(returns > 0).mean()*100:.1f}%\n"
        stats_text += f"Mean: {returns.mean()*100:.3f}%\n"
        stats_text += f"Std: {returns.std()*100:.3f}%"
        ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'factor_distribution.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"收益分布图已保存: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_rolling_metrics(
        self,
        window: int = 60,
        figsize: Tuple[int, int] = (12, 8),
        show: bool = True,
        save: bool = True,
    ) -> plt.Figure:
        """
        绘制滚动指标图
        
        Args:
            window: 滚动窗口大小
            figsize: 图表大小
            show: 是否显示
            save: 是否保存
            
        Returns:
            matplotlib Figure对象
        """
        # 计算滚动指标
        self.df['rolling_return'] = self.df['equity'].pct_change(window).fillna(0)
        self.df['rolling_volatility'] = self.df['returns'].rolling(window).std() * np.sqrt(252)
        
        # 计算滚动夏普比率
        rf_daily = 0.03 / 252
        rolling_mean = self.df['returns'].rolling(window).mean()
        rolling_std = self.df['returns'].rolling(window).std()
        self.df['rolling_sharpe'] = (rolling_mean - rf_daily) / rolling_std * np.sqrt(252)
        
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        # 滚动收益率
        ax1 = axes[0]
        ax1.plot(self.df.index, self.df['rolling_return'] * 100,
                color='#2E86AB', linewidth=1.5, label=f'{window}-day Return')
        ax1.fill_between(self.df.index, 0, self.df['rolling_return'] * 100,
                        where=self.df['rolling_return'] >= 0,
                        color='#27AE60', alpha=0.3)
        ax1.fill_between(self.df.index, 0, self.df['rolling_return'] * 100,
                        where=self.df['rolling_return'] < 0,
                        color='#E74C3C', alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax1.set_ylabel('Rolling Return (%)', fontsize=11)
        ax1.set_title(f'{self.title} - Rolling Metrics (window={window})', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 滚动波动率
        ax2 = axes[1]
        ax2.plot(self.df.index, self.df['rolling_volatility'] * 100,
                color='#E67E22', linewidth=1.5, label=f'{window}-day Volatility')
        ax2.fill_between(self.df.index, 0, self.df['rolling_volatility'] * 100,
                        alpha=0.3, color='#E67E22')
        ax2.set_ylabel('Volatility (%)', fontsize=11)
        ax2.set_title('Rolling Volatility (Annualized)', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 滚动夏普比率
        ax3 = axes[2]
        ax3.plot(self.df.index, self.df['rolling_sharpe'],
                color='#9B59B6', linewidth=1.5, label=f'{window}-day Sharpe')
        ax3.fill_between(self.df.index, 0, self.df['rolling_sharpe'],
                        where=self.df['rolling_sharpe'] >= 0,
                        color='#27AE60', alpha=0.3)
        ax3.fill_between(self.df.index, 0, self.df['rolling_sharpe'],
                        where=self.df['rolling_sharpe'] < 0,
                        color='#E74C3C', alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax3.set_xlabel('Date', fontsize=12)
        ax3.set_ylabel('Sharpe Ratio', fontsize=11)
        ax3.set_title('Rolling Sharpe Ratio', fontsize=14, fontweight='bold')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'rolling_metrics.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"滚动指标图已保存: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_monthly_returns(
        self,
        figsize: Tuple[int, int] = (12, 6),
        show: bool = True,
        save: bool = True,
    ) -> plt.Figure:
        """
        绘制月度收益热力图
        
        Args:
            figsize: 图表大小
            show: 是否显示
            save: 是否保存
            
        Returns:
            matplotlib Figure对象
        """
        # 计算月度收益
        monthly = self.df['returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # 转换为年月矩阵
        monthly_df = monthly.to_frame('return')
        monthly_df['year'] = monthly_df.index.year
        monthly_df['month'] = monthly_df.index.month
        
        # 创建透视表
        pivot = monthly_df.pivot_table(values='return', index='year', columns='month')
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制热力图
        im = ax.imshow(pivot.values * 100, cmap='RdYlGn', aspect='auto', vmin=-15, vmax=15)
        
        # 设置标签
        ax.set_xticks(range(12))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        
        # 添加数值标注
        for i in range(len(pivot.index)):
            for j in range(12):
                if j < len(pivot.columns) and not np.isnan(pivot.values[i, j]):
                    text_color = 'white' if abs(pivot.values[i, j]) > 7 else 'black'
                    ax.text(j, i, f'{pivot.values[i, j]*100:.1f}',
                           ha='center', va='center', color=text_color, fontsize=9)
        
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Year', fontsize=12)
        ax.set_title(f'{self.title} - Monthly Returns Heatmap', fontsize=14, fontweight='bold')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Return (%)', fontsize=11)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'monthly_returns.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"月度收益热力图已保存: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def save_all(
        self,
        show: bool = False,
    ) -> Dict[str, str]:
        """
        保存所有图表
        
        Args:
            show: 是否显示
            
        Returns:
            保存的文件路径字典
        """
        saved = {}
        
        try:
            fig = self.plot_returns(show=show, save=True)
            saved['returns'] = os.path.join(self.output_dir, 'returns.png')
        except Exception as e:
            print(f"绘制收益曲线失败: {e}")
        
        try:
            fig = self.plot_drawdown(show=show, save=True)
            saved['drawdown'] = os.path.join(self.output_dir, 'drawdown.png')
        except Exception as e:
            print(f"绘制回撤曲线失败: {e}")
        
        try:
            fig = self.plot_factor_distribution(show=show, save=True)
            saved['distribution'] = os.path.join(self.output_dir, 'factor_distribution.png')
        except Exception as e:
            print(f"绘制分布图失败: {e}")
        
        try:
            fig = self.plot_rolling_metrics(show=show, save=True)
            saved['rolling'] = os.path.join(self.output_dir, 'rolling_metrics.png')
        except Exception as e:
            print(f"绘制滚动指标失败: {e}")
        
        try:
            fig = self.plot_monthly_returns(show=show, save=True)
            saved['monthly'] = os.path.join(self.output_dir, 'monthly_returns.png')
        except Exception as e:
            print(f"绘制月度收益失败: {e}")
        
        return saved


# ---------- 便捷函数 ----------

def visualize_results(
    equity_curve: Union[pd.DataFrame, pd.Series, np.ndarray],
    benchmark: Optional[Union[pd.Series, np.ndarray]] = None,
    title: str = "Backtest Result",
    output_dir: str = "./output",
    show: bool = False,
) -> Dict[str, str]:
    """
    便捷可视化函数
    
    Args:
        equity_curve: 权益曲线
        benchmark: 基准收益率
        title: 图表标题
        output_dir: 输出目录
        show: 是否显示
        
    Returns:
        保存的文件路径字典
    """
    viz = ResultVisualizer(equity_curve, benchmark, title, output_dir)
    return viz.save_all(show=show)
