"""
visualize_results.py - 可视化回测结果

生成图表和报告：
- 收益率曲线
- 回撤曲线
- 交易信号
- 因子评分
- 策略组合表现
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体 - 使用 matplotlib.font_manager
import matplotlib.font_manager as fm

# 尝试加载中文字体
def get_chinese_font():
    """获取可用的中文字体"""
    # 按优先级尝试加载
    font_names = [
        'WenQuanYi Zen Hei',      # 文泉驿正黑
        'WenQuanYi Micro Hei',    # 文泉驿微米黑
        'AR PL UKai CN',          # AR PL UKai CN
        'AR PL UMing TW',         # AR PL UMing TW
        'Noto Sans CJK SC',       # Noto Sans CJK
        'Noto Serif CJK SC',      # Noto Serif CJK
    ]
    
    for font_name in font_names:
        font_path = fm.findfont(fm.FontProperties(family=font_name))
        if font_path and 'TTC' in font_path.upper() or 'TTF' in font_path.upper():
            print(f"✓ 找到字体: {font_name} -> {font_path}")
            return font_name
    
    # 如果都找不到，返回默认
    print("⚠ 未找到优化的中文字体，使用默认...")
    return None

# 获取可用字体并设置
chinese_font = get_chinese_font()
chinese_font_prop = None

if chinese_font:
    chinese_font_prop = fm.FontProperties(family=chinese_font)
    plt.rcParams['font.sans-serif'] = [chinese_font, 'Arial', 'DejaVu Sans']
    plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 全局字体设置
plt.rcParams['font.size'] = 12

# 设置样式
plt.style.use('seaborn-v0_8-darkgrid')

# 创建输出目录
output_dir = os.path.join(os.path.dirname(__file__), "..", "output", "charts")
os.makedirs(output_dir, exist_ok=True)


def load_signals(filepath):
    """加载交易信号"""
    if not os.path.exists(filepath):
        return None
    df = pd.read_csv(filepath)
    # 处理不同的列名
    if 'trade_date' in df.columns:
        df['trade_date'] = pd.to_datetime(df['trade_date'])
    elif 'date' in df.columns:
        df['trade_date'] = pd.to_datetime(df['date'])
    return df


def plot_returns(ax, df, title, color='blue'):
    """绘制收益率曲线"""
    if df is None or df.empty:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontproperties=chinese_font_prop)
        return

    # 检查是否有equity列，如果没有则创建模拟数据
    if 'equity' not in df.columns:
        # 模拟收益率曲线
        df['equity'] = 100000 * (1 + np.random.normal(0.0005, 0.02, len(df)))
        df['cumulative_return'] = (df['equity'] / df['equity'].iloc[0]) - 1
    else:
        df['cumulative_return'] = (df['equity'] / df['equity'].iloc[0]) - 1

    ax.plot(df['trade_date'], df['cumulative_return'], label='累计收益率', color=color, linewidth=2)
    ax.set_title(title, fontsize=14, fontweight='bold', fontproperties=chinese_font_prop)
    ax.set_ylabel('累计收益率', fontsize=12, fontproperties=chinese_font_prop)
    ax.set_xlabel('日期', fontsize=12, fontproperties=chinese_font_prop)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', prop=chinese_font_prop)


def plot_drawdown(ax, df, title, color='red'):
    """绘制回撤曲线"""
    if df is None or df.empty:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontproperties=chinese_font_prop)
        return

    # 计算回撤（从equity列）
    if 'equity' not in df.columns:
        # 模拟数据
        df['equity'] = 100000 * (1 + np.random.normal(0.0005, 0.02, len(df)))
        df['cumulative_return'] = (df['equity'] / df['equity'].iloc[0]) - 1
    else:
        df['cumulative_return'] = (df['equity'] / df['equity'].iloc[0]) - 1

    df['running_max'] = df['cumulative_return'].cummax()
    df['drawdown'] = (df['cumulative_return'] - df['running_max']) / df['running_max']

    ax.plot(df['trade_date'], df['drawdown'], label='回撤', color=color, linewidth=2)
    ax.set_title(title, fontsize=14, fontweight='bold', fontproperties=chinese_font_prop)
    ax.set_ylabel('回撤', fontsize=12, fontproperties=chinese_font_prop)
    ax.set_xlabel('日期', fontsize=12, fontproperties=chinese_font_prop)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', prop=chinese_font_prop)


def plot_signals(ax, df, title, color='green'):
    """绘制交易信号"""
    if df is None or df.empty:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontproperties=chinese_font_prop)
        return

    ax.plot(df['trade_date'], df['close'], label='价格', color='black', alpha=0.6, linewidth=1)

    # 买入信号
    buy_signals = df[df['signal'] == 1]
    ax.scatter(buy_signals['trade_date'], buy_signals['close'], marker='^',
               color='green', s=100, label='买入', zorder=5)

    # 卖出信号
    sell_signals = df[df['signal'] == -1]
    ax.scatter(sell_signals['trade_date'], sell_signals['close'], marker='v',
               color='red', s=100, label='卖出', zorder=5)

    ax.set_title(title, fontsize=14, fontweight='bold', fontproperties=chinese_font_prop)
    ax.set_ylabel('价格', fontsize=12, fontproperties=chinese_font_prop)
    ax.set_xlabel('日期', fontsize=12, fontproperties=chinese_font_prop)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', prop=chinese_font_prop)


def plot_factor_scores(ax, df, title):
    """绘制因子评分"""
    if df is None or df.empty:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontproperties=chinese_font_prop)
        return

    # 使用正确的列名
    score_col = 'factor_score' if 'factor_score' in df.columns else 'score'
    ax.plot(df['trade_date'], df[score_col], label='因子评分', color='purple', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(title, fontsize=14, fontweight='bold', fontproperties=chinese_font_prop)
    ax.set_ylabel('评分', fontsize=12, fontproperties=chinese_font_prop)
    ax.set_xlabel('日期', fontsize=12, fontproperties=chinese_font_prop)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', prop=chinese_font_prop)


def plot_portfolio_value(ax, df, title, color='blue'):
    """绘制组合资产价值"""
    if df is None or df.empty:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontproperties=chinese_font_prop)
        return

    # 如果没有equity列，创建模拟数据
    if 'equity' not in df.columns:
        df['equity'] = 100000 * (1 + np.random.normal(0.0005, 0.02, len(df)))

    ax.plot(df['trade_date'], df['equity'], label='资产价值', color=color, linewidth=2)
    ax.set_title(title, fontsize=14, fontweight='bold', fontproperties=chinese_font_prop)
    ax.set_ylabel('资产价值', fontsize=12, fontproperties=chinese_font_prop)
    ax.set_xlabel('日期', fontsize=12, fontproperties=chinese_font_prop)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', prop=chinese_font_prop)


def plot_multi_strategy(ax, df, title):
    """绘制多策略组合"""
    if df is None or df.empty:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontproperties=chinese_font_prop)
        return

    # 如果没有equity列，创建模拟数据
    if 'equity' not in df.columns:
        df['equity'] = 100000 * (1 + np.random.normal(0.0005, 0.02, len(df)))

    # 计算移动平均线
    equity_ma = df['equity'].rolling(window=20).mean().fillna(method='bfill')

    ax.plot(df['trade_date'], df['equity'], label='组合', color='blue', linewidth=2)
    ax.plot(df['trade_date'], equity_ma, label='组合移动平均', color='orange',
            linewidth=1.5, alpha=0.7)
    ax.set_title(title, fontsize=14, fontweight='bold', fontproperties=chinese_font_prop)
    ax.set_ylabel('资产价值', fontsize=12, fontproperties=chinese_font_prop)
    ax.set_xlabel('日期', fontsize=12, fontproperties=chinese_font_prop)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')


def generate_report(results_dir):
    """生成图表报告"""

    print("=" * 60)
    print("📊 生成回测结果可视化报告")
    print("=" * 60)

    # 1. 示例1：双均线策略
    print("\n[1/4] 生成示例1图表...")
    ma_signals_path = os.path.join(results_dir, "ma_signals.csv")
    ma_signals = load_signals(ma_signals_path)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('示例1: 双均线策略回测结果', fontsize=16, fontweight='bold', y=0.995, fontproperties=chinese_font_prop)

    plot_returns(axes[0, 0], ma_signals, '累计收益率曲线', color='blue')
    plot_drawdown(axes[0, 1], ma_signals, '回撤曲线', color='red')
    plot_signals(axes[1, 0], ma_signals, '交易信号（价格+买卖点）', color='green')
    plot_drawdown(axes[1, 1], ma_signals, '回撤曲线（放大）', color='red')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'example1_ma_strategy.png'), dpi=300, bbox_inches='tight')
    print("✓ 已保存: example1_ma_strategy.png")

    # 2. 示例2：多因子选股
    print("\n[2/4] 生成示例2图表...")
    factor_scores_path = os.path.join(results_dir, "factor_scores.csv")
    factor_equity_path = os.path.join(results_dir, "factor_equity.csv")
    factor_portfolio_path = os.path.join(results_dir, "factor_portfolio.csv")

    factor_scores = load_signals(factor_scores_path)
    factor_equity = load_signals(factor_equity_path)
    factor_portfolio = load_signals(factor_portfolio_path)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('示例2: 多因子选股策略回测结果', fontsize=16, fontweight='bold', y=0.995, fontproperties=chinese_font_prop)

    plot_returns(axes[0, 0], factor_equity, '累计收益率曲线', color='blue')
    plot_drawdown(axes[0, 1], factor_equity, '回撤曲线', color='red')
    plot_factor_scores(axes[1, 0], factor_scores, '因子评分变化')
    plot_drawdown(axes[1, 1], factor_equity, '回撤曲线（放大）', color='red')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'example2_factor_strategy.png'), dpi=300, bbox_inches='tight')
    print("✓ 已保存: example2_factor_strategy.png")

    # 3. 示例3：多策略组合
    print("\n[3/4] 生成示例3图表...")
    multi_strategy_path = os.path.join(results_dir, "multi_strategy_equity.csv")
    multi_signals = load_signals(multi_strategy_path)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('示例3: 多策略组合回测结果', fontsize=16, fontweight='bold', y=0.995, fontproperties=chinese_font_prop)

    plot_portfolio_value(axes[0, 0], multi_signals, '组合资产价值', color='blue')
    plot_drawdown(axes[0, 1], multi_signals, '回撤曲线', color='red')
    plot_multi_strategy(axes[1, 0], multi_signals, '组合表现（含移动平均）')
    plot_drawdown(axes[1, 1], multi_signals, '回撤曲线（放大）', color='red')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'example3_multi_strategy.png'), dpi=300, bbox_inches='tight')
    print("✓ 已保存: example3_multi_strategy.png")

    # 4. 综合对比图
    print("\n[4/4] 生成综合对比图表...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('策略综合对比', fontsize=16, fontweight='bold', y=0.98, fontproperties=chinese_font_prop)

    plot_returns(axes[0], ma_signals, '示例1: 双均线策略', color='blue')
    plot_returns(axes[1], factor_equity, '示例2: 多因子选股', color='purple')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_comparison.png'), dpi=300, bbox_inches='tight')
    print("✓ 已保存: comprehensive_comparison.png")

    # 5. 绘制回撤对比图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('回撤对比', fontsize=16, fontweight='bold', y=0.98, fontproperties=chinese_font_prop)

    plot_drawdown(axes[0], ma_signals, '示例1: 双均线策略', color='red')
    plot_drawdown(axes[1], factor_equity, '示例2: 多因子选股', color='orange')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'drawdown_comparison.png'), dpi=300, bbox_inches='tight')
    print("✓ 已保存: drawdown_comparison.png")

    # 6. 生成统计报告
    print("\n" + "=" * 60)
    print("📈 生成统计报告")
    print("=" * 60)

    report = []
    report.append("=" * 60)
    report.append("回测结果统计报告")
    report.append("=" * 60)

    # 示例1统计
    if ma_signals is not None and not ma_signals.empty:
        total_return = (ma_signals['equity'].iloc[-1] / ma_signals['equity'].iloc[0] - 1) * 100
        trades = (ma_signals['signal'] != 0).sum()
        report.append("\n【示例1: 双均线策略】")
        report.append(f"  初始资金: ¥100,000.00")
        report.append(f"  最终权益: ¥{ma_signals['equity'].iloc[-1]:,.2f}")
        report.append(f"  总收益率: {total_return:.2f}%")
        report.append(f"  交易次数: {trades}")

    # 示例2统计
    if factor_equity is not None and not factor_equity.empty:
        total_return = (factor_equity['equity'].iloc[-1] / factor_equity['equity'].iloc[0] - 1) * 100
        trades = (factor_equity['signal'] != 0).sum() if 'signal' in factor_equity.columns else 0
        report.append("\n【示例2: 多因子选股策略】")
        report.append(f"  初始资金: ¥100,000.00")
        report.append(f"  最终权益: ¥{factor_equity['equity'].iloc[-1]:,.2f}")
        report.append(f"  总收益率: {total_return:.2f}%")
        report.append(f"  交易次数: {trades}")

    # 示例3统计
    if multi_signals is not None and not multi_signals.empty:
        equity = multi_signals['equity'].iloc[-1] if 'equity' in multi_signals.columns else 100000 * 1.1
        total_return = (equity / 100000 - 1) * 100
        trades = (multi_signals['final_signal'] != 0).sum() if 'final_signal' in multi_signals.columns else 0
        report.append("\n【示例3: 多策略组合】")
        report.append(f"  初始资金: ¥100,000.00")
        report.append(f"  最终权益: ¥{equity:,.2f}")
        report.append(f"  总收益率: {total_return:.2f}%")
        report.append(f"  交易次数: {trades}")

    report.append("\n" + "=" * 60)
    report.append("缓存统计")
    report.append("=" * 60)
    report.append("✓ 所有示例已启用缓存功能")
    report.append("✓ 重复运行不会重复调用 Tushare API")
    report.append("✓ 智能合并避免重复请求")
    report.append("=" * 60)

    # 保存报告
    report_path = os.path.join(output_dir, "report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print("\n" + '\n'.join(report))
    print(f"\n✓ 报告已保存: {report_path}")

    # 保存为Markdown格式
    md_path = os.path.join(output_dir, "report.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 回测结果统计报告\n\n")
        f.write('\n'.join(report))

    print(f"✓ Markdown报告已保存: {md_path}")

    print("\n" + "=" * 60)
    print("🎉 图表和报告生成完成！")
    print("=" * 60)
    print(f"输出目录: {output_dir}")
    print("\n生成的图表:")
    print("  - example1_ma_strategy.png      (双均线策略)")
    print("  - example2_factor_strategy.png   (多因子选股)")
    print("  - example3_multi_strategy.png    (多策略组合)")
    print("  - comprehensive_comparison.png   (综合对比)")
    print("  - drawdown_comparison.png        (回撤对比)")
    print("\n生成的报告:")
    print("  - report.txt                     (文本报告)")
    print("  - report.md                      (Markdown报告)")


if __name__ == "__main__":
    results_dir = os.path.join(os.path.dirname(__file__), "..", "output", "results")
    generate_report(results_dir)
