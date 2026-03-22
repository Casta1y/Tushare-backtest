"""
backtest_report.py - 回测报告生成模块

提供HTML格式的回测报告生成功能，包含图表和详细指标。
"""

import os
import json
import base64
from datetime import datetime
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd


class BacktestReport:
    """
    回测报告生成器
    
    用于生成完整的回测报告：
    - HTML格式
    - 包含图表
    - 详细指标
    
    用法:
        report = BacktestReport(results)
        report.generate_html("report.html")
    """
    
    def __init__(
        self,
        results: Dict,
        equity_curve: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
        title: str = "量化回测报告",
    ):
        """
        初始化报告生成器
        
        Args:
            results: 回测结果字典
            equity_curve: 权益曲线数据（用于绑图）
            title: 报告标题
        """
        self.results = results
        self.equity_curve = equity_curve
        self.title = title
        self.charts = {}
    
    def _format_number(self, value: float, fmt: str = "{:,.2f}") -> str:
        """
        格式化数字
        
        Args:
            value: 数值
            fmt: 格式字符串
            
        Returns:
            格式化后的字符串
        """
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "N/A"
        return fmt.format(value)
    
    def _format_pct(self, value: float) -> str:
        """
        格式化百分比
        
        Args:
            value: 数值（0.05 表示 5%）
            
        Returns:
            格式化后的百分比字符串
        """
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "N/A"
        return f"{value * 100:.2f}%"
    
    def _format_ratio(self, value: float) -> str:
        """
        格式化比率
        
        Args:
            value: 数值
            
        Returns:
            格式化后的比率字符串
        """
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "N/A"
        if value == float('inf'):
            return "∞"
        return f"{value:.2f}"
    
    def _extract_metrics(self) -> Dict[str, str]:
        """
        提取关键指标
        
        Returns:
            指标字典
        """
        r = self.results
        
        return {
            # 基本信息
            'strategy_name': r.get('strategy_name', 'Unknown'),
            'start_date': r.get('start_date', 'N/A'),
            'end_date': r.get('end_date', 'N/A'),
            'initial_cash': self._format_number(r.get('initial_cash', 0)),
            'final_value': self._format_number(r.get('final_value', 0)),
            
            # 收益率指标
            'total_return': self._format_pct(r.get('total_return_pct', r.get('total_return', 0)) / 100),
            'annual_return': self._format_pct(r.get('annual_return_pct', r.get('annual_return', 0)) / 100),
            
            # 风险指标
            'max_drawdown': self._format_pct(r.get('max_drawdown_pct', r.get('max_drawdown', 0)) / 100),
            'volatility': self._format_pct(r.get('volatility', 0) / 100),
            
            # 收益指标
            'sharpe_ratio': self._format_ratio(r.get('sharpe_ratio', 0)),
            'sortino_ratio': self._format_ratio(r.get('sortino_ratio', 0)),
            'calmar_ratio': self._format_ratio(r.get('calmar_ratio', 0)),
            
            # 交易统计
            'total_trades': r.get('total_trades', 0),
            'winning_trades': r.get('winning_trades', 0),
            'losing_trades': r.get('losing_trades', 0),
            'win_rate': self._format_pct(r.get('win_rate', 0) / 100),
            'avg_win': self._format_pct(r.get('avg_win', 0) / 100),
            'avg_loss': self._format_pct(r.get('avg_loss', 0) / 100),
            'profit_factor': self._format_ratio(r.get('profit_factor', 0)),
        }
    
    def _get_html_header(self) -> str:
        """
        获取HTML头部
        
        Returns:
            HTML头部字符串
        """
        return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>""" + self.title + """</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 32px;
            font-weight: 700;
            margin-bottom: 10px;
        }
        
        .header .subtitle {
            font-size: 14px;
            opacity: 0.8;
        }
        
        .section {
            padding: 30px 40px;
            border-bottom: 1px solid #eee;
        }
        
        .section:last-child {
            border-bottom: none;
        }
        
        .section-title {
            font-size: 20px;
            font-weight: 600;
            color: #1a1a2e;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .section-title::before {
            content: '';
            width: 4px;
            height: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 2px;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }
        
        .metric-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        }
        
        .metric-card.highlight {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .metric-card.positive {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
        }
        
        .metric-card.negative {
            background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
            color: white;
        }
        
        .metric-label {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
            opacity: 0.8;
        }
        
        .metric-value {
            font-size: 28px;
            font-weight: 700;
        }
        
        .chart-container {
            margin: 20px 0;
            text-align: center;
        }
        
        .chart-container img {
            max-width: 100%;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        
        .chart-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        
        th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        
        th {
            background: #f8f9fa;
            font-weight: 600;
            color: #1a1a2e;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        tr:hover {
            background: #f8f9fa;
        }
        
        .footer {
            background: #1a1a2e;
            color: white;
            padding: 20px 40px;
            text-align: center;
            font-size: 12px;
            opacity: 0.8;
        }
        
        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }
        
        .badge-success {
            background: #d4edda;
            color: #155724;
        }
        
        .badge-danger {
            background: #f8d7da;
            color: #721c24;
        }
        
        .badge-warning {
            background: #fff3cd;
            color: #856404;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 """ + self.title + """</h1>
            <div class="subtitle">生成时间: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</div>
        </div>
"""
    
    def _get_metrics_section(self) -> str:
        """
        获取指标部分HTML
        
        Returns:
            HTML字符串
        """
        m = self._extract_metrics()
        
        # 判断收益正负
        return_pct = self.results.get('total_return_pct', self.results.get('total_return', 0))
        is_positive = return_pct > 0
        
        dd = self.results.get('max_drawdown_pct', self.results.get('max_drawdown', 0))
        is_high_dd = abs(dd) > 20  # 超过20%认为是高回撤
        
        html = """
        <div class="section">
            <h2 class="section-title">📈 策略概览</h2>
            <div class="metrics-grid">
                <div class="metric-card highlight">
                    <div class="metric-label">策略名称</div>
                    <div class="metric-value" style="font-size: 20px;">""" + m['strategy_name'] + """</div>
                </div>
                <div class="metric-card """ + ("positive" if is_positive else "negative") + """">
                    <div class="metric-label">总收益率</div>
                    <div class="metric-value">""" + m['total_return'] + """</div>
                </div>
                <div class="metric-card """ + ("positive" if is_positive else "negative") + """">
                    <div class="metric-label">年化收益率</div>
                    <div class="metric-value">""" + m['annual_return'] + """</div>
                </div>
                <div class="metric-card """ + ("negative" if is_high_dd else "") + """">
                    <div class="metric-label">最大回撤</div>
                    <div class="metric-value">""" + m['max_drawdown'] + """</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2 class="section-title">📅 回测信息</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">回测开始日期</div>
                    <div class="metric-value" style="font-size: 18px;">""" + m['start_date'] + """</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">回测结束日期</div>
                    <div class="metric-value" style="font-size: 18px;">""" + m['end_date'] + """</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">初始资金</div>
                    <div class="metric-value" style="font-size: 18px;">¥""" + m['initial_cash'] + """</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">最终资产</div>
                    <div class="metric-value" style="font-size: 18px;">¥""" + m['final_value'] + """</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2 class="section-title">⚖️ 风险指标</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">夏普比率</div>
                    <div class="metric-value">""" + m['sharpe_ratio'] + """</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Sortino比率</div>
                    <div class="metric-value">""" + m['sortino_ratio'] + """</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Calmar比率</div>
                    <div class="metric-value">""" + m['calmar_ratio'] + """</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">年化波动率</div>
                    <div class="metric-value">""" + m['volatility'] + """</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2 class="section-title">🎯 交易统计</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">总交易次数</div>
                    <div class="metric-value">""" + str(m['total_trades']) + """</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">盈利次数</div>
                    <div class="metric-value" style="color: #27AE60;">""" + str(m['winning_trades']) + """</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">亏损次数</div>
                    <div class="metric-value" style="color: #E74C3C;">""" + str(m['losing_trades']) + """</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">胜率</div>
                    <div class="metric-value">""" + m['win_rate'] + """</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">平均盈利</div>
                    <div class="metric-value" style="color: #27AE60;">""" + m['avg_win'] + """</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">平均亏损</div>
                    <div class="metric-value" style="color: #E74C3C;">""" + m['avg_loss'] + """</div>
                </div>
            </div>
        </div>
"""
        return html
    
    def _get_charts_section(self) -> str:
        """
        获取图表部分HTML
        
        Returns:
            HTML字符串
        """
        # 检查是否有图表需要嵌入
        charts_html = ""
        
        if self.charts:
            charts_html = """
        <div class="section">
            <h2 class="section-title">📊 收益曲线</h2>
            <div class="chart-container">
                <img src="{}" alt="Returns Chart">
            </div>
        </div>
""".format(self.charts.get('returns', ''))
        
        return charts_html
    
    def _get_footer(self) -> str:
        """
        获取HTML页脚
        
        Returns:
            HTML页脚字符串
        """
        return """
        <div class="footer">
            <p>量化回测框架 | Generated by Backtest System</p>
        </div>
    </div>
</body>
</html>
"""
    
    def generate_html(
        self,
        filepath: str = "backtest_report.html",
        include_charts: bool = True,
    ) -> str:
        """
        生成HTML报告
        
        Args:
            filepath: 输出文件路径
            include_charts: 是否包含图表
            
        Returns:
            生成的HTML内容
        """
        # 如果需要图表，尝试生成
        if include_charts and self.equity_curve is not None:
            try:
                from .result_visualizer import ResultVisualizer
                
                output_dir = os.path.dirname(filepath) or "."
                viz = ResultVisualizer(
                    self.equity_curve,
                    title=self.results.get('strategy_name', 'Strategy')
                )
                
                # 生成并保存图表
                saved = viz.save_all(show=False)
                
                # 读取图表并转换为base64
                for name, path in saved.items():
                    if os.path.exists(path):
                        with open(path, 'rb') as f:
                            img_data = base64.b64encode(f.read()).decode()
                            self.charts[name] = f"data:image/png;base64,{img_data}"
            except Exception as e:
                print(f"生成图表时出错: {e}")
        
        # 组装HTML
        html = self._get_html_header()
        html += self._get_metrics_section()
        html += self._get_charts_section()
        html += self._get_footer()
        
        # 保存文件
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"报告已生成: {filepath}")
        
        return html
    
    def generate_json(
        self,
        filepath: str = "backtest_report.json",
    ) -> Dict:
        """
        生成JSON格式的报告
        
        Args:
            filepath: 输出文件路径
            
        Returns:
            结果字典
        """
        # 保存为JSON
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        
        # 移除不可序列化的字段
        serializable_results = {}
        for k, v in self.results.items():
            if isinstance(v, (str, int, float, bool, type(None))):
                serializable_results[k] = v
            elif isinstance(v, list):
                # 保留列表但限制长度
                serializable_results[k] = v[:100] if len(v) > 100 else v
            elif isinstance(v, dict):
                serializable_results[k] = str(v)
            else:
                serializable_results[k] = str(v)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        print(f"JSON报告已生成: {filepath}")
        
        return serializable_results
    
    def generate_markdown(
        self,
        filepath: str = "backtest_report.md",
    ) -> str:
        """
        生成Markdown格式的报告
        
        Args:
            filepath: 输出文件路径
            
        Returns:
            Markdown内容
        """
        m = self._extract_metrics()
        
        md = f"""# {self.title}

**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## 📈 策略概览

| 指标 | 值 |
|------|-----|
| 策略名称 | {m['strategy_name']} |
| 总收益率 | **{m['total_return']}** |
| 年化收益率 | **{m['annual_return']}** |
| 最大回撤 | {m['max_drawdown']} |

---

## 📅 回测信息

| 指标 | 值 |
|------|-----|
| 回测开始日期 | {m['start_date']} |
| 回测结束日期 | {m['end_date']} |
| 初始资金 | ¥{m['initial_cash']} |
| 最终资产 | ¥{m['final_value']} |

---

## ⚖️ 风险指标

| 指标 | 值 |
|------|-----|
| 夏普比率 | {m['sharpe_ratio']} |
| Sortino比率 | {m['sortino_ratio']} |
| Calmar比率 | {m['calmar_ratio']} |
| 年化波动率 | {m['volatility']} |

---

## 🎯 交易统计

| 指标 | 值 |
|------|-----|
| 总交易次数 | {m['total_trades']} |
| 盈利次数 | {m['winning_trades']} |
| 亏损次数 | {m['losing_trades']} |
| 胜率 | {m['win_rate']} |
| 平均盈利 | {m['avg_win']} |
| 平均亏损 | {m['avg_loss']} |
| 盈利因子 | {m['profit_factor']} |

---

*量化回测框架 | Generated by Backtest System*
"""
        
        # 保存文件
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(md)
        
        print(f"Markdown报告已生成: {filepath}")
        
        return md


# ---------- 便捷函数 ----------

def generate_report(
    results: Dict,
    equity_curve: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
    output_dir: str = "./output",
    title: str = "量化回测报告",
) -> str:
    """
    便捷报告生成函数
    
    Args:
        results: 回测结果字典
        equity_curve: 权益曲线
        output_dir: 输出目录
        title: 报告标题
        
    Returns:
        生成的HTML内容
    """
    os.makedirs(output_dir, exist_ok=True)
    
    report = BacktestReport(results, equity_curve, title)
    
    # 生成HTML报告
    html_path = os.path.join(output_dir, "backtest_report.html")
    html = report.generate_html(html_path, include_charts=True)
    
    # 生成Markdown报告
    md_path = os.path.join(output_dir, "backtest_report.md")
    report.generate_markdown(md_path)
    
    # 生成JSON报告
    json_path = os.path.join(output_dir, "backtest_report.json")
    report.generate_json(json_path)
    
    return html
