"""
DocuSense AI - Eval Report Generator
Generates HTML and JSON reports
from evaluation results
"""

import json
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from loguru import logger


REPORTS_DIR = Path("evals/benchmark_results")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class ReportGenerator:
    """
    Generates evaluation reports in
    multiple formats:
    - JSON (machine readable)
    - HTML (human readable, recruiter-friendly)
    - Console (immediate feedback)
    """

    def generate_all(
        self,
        report,
        results : list
    ):
        """Generate all report formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.save_json(report, results, timestamp)
        self.save_html(report, results, timestamp)
        self.print_console(report)

        logger.success(
            f"Reports saved to {REPORTS_DIR}"
        )

    def save_json(self, report, results, timestamp):
        """Save machine-readable JSON report"""
        path = REPORTS_DIR / f"eval_{timestamp}.json"

        data = {
            "timestamp" : timestamp,
            "report"    : asdict(report),
            "results"   : [asdict(r) for r in results]
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        # Also save as latest
        latest = REPORTS_DIR / "latest.json"
        with open(latest, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"JSON report: {path.name}")

    def save_html(self, report, results, timestamp):
        """Generate beautiful HTML report"""
        path = REPORTS_DIR / f"eval_{timestamp}.html"

        # Score to color mapping
        def score_color(score):
            if score >= 0.7:
                return "#00c851"
            elif score >= 0.4:
                return "#ffbb33"
            else:
                return "#ff4444"

        def pct(val):
            return f"{val:.1%}"

        # Build category rows
        cat_rows = ""
        for cat, scores in report.category_scores.items():
            cat_rows += f"""
            <tr>
                <td>{cat.replace('_',' ').title()}</td>
                <td>{scores['count']}</td>
                <td style="color:{score_color(scores['keyword_overlap'])}">
                    {pct(scores['keyword_overlap'])}
                </td>
                <td style="color:{score_color(scores['confidence'])}">
                    {pct(scores['confidence'])}
                </td>
                <td style="color:{score_color(scores['correct_doc'])}">
                    {pct(scores['correct_doc'])}
                </td>
            </tr>
            """

        # Build results rows
        result_rows = ""
        for r in results:
            status_icon = "✅" if r.status == "success" else "❌"
            doc_icon    = "🎯" if r.correct_doc_found else "⚠️"
            result_rows += f"""
            <tr>
                <td>{status_icon}</td>
                <td title="{r.question}">{r.question[:60]}...</td>
                <td><span class="badge">{r.category}</span></td>
                <td><span class="badge">{r.difficulty}</span></td>
                <td style="color:{score_color(r.keyword_overlap)}">
                    {r.keyword_overlap:.3f}
                </td>
                <td style="color:{score_color(r.confidence)}">
                    {r.confidence:.2f}
                </td>
                <td>{doc_icon}</td>
                <td>{r.citation_count}</td>
                <td>{r.processing_time:.1f}s</td>
            </tr>
            """

        status_color = "#00c851" if report.passed_baseline else "#ff4444"
        status_text  = "PASSED" if report.passed_baseline else "FAILED"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>DocuSense AI - Eval Report</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont,
                         'Segoe UI', sans-serif;
            background: #0d1117;
            color: #e6edf3;
            padding: 20px;
        }}
        .header {{
            background: linear-gradient(135deg,#1a1a2e,#16213e);
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 24px;
            border: 1px solid #30363d;
        }}
        h1 {{ font-size: 28px; margin-bottom: 8px; }}
        h2 {{ font-size: 20px; margin: 20px 0 12px; }}
        .meta {{ color: #8b949e; font-size: 14px; }}
        .status-badge {{
            display: inline-block;
            padding: 6px 16px;
            border-radius: 20px;
            font-weight: bold;
            color: white;
            background: {status_color};
            margin-top: 10px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px,1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .card {{
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 10px;
            padding: 20px;
        }}
        .card-value {{
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 4px;
        }}
        .card-label {{
            color: #8b949e;
            font-size: 13px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: #161b22;
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 24px;
        }}
        th {{
            background: #21262d;
            padding: 12px 16px;
            text-align: left;
            font-size: 13px;
            color: #8b949e;
            text-transform: uppercase;
        }}
        td {{
            padding: 10px 16px;
            border-top: 1px solid #21262d;
            font-size: 13px;
        }}
        tr:hover td {{ background: #1c2128; }}
        .badge {{
            background: #21262d;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
        }}
        .section {{
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>

<div class="header">
    <h1>🔍 DocuSense AI — Evaluation Report</h1>
    <p class="meta">Generated: {timestamp}</p>
    <div class="status-badge">
        Baseline: {status_text}
    </div>
</div>

<h2>📊 Key Metrics</h2>
<div class="grid">
    <div class="card">
        <div class="card-value"
             style="color:{score_color(report.correct_doc_rate)}">
            {pct(report.correct_doc_rate)}
        </div>
        <div class="card-label">Correct Doc Rate</div>
    </div>
    <div class="card">
        <div class="card-value"
             style="color:{score_color(report.citation_rate)}">
            {pct(report.citation_rate)}
        </div>
        <div class="card-label">Citation Rate</div>
    </div>
    <div class="card">
        <div class="card-value"
             style="color:{score_color(report.avg_confidence)}">
            {pct(report.avg_confidence)}
        </div>
        <div class="card-label">Avg Confidence</div>
    </div>
    <div class="card">
        <div class="card-value"
             style="color:{score_color(report.avg_keyword_overlap)}">
            {pct(report.avg_keyword_overlap)}
        </div>
        <div class="card-label">Keyword Overlap</div>
    </div>
    <div class="card">
        <div class="card-value">
            {report.avg_processing_time}s
        </div>
        <div class="card-label">Avg Response Time</div>
    </div>
    <div class="card">
        <div class="card-value">
            {report.successful}/{report.total_questions}
        </div>
        <div class="card-label">Successful</div>
    </div>
</div>

<h2>🎯 Baseline Check</h2>
<div class="section">
<table>
<thead>
    <tr>
        <th>Metric</th>
        <th>Required</th>
        <th>Achieved</th>
        <th>Status</th>
    </tr>
</thead>
<tbody>
    <tr>
        <td>Correct Doc Rate</td>
        <td>{pct(report.baseline_metrics['correct_doc_rate'])}</td>
        <td>{pct(report.correct_doc_rate)}</td>
        <td>{"✅" if report.correct_doc_rate >= report.baseline_metrics['correct_doc_rate'] else "❌"}</td>
    </tr>
    <tr>
        <td>Citation Rate</td>
        <td>{pct(report.baseline_metrics['citation_rate'])}</td>
        <td>{pct(report.citation_rate)}</td>
        <td>{"✅" if report.citation_rate >= report.baseline_metrics['citation_rate'] else "❌"}</td>
    </tr>
    <tr>
        <td>Avg Confidence</td>
        <td>{pct(report.baseline_metrics['avg_confidence'])}</td>
        <td>{pct(report.avg_confidence)}</td>
        <td>{"✅" if report.avg_confidence >= report.baseline_metrics['avg_confidence'] else "❌"}</td>
    </tr>
    <tr>
        <td>Keyword Overlap</td>
        <td>{pct(report.baseline_metrics['avg_keyword_overlap'])}</td>
        <td>{pct(report.avg_keyword_overlap)}</td>
        <td>{"✅" if report.avg_keyword_overlap >= report.baseline_metrics['avg_keyword_overlap'] else "❌"}</td>
    </tr>
</tbody>
</table>
</div>

<h2>📂 Results by Category</h2>
<table>
<thead>
    <tr>
        <th>Category</th><th>Count</th>
        <th>Keyword Overlap</th>
        <th>Confidence</th>
        <th>Correct Doc</th>
    </tr>
</thead>
<tbody>{cat_rows}</tbody>
</table>

<h2>📋 Individual Results</h2>
<table>
<thead>
    <tr>
        <th></th><th>Question</th><th>Category</th>
        <th>Difficulty</th><th>Overlap</th>
        <th>Conf</th><th>Doc</th>
        <th>Citations</th><th>Time</th>
    </tr>
</thead>
<tbody>{result_rows}</tbody>
</table>

</body>
</html>"""

        with open(path, "w", encoding="utf-8") as f:
            f.write(html)

        # Also save as latest
        latest = REPORTS_DIR / "latest.html"
        with open(latest, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"HTML report: {path.name}")

    def print_console(self, report):
        """Print summary to terminal"""
        status = "✅ PASSED" if report.passed_baseline else "❌ FAILED"

        print("\n" + "=" * 60)
        print("  DOCUSENSE AI — EVALUATION REPORT")
        print("=" * 60)
        print(f"\n  Baseline: {status}")
        print(f"\n  Questions : {report.total_questions}")
        print(f"  Successful: {report.successful}")
        print(f"  Failed    : {report.failed}")
        print("\n  METRICS:")
        print(f"  {'Correct Doc Rate':<25} "
              f"{report.correct_doc_rate:.1%}")
        print(f"  {'Citation Rate':<25} "
              f"{report.citation_rate:.1%}")
        print(f"  {'Avg Confidence':<25} "
              f"{report.avg_confidence:.1%}")
        print(f"  {'Keyword Overlap':<25} "
              f"{report.avg_keyword_overlap:.1%}")
        print(f"  {'Avg Response Time':<25} "
              f"{report.avg_processing_time:.2f}s")
        print("\n  BY CATEGORY:")
        for cat, scores in report.category_scores.items():
            print(
                f"  {cat:<22} "
                f"overlap={scores['keyword_overlap']:.2f} "
                f"conf={scores['confidence']:.2f}"
            )
        print("\n" + "=" * 60)
        print(f"  Reports saved to: evals/benchmark_results/")
        print("=" * 60)