"""
dashboard.py
Generates a clean HTML coach dashboard with:
  - Per-player stats table (distance, speed, zones)
  - Bar charts for distance and speed
  - Embedded heatmap images
  - Match summary

Usage:
  Called automatically by detector.py after video processing.
  Or run standalone:
    python src/dashboard.py --stats output/match_stats.csv --heatmaps output/heatmaps/
"""
from typing import Optional
import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse
import json


def generate_dashboard(
    df: pd.DataFrame,
    output_path: str,
    heatmap_dir: Optional[str] = None,
    match_name: str = "Basketball Match",
    video_path: str = "",
) -> str:
    """
    Generate a self-contained HTML dashboard file.
    Returns the path to the saved HTML file.
    """
    from typing import Optional

    if df.empty:
        print("[WARN] No player stats to display.")
        return ""

    # Prepare chart data
    player_ids  = df["Player ID"].tolist()
    distances   = df["Distance (m)"].tolist()
    avg_speeds  = df["Avg Speed (m/s)"].tolist()
    max_speeds  = df["Max Speed (m/s)"].tolist()
    teams       = df["Team"].tolist()

    # Team color mapping
    unique_teams = list(dict.fromkeys(teams))
    team_colors_map = {
        unique_teams[0]: "#3b82f6" if len(unique_teams) > 0 else "#gray",
        unique_teams[1]: "#ef4444" if len(unique_teams) > 1 else "#gray",
    }
    bar_colors = [team_colors_map.get(t, "#6b7280") for t in teams]

    # Heatmap cards HTML
    heatmap_html = ""
    if heatmap_dir and Path(heatmap_dir).exists():
        heatmaps = sorted(Path(heatmap_dir).glob("*.png"))
        if heatmaps:
            cards = ""
            for hm in heatmaps:
                rel = Path(heatmap_dir).name + "/" + hm.name
                label = hm.stem.replace("_", " ").title()
                cards += f"""
                <div class="heatmap-card">
                    <img src="{rel}" alt="{label}" loading="lazy"/>
                    <p>{label}</p>
                </div>"""
            heatmap_html = f"""
            <section class="section">
                <h2>🗺️ Court Heatmaps</h2>
                <div class="heatmap-grid">{cards}</div>
            </section>"""

    # Stats table rows
    table_rows = ""
    for _, row in df.iterrows():
        team_color = team_colors_map.get(row["Team"], "#6b7280")
        dominant = row.get("Dominant Zone", "—")
        table_rows += f"""
        <tr>
            <td><strong>{row['Player ID']}</strong></td>
            <td><span class="team-badge" style="background:{team_color}">{row['Team']}</span></td>
            <td>{row['Time on Court (s)']}s</td>
            <td><strong>{row['Distance (m)']}m</strong></td>
            <td>{row['Avg Speed (m/s)']} m/s</td>
            <td>{row['Max Speed (m/s)']} m/s</td>
            <td>{dominant}</td>
        </tr>"""

    # Summary cards
    total_players = len(df)
    teams_found   = df["Team"].nunique()
    top_runner    = df.iloc[0]["Player ID"] if not df.empty else "—"
    top_dist      = df.iloc[0]["Distance (m)"] if not df.empty else 0
    fastest       = df.loc[df["Max Speed (m/s)"].idxmax(), "Player ID"] if not df.empty else "—"
    fastest_speed = df["Max Speed (m/s)"].max() if not df.empty else 0

    now = datetime.now().strftime("%B %d, %Y — %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>🏀 {match_name} — Coach Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: #0f172a; color: #e2e8f0; min-height: 100vh; }}
  header {{ background: linear-gradient(135deg, #1e3a5f, #0f172a);
            padding: 28px 40px; border-bottom: 2px solid #1e40af; }}
  header h1 {{ font-size: 1.8rem; color: #f8fafc; }}
  header p  {{ color: #94a3b8; margin-top: 4px; font-size: 0.9rem; }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 32px 24px; }}
  .section {{ margin-bottom: 40px; }}
  .section h2 {{ font-size: 1.2rem; color: #93c5fd; margin-bottom: 16px;
                 border-left: 3px solid #3b82f6; padding-left: 12px; }}
  .summary-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                    gap: 16px; margin-bottom: 40px; }}
  .card {{ background: #1e293b; border-radius: 12px; padding: 20px;
           border: 1px solid #334155; }}
  .card .value {{ font-size: 2rem; font-weight: 700; color: #60a5fa; }}
  .card .label {{ font-size: 0.8rem; color: #94a3b8; margin-top: 4px; text-transform: uppercase; }}
  .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
  .chart-box {{ background: #1e293b; border-radius: 12px; padding: 20px;
                border: 1px solid #334155; }}
  .chart-box h3 {{ font-size: 0.95rem; color: #cbd5e1; margin-bottom: 16px; }}
  table {{ width: 100%; border-collapse: collapse; background: #1e293b;
           border-radius: 12px; overflow: hidden; border: 1px solid #334155; }}
  thead {{ background: #0f2240; }}
  th {{ padding: 12px 16px; text-align: left; font-size: 0.8rem;
        color: #93c5fd; text-transform: uppercase; letter-spacing: 0.05em; }}
  td {{ padding: 12px 16px; border-top: 1px solid #1e3a5f; font-size: 0.9rem; }}
  tr:hover td {{ background: #162032; }}
  .team-badge {{ display: inline-block; padding: 2px 10px; border-radius: 99px;
                 font-size: 0.75rem; font-weight: 600; color: white; }}
  .heatmap-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
                   gap: 20px; }}
  .heatmap-card {{ background: #1e293b; border-radius: 12px; overflow: hidden;
                   border: 1px solid #334155; }}
  .heatmap-card img {{ width: 100%; display: block; }}
  .heatmap-card p {{ padding: 10px 14px; font-size: 0.85rem; color: #94a3b8; }}
  @media (max-width: 768px) {{
    .charts {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<header>
  <h1>🏀 {match_name}</h1>
  <p>Generated {now}{' | Source: ' + video_path if video_path else ''}</p>
</header>

<div class="container">

  <!-- Summary Cards -->
  <div class="summary-cards">
    <div class="card">
      <div class="value">{total_players}</div>
      <div class="label">Players Tracked</div>
    </div>
    <div class="card">
      <div class="value">{teams_found}</div>
      <div class="label">Teams Detected</div>
    </div>
    <div class="card">
      <div class="value">{top_runner}</div>
      <div class="label">Most Distance ({top_dist}m)</div>
    </div>
    <div class="card">
      <div class="value">{fastest}</div>
      <div class="label">Fastest ({fastest_speed} m/s)</div>
    </div>
    <div class="card">
      <div class="value">{df['Distance (m)'].sum():.0f}m</div>
      <div class="label">Total Distance (all players)</div>
    </div>
  </div>

  <!-- Charts -->
  <section class="section">
    <h2>📊 Player Performance Charts</h2>
    <div class="charts">
      <div class="chart-box">
        <h3>Distance Covered per Player (m)</h3>
        <canvas id="distChart"></canvas>
      </div>
      <div class="chart-box">
        <h3>Max Speed per Player (m/s)</h3>
        <canvas id="speedChart"></canvas>
      </div>
    </div>
  </section>

  <!-- Stats Table -->
  <section class="section">
    <h2>📋 Full Player Stats</h2>
    <table>
      <thead>
        <tr>
          <th>Player</th><th>Team</th><th>Time</th>
          <th>Distance</th><th>Avg Speed</th><th>Max Speed</th><th>Dominant Zone</th>
        </tr>
      </thead>
      <tbody>{table_rows}</tbody>
    </table>
  </section>

  {heatmap_html}

</div>

<script>
const labels  = {json.dumps(player_ids)};
const dists   = {json.dumps(distances)};
const speeds  = {json.dumps(max_speeds)};
const colors  = {json.dumps(bar_colors)};

new Chart(document.getElementById('distChart'), {{
  type: 'bar',
  data: {{ labels, datasets: [{{ label: 'Distance (m)', data: dists,
    backgroundColor: colors, borderRadius: 6 }}] }},
  options: {{ responsive: true, plugins: {{ legend: {{ display: false }} }},
    scales: {{ x: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#1e3a5f' }} }},
               y: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#1e3a5f' }} }} }} }}
}});

new Chart(document.getElementById('speedChart'), {{
  type: 'bar',
  data: {{ labels, datasets: [{{ label: 'Max Speed (m/s)', data: speeds,
    backgroundColor: colors, borderRadius: 6 }}] }},
  options: {{ responsive: true, plugins: {{ legend: {{ display: false }} }},
    scales: {{ x: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#1e3a5f' }} }},
               y: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#1e3a5f' }} }} }} }}
}});
</script>
</body>
</html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[INFO] Dashboard saved to: {output_path}")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate coach dashboard from stats CSV")
    parser.add_argument("--stats", required=True, help="Path to match_stats.csv")
    parser.add_argument("--heatmaps", default=None, help="Path to heatmaps folder")
    parser.add_argument("--output", default="output/dashboard.html", help="Output HTML path")
    parser.add_argument("--match", default="Basketball Match", help="Match name")
    return parser.parse_args()




if __name__ == "__main__":
    args = parse_args()
    df = pd.read_csv(args.stats)
    generate_dashboard(df, args.output, args.heatmaps, args.match)
    print(f"Open in browser: {args.output}")
