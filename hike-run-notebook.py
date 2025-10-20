import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import requests
    import pandas as pd
    from io import StringIO
    import calendar

    url = "https://www.cmiles.info/Data/anonymousActivityData.json"
    response = requests.get(url)
    response.raise_for_status()

    lines = pd.read_json(StringIO(response.text))
    lines = lines.query('activityType == "On Foot"')

    lines['folder'] = pd.Categorical(lines['folder'].astype(str))
    lines['activityType'] = pd.Categorical(lines['activityType'])

    lines['start'] = pd.to_datetime(lines['start'])
    lines['end'] = pd.to_datetime(lines['end'])

    startYearRange = range(lines['start'].min().year, lines['start'].max().year + 1)
    lines['startYear'] = pd.Categorical(lines['start'].dt.year, categories=startYearRange, ordered=True)

    lines['durationMinutes'] = (lines['end'].subtract(lines['start']).dt.total_seconds() / 60).round(0)

    lines['startMonth'] = pd.Categorical(lines['start'].dt.month_name(), categories=list(calendar.month_name)[1:], ordered=True)
    lines['startWeekday'] = pd.Categorical(lines['start'].dt.day_name(), categories=list(calendar.day_name), ordered=True)
    lines['startHour'] = pd.Categorical(lines['start'].dt.hour, categories=range(0,25), ordered=True)

    lines
    return lines, pd


@app.cell
def _(lines, pd):
    import marimo as mo
    import altair as alt
    import date_tools as date_tools
    from datetime import date
    import importlib

    importlib.reload(date_tools)

    lastFourWeeksPreviousYears = date_tools.previous_weeks_years_back(date.today(), weeks_back = 4, years_back = 10)

    linesLastFourWeeks = date_tools.filter_in_ranges(lines, 'start', lastFourWeeksPreviousYears)

    summaryLastFourWeeks = linesLastFourWeeks.groupby('matchStart').agg(
        totalDistanceMiles=('distanceMiles', 'sum'),
        totalActivities=('folder', 'count'),
        totalDurationMinutes=('durationMinutes', 'sum'),
        totalClimbFeet=('climbFeet', 'sum')).reset_index()

    summaryLastFourWeeks['totalDurationHours'] = (summaryLastFourWeeks['totalDurationMinutes'] / 60).round(0)
    summaryLastFourWeeks['matchLabel'] = summaryLastFourWeeks['matchStart'].dt.date.astype(str)

    med = summaryLastFourWeeks.median(numeric_only=True)

    summaryLastFourWeeksMedianRow = {
        "matchStart": pd.NaT,
        "totalDistanceMiles": med["totalDistanceMiles"],
        "totalActivities": med["totalActivities"],
        "totalDurationHours": med["totalDurationHours"],
        "totalDurationMinutes": med["totalDurationMinutes"],
        "totalClimbFeet": med["totalClimbFeet"],
        "matchLabel": "10 Year Median",
    }

    summaryLastFourWeeksLatestRow = summaryLastFourWeeks.sort_values("matchStart", ascending=False).iloc[0]

    summaryWithMedian = pd.concat(
        [summaryLastFourWeeks, pd.DataFrame([summaryLastFourWeeksMedianRow])],
        ignore_index=True
    )

    summaryWithMedian
    return mo, summaryLastFourWeeks, summaryWithMedian


@app.cell
def _(mo, pd, summaryLastFourWeeks, summaryWithMedian):
    import plotly.graph_objects as go

    # --- Build comparison data: Median + latest 3 weeks ---
    median_only = summaryWithMedian.loc[summaryWithMedian["matchLabel"] == "10 Year Median"].copy()
    last3 = summaryLastFourWeeks.sort_values("matchStart", ascending=False).head(3).copy()

    last3_labels = last3["matchStart"].dt.date.astype(str).tolist()
    category_order = ["10 Year Median", *last3_labels][::-1]

    compare_rows = pd.concat([median_only, last3], ignore_index=True)

    def comparison_frame(metric_col: str) -> pd.DataFrame:
        return compare_rows.rename(columns={metric_col: "value"}).copy()

    def comparison_bar_trace(df_metric: pd.DataFrame, title, suffix=""):
        return go.Bar(
            y=df_metric["matchLabel"].astype(str),
            x=df_metric["value"],
            orientation="h",
            textangle=0,
            text=[f"{v:.0f}{(' ' + suffix) if suffix else ''}" for v in df_metric["value"]],
            textposition="auto",
            name=title,
            showlegend=False,
        )

    def build_bar_fig(
        df_metric: pd.DataFrame, title, suffix="",
        category_order=None, *, width=480, height=150, margins=(70,16,20,16)  # l,r,t,b
    ):
        fig = go.Figure()
        fig.add_trace(comparison_bar_trace(df_metric, title, suffix))
        if category_order is None:
            category_order = df_metric["matchLabel"].astype(str).tolist()
        l, r, t, b = margins
        fig.update_layout(
            title=None,         # title sits in the card
            autosize=False,
            width=width,
            height=height,
            margin=dict(l=l, r=r, t=t, b=b),
        )
        fig.update_yaxes(
            type="category",
            categoryorder="array",
            categoryarray=category_order
        )
        # compact x-axis
        fig.update_xaxes(ticks="outside", showgrid=True, griddash="dot")
        return fig

    def gauge_trace(
        latest_value, median_value, suffix="",
        number_font_size=12, delta_font_size=10, rng=None,
        tolerance_ratio=0.03
    ):
        v = float(latest_value) if pd.notna(latest_value) else 0.0
        m = float(median_value) if pd.notna(median_value) else 0.0
        upper = max(v, m, 1.0)
        if rng is None:
            rng = (0, upper * 1.2)

        # --- softer motivational palette ---
        color_below = "#E88B5A"   # warm clay/orange-rose
        color_near  = "#B7C9A9"   # gentle sage
        color_above = "#5CA793"   # uplifting teal-green

        if m == 0:
            bar_color = color_near
        else:
            diff_ratio = (v - m) / m
            if abs(diff_ratio) <= tolerance_ratio:
                bar_color = color_near
            elif v > m:
                bar_color = color_above
            else:
                bar_color = color_below

        return go.Indicator(
            mode="gauge+number+delta",
            value=v,
            delta=dict(
                reference=m,
                relative=False,
                font={"size": delta_font_size},
                increasing={"color": color_above},
                decreasing={"color": color_below},
            ),
            title={"text": ""},
            number={"suffix": f" {suffix}" if suffix else "", "font": {"size": number_font_size}},
            gauge=dict(
                axis={"range": [rng[0], rng[1]]},
                bar={"thickness": 0.45, "color": bar_color},
                threshold={"line": {"color": "#444", "width": 1.5}, "thickness": 0.9, "value": m}
            ),
            domain={"x": [0, 1], "y": [0, 1]},
        )

    def build_gauge_fig(
        latest_value, median_value, title="", suffix="",
        *, width=150, height=130, margins=(24, 32, 12, 12),
        number_font_size=12, delta_font_size=10
    ):
        """
        Smaller fonts + slightly wider margins prevent number overlap with gauge.
        """
        fig = go.Figure()
        fig.add_trace(gauge_trace(latest_value, median_value, suffix, number_font_size, delta_font_size))
        l, r, t, b = margins
        fig.update_layout(
            autosize=False,
            width=width,
            height=height,
            margin=dict(l=l, r=r, t=t, b=b),
        )
        return fig


    latest = last3.iloc[0]
    med_vals = median_only.iloc[0]

    # Metric figs (sizes controlled here)
    gauge_size = dict(width=160, height=150, margins=(22, 30, 12, 12), number_font_size=16, delta_font_size=12)
    bar_size   = dict(width=520, height=150, margins=(80, 18, 20, 16))

    # Distance
    dist_df = comparison_frame("totalDistanceMiles")
    gauge_fig_dist = build_gauge_fig(latest["totalDistanceMiles"], med_vals["totalDistanceMiles"], "mi", **gauge_size)
    bar_fig_dist   = build_bar_fig(dist_df, "Distance", "mi", category_order=category_order, **bar_size)

    # Activities
    act_df = comparison_frame("totalActivities")
    gauge_fig_act = build_gauge_fig(latest["totalActivities"], med_vals["totalActivities"], "", **gauge_size)
    bar_fig_act   = build_bar_fig(act_df, "Activities", "", category_order=category_order, **bar_size)

    # Duration (convert minutes -> hours)
    dur_df = comparison_frame("totalDurationMinutes").assign(value=lambda d: d["value"]/60.0)
    gauge_fig_dur = build_gauge_fig(latest["totalDurationMinutes"]/60.0, med_vals["totalDurationMinutes"]/60.0, "h", **gauge_size)
    bar_fig_dur   = build_bar_fig(dur_df, "Duration", "h", category_order=category_order, **bar_size)

    # Climb
    clm_df = comparison_frame("totalClimbFeet")
    gauge_fig_clm = build_gauge_fig(latest["totalClimbFeet"], med_vals["totalClimbFeet"], "ft", **gauge_size)
    bar_fig_clm   = build_bar_fig(clm_df, "Climb", "ft", category_order=category_order, **bar_size)

    # --- Marimo render: each card is its own ROW + a single title above all rows ---
    def render_panel(title, gauge_fig, bar_fig):
        return f"""
        <div class="panel">
          <div class="panel-title">{title}</div>
          <div class="panel-gauge">{mo.as_html(gauge_fig)}</div>
          <div class="panel-bar">{mo.as_html(bar_fig)}</div>
        </div>
        """

    def render_flex_dashboard(items, big_title="Last Four Weeks", gap_row=12):
        cards_html = "\n".join(render_panel(t, g, b) for (t, g, b) in items)
        return mo.md(f"""
    <style>
      .dash-title {{
        margin: 0 0 10px 0;
        font-size: 1.4rem;
        font-weight:600;
        text-align:center;
      }}
      /* whole page stack of rows */
      .flex-wrap-panels {{
        display:flex;
        flex-direction:column;     /* each card on its own row */
        row-gap:{gap_row}px;
      }}
      /* a single row/card, laid out horizontally */
      .panel {{
        display:flex;
        flex-direction:row;        /* Title -> Gauge -> Bar */
        flex-wrap:nowrap;
        justify-content:center;
        align-items:center;
        gap:12px;
        padding:8px 12px;
        border:1px solid #ddd;
        border-radius:10px;
        box-shadow:0 1px 3px rgba(0,0,0,0.08);
        background:#fff;
        width:100%;
      }}
      .panel-title {{
        width:110px;           /* give the title room */
        font-size:0.95rem;
        font-weight:600;
        text-align:left;
        white-space:nowrap;        /* keep on one line */
      }}
      /* Plotly figures define size; no CSS sizing here */
      .panel-gauge, .panel-bar {{ overflow:hidden; }}
    </style>

    <p class="dash-title">{big_title}</p>

    <div class="flex-wrap-panels">
      {cards_html}
    </div>
    """)

    items = [
        ("Distance",  gauge_fig_dist, bar_fig_dist),
        ("Activities", gauge_fig_act, bar_fig_act),
        ("Duration",  gauge_fig_dur, bar_fig_dur),
        ("Climb",     gauge_fig_clm, bar_fig_clm),
    ]

    render_flex_dashboard(items, big_title="Last Four Weeks vs. 10 Year Median")
    return


if __name__ == "__main__":
    app.run()
