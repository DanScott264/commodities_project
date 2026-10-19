"""
U.S. crude oil balance and inventory analysis

Rebuilds the weekly U.S. crude oil supply/demand balance from EIA data, checks it
against reported commercial inventory changes, compares stocks with their 5-year
seasonal norm, links Cushing stocks to the WTI time spread, and writes a short
weekly market snapshot.

Engineering framing: the U.S. crude system is treated as a control volume.
    production + imports - exports - refinery runs  =  change in total stocks
Total stocks = commercial stocks + Strategic Petroleum Reserve (SPR), and whatever
the reported flows can't explain is EIA's balancing "adjustment" (the unmetered flow).

Usage
-----
    python3 oil_analysis.py           # saves charts to figures/ and a summary to output/
    python3 oil_analysis.py --show    # also opens the charts on screen

Data files go in a data/ folder next to this script (or next to the script itself).
See README.md for where to download each file.
"""

import argparse
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
FIG_DIR = PROJECT_DIR / "figures"
OUT_DIR = PROJECT_DIR / "output"

ANALYSIS_START = "2018-01-01"   # main analysis window
SEASONAL_YEARS = 5              # length of the seasonal baseline
ALT_BASELINE_EXCLUDES = (2020,)  # the COVID glut distorts any 5-year average containing 2020
FORWARD_WEEKS = 4               # horizon for forward WTI returns
RELEASE_LAG_WEEKS = 1           # data for the week ending Friday t is published the following Wednesday
CUSHING_BIN_MB = 5              # bin width for the Cushing vs time-spread relationship

FILES = {
    # required
    "wti": "wti.csv",                     # FRED DCOILWTICO (daily WTI spot)
    "inventories": "inventories.csv",     # EIA WCESTUS1 (commercial stocks excl. SPR)
    "production": "production.xls",       # EIA WCRFPUS2
    "refinery": "refinery_inputs.xls",    # EIA refiner net input of crude
    "imports": "imports.xls",             # EIA WCRIMUS2
    "exports": "exports.xls",             # EIA crude exports (sheet "Data 2")
    # optional - the script runs without them and skips the related analysis
    "spr": "spr.xls",                     # EIA WCSSTUS1 (SPR stocks)
    "adjustment": "adjustment.xls",       # EIA weekly crude oil adjustment
    "cushing": "cushing.xls",             # EIA Cushing, OK stocks
    "futures": "futures.xls",             # EIA NYMEX WTI contracts 1-4 (history ends Apr 2024)
}

# Column selections for the required Excel files (same as the original script).
REFINERY_COLUMN = "Weekly U.S. Refiner Net Input of Crude Oil (Thousand Barrels per Day)"
EXPORTS_SHEET = "Data 2"
EXPORTS_COLUMN = "Weekly U.S. Exports of Crude Oil (Thousand Barrels per Day)"

# Chart palette: one validated categorical order, recessive chrome.
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
LOOSENS, TIGHTENS = "#2a78d6", "#e34948"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def resolve_path(data_dir, filename):
    """Return the file path, accepting .xls/.xlsx interchangeably. None if missing."""
    path = data_dir / filename
    if path.exists():
        return path
    if path.suffix in (".xls", ".xlsx"):
        alt = path.with_suffix(".xlsx" if path.suffix == ".xls" else ".xls")
        if alt.exists():
            return alt
    return None


def normalise(text):
    return " ".join(str(text).split()).lower()


def match_column(columns, wanted):
    """Find a column by position, exact name, or whitespace-insensitive name."""
    if isinstance(wanted, int):
        return columns[wanted]
    for col in columns:
        if normalise(col) == normalise(wanted):
            return col
    raise KeyError(f"Column '{wanted}' not found. Available: {list(columns)}")


def clean_series(df, name):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df[name] = pd.to_numeric(df[name], errors="coerce")
    df = df.dropna(subset=["Date", name])
    df = df.sort_values("Date").drop_duplicates("Date", keep="last")
    return df.reset_index(drop=True)


def load_fred_csv(path, name):
    df = pd.read_csv(path).iloc[:, :2]
    df.columns = ["Date", name]
    return clean_series(df, name)


def load_eia_csv(path, name, skiprows=4):
    df = pd.read_csv(path, skiprows=skiprows).iloc[:, :2]
    df.columns = ["Date", name]
    return clean_series(df, name)


def load_eia_excel(path, name, sheet="Data 1", column=1):
    df = pd.read_excel(path, sheet_name=sheet, header=2)
    value_col = match_column(df.columns, column)
    df = df[[df.columns[0], value_col]]
    df.columns = ["Date", name]
    return clean_series(df, name)


def find_eia_column(path, include, exclude=()):
    """Search every 'Data' sheet for the first column containing all `include` words."""
    book = pd.ExcelFile(path)
    for sheet in book.sheet_names:
        if not sheet.lower().startswith("data"):
            continue
        header = book.parse(sheet, header=2, nrows=0).columns
        for col in header[1:]:
            text = normalise(col)
            if all(w in text for w in include) and not any(w in text for w in exclude):
                return sheet, col
    return None, None


def load_required_excel(path, name, sheet, column, search, exclude=()):
    """Load from the expected sheet/column; if the layout differs (e.g. a single-series
    download instead of a table), search the file for a matching column instead."""
    try:
        return load_eia_excel(path, name, sheet=sheet, column=column)
    except (KeyError, ValueError, IndexError):
        found_sheet, col = find_eia_column(path, search, exclude)
        if col is None:
            raise KeyError(f"Couldn't find the {name} column in {path.name}")
        return load_eia_excel(path, name, sheet=found_sheet, column=col)


def load_optional(data_dir, key, name, include_options, exclude=()):
    """Load an optional EIA series by searching its file for a matching column."""
    path = resolve_path(data_dir, FILES[key])
    if path is None:
        print(f"  [optional] {FILES[key]} not found - skipping {name}")
        return None
    for include in include_options:
        sheet, col = find_eia_column(path, include, exclude)
        if col is not None:
            series = load_eia_excel(path, name, sheet=sheet, column=col)
            print(f"  [optional] {name}: '{' '.join(str(col).split())}'")
            return series
    print(f"  [optional] no matching column in {path.name} - skipping {name}")
    return None


def to_weekly_friday(df, name):
    """Daily series -> last available value in each week ending Friday."""
    weekly = df.set_index("Date")[name].resample("W-FRI").last().dropna()
    return weekly.reset_index()


# ---------------------------------------------------------------------------
# Seasonal baseline
# ---------------------------------------------------------------------------

def seasonal_baseline(df, value_col, n_years=SEASONAL_YEARS, exclude_years=()):
    """Average level for the same ISO week over the previous n available years.

    Week 53 is folded into week 52. Years in `exclude_years` are skipped, so the
    window reaches one year further back instead.
    """
    work = df[["Date", value_col]].copy()
    iso = work["Date"].dt.isocalendar()
    work["iso_year"] = iso["year"].astype(int)
    work["iso_week"] = iso["week"].astype(int).clip(upper=52)

    pivot = work.pivot_table(index="iso_year", columns="iso_week",
                             values=value_col, aggfunc="mean")
    years = list(pivot.index)
    rows = []
    for year in years:
        prior = [y for y in years if y < year and y not in exclude_years][-n_years:]
        if len(prior) < n_years:
            continue
        avg = pivot.loc[prior].mean(axis=0, skipna=False)
        rows.extend((year, week, value) for week, value in avg.items() if pd.notna(value))

    base = pd.DataFrame(rows, columns=["iso_year", "iso_week", "baseline"])
    merged = work.merge(base, on=["iso_year", "iso_week"], how="left")
    return pd.Series(merged["baseline"].to_numpy(), index=df.index)


# ---------------------------------------------------------------------------
# Building the weekly dataset
# ---------------------------------------------------------------------------

def build_dataset(data_dir):
    print(f"Loading data from {data_dir}")
    paths = {}
    for key in ["wti", "inventories", "production", "refinery", "imports", "exports"]:
        path = resolve_path(data_dir, FILES[key])
        if path is None and key == "inventories":
            path = resolve_path(data_dir, "inventories.xls")   # the EIA Excel download also works
        if path is None:
            raise FileNotFoundError(f"Required file '{FILES[key]}' not found in {data_dir}")
        paths[key] = path

    wti = to_weekly_friday(load_fred_csv(paths["wti"], "WTI"), "WTI")
    wti.loc[wti["WTI"] <= 0, "WTI"] = np.nan   # guard for the April 2020 negative print

    if paths["inventories"].suffix == ".csv":
        inventories = load_eia_csv(paths["inventories"], "Inventories")
    else:
        inventories = load_required_excel(paths["inventories"], "Inventories", "Data 1", 1,
                                          ["stocks", "crude"], exclude=["spr)", "in spr"])
    production = load_eia_excel(paths["production"], "Production", column=1)
    refinery = load_required_excel(paths["refinery"], "Refinery_inputs", "Data 1",
                                   REFINERY_COLUMN, ["refiner net input of crude"])
    imports = load_eia_excel(paths["imports"], "Imports", column=1)
    exports = load_required_excel(paths["exports"], "Exports", EXPORTS_SHEET, EXPORTS_COLUMN,
                                  ["exports of crude oil"], exclude=["products"])

    # Weekly changes and seasonal baselines use the FULL history, before any
    # filtering, so the first analysis week and the seasonal average are valid.
    inventories["Inventory_change_mb"] = inventories["Inventories"].diff() / 1000
    inventories["Seasonal_avg"] = seasonal_baseline(inventories, "Inventories")
    inventories["Seasonal_avg_ex2020"] = seasonal_baseline(
        inventories, "Inventories", exclude_years=ALT_BASELINE_EXCLUDES)

    first_needed = pd.Timestamp(ANALYSIS_START) - pd.DateOffset(years=SEASONAL_YEARS)
    if inventories["Date"].min() > first_needed:
        print(f"  Note: inventories.csv starts {inventories['Date'].min().date()}; download "
              f"the full history so the 5-year average covers the whole analysis window.")

    data = inventories
    for frame in [production, refinery, imports, exports]:
        data = data.merge(frame, on="Date", how="inner")

    spr = load_optional(data_dir, "spr", "SPR", [["in spr"], ["spr"]], exclude=["excl"])
    if spr is not None:
        spr["SPR_change_mb"] = spr["SPR"].diff() / 1000
        data = data.merge(spr, on="Date", how="left")

    adjustment = load_optional(data_dir, "adjustment", "Adjustment",
                               [["crude", "adjustment"], ["unaccounted"]],
                               exclude=["gasoline"])
    if adjustment is not None:
        data = data.merge(adjustment, on="Date", how="left")

    # WTI joins last and as a left join, so a stale price file never drops EIA weeks.
    data = data.merge(wti, on="Date", how="left")
    data = data[data["Date"] >= ANALYSIS_START].sort_values("Date").reset_index(drop=True)

    gaps = int((data["Date"].diff().dt.days.dropna() != 7).sum())
    if gaps:
        print(f"  Warning: {gaps} gap(s) in the weekly series - check all files cover the same weeks")

    data = add_derived_columns(data)
    print(f"  {len(data)} weeks: {data['Date'].min().date()} to {data['Date'].max().date()}\n")
    return data


def add_derived_columns(df):
    df = df.copy()

    # The control-volume balance from reported flows (thousand barrels/day).
    df["Balance_kbd"] = (df["Production"] + df["Imports"]
                         - df["Refinery_inputs"] - df["Exports"])
    df["Implied_flows_mb"] = df["Balance_kbd"] * 7 / 1000

    # Commercial stocks = total - SPR, so SPR movements come off the implied change.
    if "SPR_change_mb" in df:
        df["SPR_change_kbd"] = df["SPR_change_mb"] * 1000 / 7
        df["SPR_release_kbd"] = -df["SPR_change_kbd"]
        df["Implied_commercial_mb"] = df["Implied_flows_mb"] - df["SPR_change_mb"]
    else:
        df["Implied_commercial_mb"] = df["Implied_flows_mb"]

    if "Adjustment" in df:
        df["Implied_with_adjustment_mb"] = df["Implied_commercial_mb"] + df["Adjustment"] * 7 / 1000

    # Best available estimate of the commercial balance, in kb/d.
    df["Commercial_balance_kbd"] = df["Implied_commercial_mb"] * 1000 / 7
    df["Commercial_balance_4w"] = df["Commercial_balance_kbd"].rolling(4).mean()
    df["Balance_4w"] = df["Balance_kbd"].rolling(4).mean()
    df["Unexplained_kbd"] = (df["Inventory_change_mb"] - df["Implied_commercial_mb"]) * 1000 / 7

    df["Inventory_vs_seasonal_mb"] = (df["Inventories"] - df["Seasonal_avg"]) / 1000
    df["Inventory_vs_seasonal_ex2020_mb"] = (df["Inventories"] - df["Seasonal_avg_ex2020"]) / 1000
    return df


def balance_versions(df):
    """The balance estimates available, from simplest to most complete."""
    versions = [("Reported flows only", "Implied_flows_mb")]
    if "SPR_change_mb" in df:
        versions.append(("Flows + SPR", "Implied_commercial_mb"))
    if "Implied_with_adjustment_mb" in df:
        label = "Flows + SPR + EIA adjustment" if "SPR_change_mb" in df else "Flows + EIA adjustment"
        versions.append((label, "Implied_with_adjustment_mb"))
    return versions


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def reconciliation_stats(df, implied_col):
    """How well an implied stock change reconciles with the reported change.

    All inputs come out in the same weekly EIA report, so this measures how well
    the published flows add up - it is not a forecast.
    """
    d = df[["Inventory_change_mb", implied_col]].dropna()
    actual, implied = d["Inventory_change_mb"], d[implied_col]
    resid = actual - implied
    actual_build, model_build = actual > 0, implied > 0
    corr = actual.corr(implied)
    hit = (actual_build == model_build).mean() * 100
    naive = max(actual_build.mean(), 1 - actual_build.mean()) * 100
    return {
        "Weeks": len(d),
        "Mean gap (kb/d)": resid.mean() * 1000 / 7,
        "MAE (mb/week)": resid.abs().mean(),
        "Correlation": corr,
        "R²": corr ** 2,
        "Direction hit rate (%)": hit,
        "Naive baseline (%)": naive,
        "Edge (pp)": hit - naive,
        "Hit rate on builds (%)": (model_build[actual_build]).mean() * 100,
        "Hit rate on draws (%)": (~model_build[~actual_build]).mean() * 100,
        "Weeks called as draws (%)": (~model_build).mean() * 100,
    }


def cumulative_reconciliation(df, versions):
    cols = ["Inventory_change_mb"] + [col for _, col in versions]
    d = df.dropna(subset=cols)
    out = pd.DataFrame({"Date": d["Date"], "Actual": d["Inventory_change_mb"].cumsum()})
    for label, col in versions:
        out[label] = d[col].cumsum()
    return out.reset_index(drop=True)


DRIVERS = [
    # column, chart label, phrase, sign in the balance
    ("Production", "Production", "production", 1),
    ("Imports", "Imports", "imports", 1),
    ("Refinery_inputs", "Refinery runs", "refinery runs", -1),
    ("Exports", "Exports", "exports", -1),
]


def balance_drivers(df, weeks=4):
    """Change in each flow's 4-week average vs 4 weeks earlier, signed by its effect on
    the commercial balance (+ loosens, - tightens)."""
    drivers = list(DRIVERS)
    if "SPR_release_kbd" in df:
        drivers.append(("SPR_release_kbd", "SPR releases", "SPR releases", 1))
    cols = [d[0] for d in drivers]
    avg = df.set_index("Date")[cols].rolling(weeks).mean()
    delta = (avg - avg.shift(weeks)).dropna()
    last, date = delta.iloc[-1], delta.index[-1]

    rows = []
    for col, label, phrase, sign in drivers:
        change = last[col]
        if col == "SPR_release_kbd":
            words = ("Larger", "Smaller")
        else:
            words = ("Higher", "Lower")
        rows.append({
            "Driver": label,
            "Change (kb/d)": change,
            "Contribution (kb/d)": sign * change,
            "Phrase": f"{words[0] if change > 0 else words[1]} {phrase}",
        })
    return pd.DataFrame(rows), date


def forward_return_regimes(df, balance_col="Commercial_balance_4w"):
    """Median forward WTI return by balance tercile.

    The return starts one week after the data week (when the report is public) and
    uses log returns. Descriptive only: overlapping windows, full-sample terciles.
    """
    d = df.copy()
    start = d["WTI"].shift(-RELEASE_LAG_WEEKS)
    end = d["WTI"].shift(-(RELEASE_LAG_WEEKS + FORWARD_WEEKS))
    d["WTI_fwd_pct"] = np.log(end / start) * 100
    d = d.dropna(subset=[balance_col, "WTI_fwd_pct"])
    if len(d) < 30:
        return None
    d["Regime"] = pd.qcut(d[balance_col], q=3, labels=["Tight", "Neutral", "Loose"])
    summary = d.groupby("Regime", observed=True)["WTI_fwd_pct"].agg(
        Weeks="count",
        Median="median",
        Mean="mean",
        Positive=lambda x: (x > 0).mean() * 100,
    )
    return summary.rename(columns={"Median": "Median fwd return (%)",
                                   "Mean": "Mean fwd return (%)",
                                   "Positive": "Share positive (%)"})


def build_timespread(data_dir):
    """Weekly Cushing stocks joined to the WTI front-month minus second-month spread."""
    cushing = load_optional(data_dir, "cushing", "Cushing", [["cushing"]])
    if cushing is None:
        return None, None
    cushing["Cushing_mb"] = cushing["Cushing"] / 1000
    cushing["Cushing_seasonal"] = seasonal_baseline(cushing, "Cushing")
    cushing["Cushing_vs_seasonal_mb"] = (cushing["Cushing"] - cushing["Cushing_seasonal"]) / 1000

    m1 = load_optional(data_dir, "futures", "M1", [["crude", "contract 1"]])
    m2 = load_optional(data_dir, "futures", "M2", [["crude", "contract 2"]])
    if m1 is None or m2 is None:
        return None, cushing

    daily = m1.merge(m2, on="Date", how="inner")
    daily["Spread"] = daily["M1"] - daily["M2"]
    spread = to_weekly_friday(daily, "Spread")
    ts = spread.merge(cushing[["Date", "Cushing_mb"]], on="Date", how="inner")
    return ts, cushing


def spread_relationship(ts, bin_mb=CUSHING_BIN_MB, min_weeks=5):
    lo = np.floor(ts["Cushing_mb"].min() / bin_mb) * bin_mb
    hi = np.ceil(ts["Cushing_mb"].max() / bin_mb) * bin_mb + bin_mb
    bins = np.arange(lo, hi, bin_mb)
    binned = ts.assign(Bin=pd.cut(ts["Cushing_mb"], bins)).groupby("Bin", observed=True).agg(
        Cushing_mb=("Cushing_mb", "median"),
        Spread=("Spread", "median"),
        Weeks=("Spread", "size"),
    )
    binned = binned[binned["Weeks"] >= min_weeks].reset_index()
    rho = ts["Cushing_mb"].rank().corr(ts["Spread"].rank())   # Spearman, no scipy needed
    return binned, rho


def implied_spread(binned, level):
    for _, row in binned.iterrows():
        if level in row["Bin"]:
            return row["Spread"]
    return None


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE, "axes.labelcolor": INK_2, "axes.linewidth": 0.8,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2, "text.color": INK,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
        "grid.linewidth": 0.8, "grid.linestyle": "-", "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "legend.frameon": False, "lines.linewidth": 2,
    })


def set_titles(ax, title, subtitle=None):
    """Takeaway title (bold) with a smaller descriptive subtitle underneath."""
    width_in = ax.figure.get_figwidth()
    sub_lines = textwrap.wrap(subtitle, int(width_in * 11)) if subtitle else []
    ax.set_title("\n".join(textwrap.wrap(title, int(width_in * 9))), loc="left", fontsize=12,
                 fontweight="bold", color=INK, pad=10 + 12 * len(sub_lines))
    if sub_lines:
        ax.annotate("\n".join(sub_lines), xy=(0, 1), xycoords="axes fraction",
                    xytext=(0, 6), textcoords="offset points", fontsize=9,
                    color=INK_2, va="bottom", ha="left")


def label_line_ends(ax, items, min_gap_frac=0.06):
    """Direct-label line ends, nudging labels apart so they don't overlap."""
    if not items:
        return
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap_frac
    items = sorted(items, key=lambda t: t[1])
    placed = []
    for x, y, text in items:
        if placed and y - placed[-1][1] < gap:
            y = placed[-1][1] + gap
        placed.append((x, y, text))
    for x, y, text in placed:
        ax.annotate(text, xy=(x, y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK_2)


def save(fig, name, show):
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    if not show:
        plt.close(fig)


def plot_reconciliation(cum, show):
    versions = [c for c in cum.columns if c not in ("Date", "Actual")]
    end = cum.iloc[-1]
    start_year = cum["Date"].iloc[0].year
    gap_raw = end[versions[0]] - end["Actual"]
    word = "draws" if gap_raw < 0 else "builds"

    if abs(gap_raw) < 10:
        title = "Reported flows track actual commercial stock changes closely over time"
    elif len(versions) == 1:
        title = (f"Reported flows alone imply {abs(gap_raw):,.0f} mb more {word} "
                 f"than actually happened since {start_year}")
    else:
        explained = [(end[versions[i - 1]] - end[versions[i]]) / gap_raw
                     for i in range(1, len(versions))]
        if len(versions) == 2:
            title = (f"SPR flows explain {explained[0]:.0%} of the {abs(gap_raw):,.0f} mb gap between "
                     f"reported flows and actual stocks since {start_year}; EIA's adjustment is the rest")
        else:
            main = "EIA's adjustment matters" if explained[1] > explained[0] else "SPR flows matter"
            title = (f"SPR flows explain {explained[0]:.0%} and EIA's adjustment {explained[1]:.0%} "
                     f"of the {abs(gap_raw):,.0f} mb gap since {start_year}; {main} most")

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.axhline(0, color=BASELINE, linewidth=1)
    series = [("Actual", "Actual commercial stocks", 2.6)] + [(v, v, 2.0) for v in versions]
    ends = []
    for i, (col, label, width) in enumerate(series):
        ax.plot(cum["Date"], cum[col], color=SERIES[i % len(SERIES)], linewidth=width, label=label)
        ends.append((cum["Date"].iloc[-1], cum[col].iloc[-1], f"{cum[col].iloc[-1]:+,.0f} mb"))
    ax.set_ylabel("Cumulative change (million barrels)")
    ax.legend(loc="lower left")
    ax.set_xlim(cum["Date"].iloc[0], cum["Date"].iloc[-1] + pd.Timedelta(days=200))
    label_line_ends(ax, ends)
    set_titles(ax, title, f"Cumulative change in U.S. commercial crude stocks since "
                          f"{cum['Date'].iloc[0]:%b %Y}, actual vs implied by each balance")
    save(fig, "01_balance_reconciliation.png", show)


def plot_seasonal(df, show):
    d = df.dropna(subset=["Inventory_vs_seasonal_mb"])
    if d.empty:
        print("  Skipping seasonal chart - not enough inventory history")
        return
    last = d.iloc[-1]
    gap = last["Inventory_vs_seasonal_mb"]
    title = (f"Commercial crude stocks are {abs(gap):.1f} mb {'above' if gap > 0 else 'below'} "
             f"the 5-year average for the week ending {last['Date']:%d %b %Y}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(0, color=BASELINE, linewidth=1)
    ax.plot(d["Date"], d["Inventory_vs_seasonal_mb"], color=SERIES[0],
            label="vs previous 5 years")
    ends = [(last["Date"], gap, f"{gap:+.1f} mb")]
    if d["Inventory_vs_seasonal_ex2020_mb"].notna().any():
        ax.plot(d["Date"], d["Inventory_vs_seasonal_ex2020_mb"], color=SERIES[1], linewidth=1.6,
                label="vs previous 5 years, skipping 2020")
    ax.set_ylabel("Stocks minus seasonal average (million barrels)")
    ax.legend(loc="upper right")
    ax.set_xlim(d["Date"].iloc[0], d["Date"].iloc[-1] + pd.Timedelta(days=150))
    label_line_ends(ax, ends)
    set_titles(ax, title, "U.S. commercial crude stocks (excl. SPR) minus the average for the same "
                          "week. Any 5-year window containing the 2020 glut raises the average and "
                          "makes stocks look tighter than they are.")
    save(fig, "02_seasonal_position.png", show)


def plot_drivers(drivers, date, show):
    top = drivers.loc[drivers["Contribution (kb/d)"].abs().idxmax()]
    verb = "tighten" if top["Contribution (kb/d)"] < 0 else "loosen"
    net = drivers["Contribution (kb/d)"].sum()
    title = f"{top['Phrase']} did most to {verb} the U.S. crude balance over the last 4 weeks"

    labels = list(drivers["Driver"]) + ["Net change"]
    values = list(drivers["Contribution (kb/d)"]) + [net]
    colors = [LOOSENS if v > 0 else TIGHTENS for v in values[:-1]] + [INK_2]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=colors, width=0.62)
    ax.axhline(0, color=BASELINE, linewidth=1)
    lo, hi = min(0, min(values)), max(0, max(values))
    span = (hi - lo) or 1
    for bar, v in zip(bars, values):
        offset = span * 0.02 * (1 if v >= 0 else -1)
        ax.text(bar.get_x() + bar.get_width() / 2, v + offset, f"{v:+,.0f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=9, color=INK_2)
    ax.set_ylim(lo - span * 0.12, hi + span * 0.12)
    ax.set_ylabel("Effect on balance (thousand barrels/day)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=LOOSENS), plt.Rectangle((0, 0), 1, 1, color=TIGHTENS)]
    ax.legend(handles, ["Loosens (more crude available)", "Tightens (less crude available)"],
              loc="upper left", bbox_to_anchor=(0, -0.08), ncol=2)
    set_titles(ax, title, f"Change in each flow's 4-week average vs the previous 4 weeks, "
                          f"to week ending {date:%d %b %Y}")
    save(fig, "03_balance_drivers.png", show)


def plot_timespread(ts, binned, rho, cushing, show):
    start, end = ts["Date"].min(), ts["Date"].max()
    if rho < -0.3:
        title = "When Cushing stocks are low, WTI trades in backwardation (front month above second month)"
    elif rho > 0.3:
        title = "Higher Cushing stocks have gone with a stronger WTI front-month premium in this sample"
    else:
        title = "Cushing stocks and the WTI time spread show only a weak relationship in this sample"

    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.grid(axis="x", visible=True)
    ax.axhline(0, color=BASELINE, linewidth=1)
    ax.scatter(ts["Cushing_mb"], ts["Spread"], s=14, color=SERIES[0], alpha=0.35,
               edgecolors="none", label="Weekly observation")
    ax.plot(binned["Cushing_mb"], binned["Spread"], color=SERIES[1], marker="o",
            markersize=6, label=f"Median per {CUSHING_BIN_MB} mb bin")

    q_lo, q_hi = ts["Spread"].quantile([0.005, 0.995])
    pad = (q_hi - q_lo) * 0.15
    ax.set_ylim(q_lo - pad, q_hi + pad)
    hidden = int(((ts["Spread"] < q_lo - pad) | (ts["Spread"] > q_hi + pad)).sum())

    note = f"Weekly, {start:%b %Y} to {end:%b %Y} · Spearman rank correlation {rho:.2f}"
    if hidden:
        note += f" · {hidden} extreme week(s) off-scale (April 2020)"
    if cushing is not None and not cushing.empty:
        now = cushing.iloc[-1]
        ax.axvline(now["Cushing_mb"], color=INK_2, linewidth=1)
        ax.annotate(f"Cushing now: {now['Cushing_mb']:.1f} mb\n(week ending {now['Date']:%d %b %Y})",
                    xy=(now["Cushing_mb"], ax.get_ylim()[1]), xytext=(6, -6),
                    textcoords="offset points", va="top", fontsize=8.5, color=INK_2)
    ax.set_xlabel("Cushing, OK crude stocks (million barrels)")
    ax.set_ylabel("WTI front month minus second month ($/bbl)\nabove 0 = backwardation")
    ax.legend(loc="lower right")
    set_titles(ax, title, note)
    save(fig, "04_cushing_vs_timespread.png", show)


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

def md_table(df, fmt="{:,.1f}", formats=None):
    """Markdown table. `formats` maps column names to format strings."""
    formats = formats or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for col in cols:
            v = row[col]
            numeric = isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool)
            cells.append(formats.get(col, fmt).format(v) if numeric and pd.notna(v) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def format_stats(stats):
    """Format the reconciliation table row by row (counts, ratios and percentages differ)."""
    row_formats = {"Weeks": "{:,.0f}", "Correlation": "{:.2f}", "R²": "{:.2f}"}
    shown = pd.DataFrame(index=stats.index, columns=stats.columns, dtype=object)
    for metric in stats.index:
        f = row_formats.get(metric, "{:+,.1f}" if metric in ("Mean gap (kb/d)", "Edge (pp)")
                            else "{:,.1f}")
        shown.loc[metric] = [f.format(v) for v in stats.loc[metric]]
    return shown.reset_index().rename(columns={"index": "Metric"})


def build_snapshot(df, drivers, driver_date, stats, regimes, ts, binned, rho, cushing):
    last = df.iloc[-1]
    prev4 = df.iloc[-5] if len(df) >= 5 else None
    wti_rows = df.dropna(subset=["WTI"])
    lines = [f"# U.S. crude snapshot - week ending {last['Date']:%d %b %Y}", ""]

    lines += ["## Stocks", ""]
    if not wti_rows.empty:
        w = wti_rows.iloc[-1]
        lines.append(f"- WTI: ${w['WTI']:.2f}/bbl (week ending {w['Date']:%d %b %Y})")
    lines.append(f"- Commercial crude stocks: {last['Inventories'] / 1000:.1f} mb "
                 f"({last['Inventory_change_mb']:+.1f} mb on the week)")
    if pd.notna(last["Inventory_vs_seasonal_mb"]):
        pct = (df["Inventory_vs_seasonal_mb"].rank(pct=True).iloc[-1]) * 100
        lines.append(f"- vs 5-year average: {last['Inventory_vs_seasonal_mb']:+.1f} mb "
                     f"(higher than {pct:.0f}% of weeks since {df['Date'].min():%Y})")
        if pd.notna(last["Inventory_vs_seasonal_ex2020_mb"]):
            lines.append(f"- vs 5-year average skipping 2020: "
                         f"{last['Inventory_vs_seasonal_ex2020_mb']:+.1f} mb")
        if prev4 is not None and pd.notna(prev4["Inventory_vs_seasonal_mb"]):
            moved = last["Inventory_vs_seasonal_mb"] - prev4["Inventory_vs_seasonal_mb"]
            lines.append(f"- Change in seasonal position over 4 weeks: {moved:+.1f} mb")
    if "SPR" in df and pd.notna(last.get("SPR", np.nan)):
        lines.append(f"- SPR: {last['SPR'] / 1000:.1f} mb")

    lines += ["", "## Balance, 4-week averages (kb/d)", ""]
    recent = df.tail(4)
    lines.append(f"- Reported flows (production + imports - runs - exports): "
                 f"{recent['Balance_kbd'].mean():+,.0f}")
    if "SPR_change_kbd" in df:
        lines.append(f"- SPR stock change: {recent['SPR_change_kbd'].mean():+,.0f}")
        lines.append(f"- Implied commercial stock change: "
                     f"{recent['Commercial_balance_kbd'].mean():+,.0f}")
    actual = recent["Inventory_change_mb"].mean() * 1000 / 7
    lines.append(f"- Actual commercial stock change: {actual:+,.0f}")
    unexplained_label = ("EIA's adjustment territory" if "SPR_change_kbd" in df
                         else "SPR flows + EIA's adjustment; add spr.xls to split them")
    lines.append(f"- Unexplained ({unexplained_label}): {recent['Unexplained_kbd'].mean():+,.0f}")
    if "Adjustment" in df:
        lines.append(f"- EIA published adjustment: {recent['Adjustment'].mean():+,.0f}")

    lines += ["", f"## What changed vs the previous 4 weeks (to {driver_date:%d %b %Y})", "",
              md_table(drivers[["Driver", "Change (kb/d)", "Contribution (kb/d)"]], "{:+,.0f}"),
              "", f"Net effect on the balance: {drivers['Contribution (kb/d)'].sum():+,.0f} kb/d "
              "(+ loosens, - tightens)"]

    if cushing is not None and not cushing.empty:
        c = cushing.iloc[-1]
        lines += ["", "## Cushing", "", f"- Stocks: {c['Cushing_mb']:.1f} mb "
                                          f"(week ending {c['Date']:%d %b %Y})"]
        if pd.notna(c["Cushing_vs_seasonal_mb"]):
            lines.append(f"- vs 5-year average: {c['Cushing_vs_seasonal_mb']:+.1f} mb")
        if binned is not None:
            spread = implied_spread(binned, c["Cushing_mb"])
            if spread is not None:
                lines.append(f"- Historical median M1-M2 spread at this Cushing level: "
                             f"${spread:+.2f}/bbl (based on {ts['Date'].min():%Y}-"
                             f"{ts['Date'].max():%Y} data; rank correlation {rho:.2f})")

    lines += ["", "## How well the weekly balance reconciles", "",
              "All inputs come from the same weekly report, so this measures how well the "
              "published flows add up - not a forecast.", "",
              md_table(format_stats(stats))]

    if regimes is not None:
        lines += ["", f"## Forward WTI returns by balance tercile "
                      f"({FORWARD_WEEKS} weeks, starting {RELEASE_LAG_WEEKS} week after the data week)", "",
                  "Descriptive only: overlapping windows and full-sample terciles.", "",
                  md_table(regimes.reset_index(), "{:,.1f}", formats={"Weeks": "{:,.0f}"})]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="folder with the data files (default: ./data, else the script folder)")
    parser.add_argument("--show", action="store_true", help="open the charts on screen")
    args = parser.parse_args()

    data_dir = args.data_dir or (PROJECT_DIR / "data")
    if not data_dir.exists():
        data_dir = PROJECT_DIR
    if not args.show:
        plt.switch_backend("Agg")
    apply_style()

    data = build_dataset(data_dir)
    versions = balance_versions(data)

    stats = pd.DataFrame({label: reconciliation_stats(data, col) for label, col in versions})
    cum = cumulative_reconciliation(data, versions)
    drivers, driver_date = balance_drivers(data)
    regimes = forward_return_regimes(data)
    ts, cushing = build_timespread(data_dir)
    binned, rho = spread_relationship(ts) if ts is not None and len(ts) > 50 else (None, None)

    print("Saving charts")
    plot_reconciliation(cum, args.show)
    plot_seasonal(data, args.show)
    plot_drivers(drivers, driver_date, args.show)
    if binned is not None:
        plot_timespread(ts, binned, rho, cushing, args.show)
    else:
        print("  Skipping time-spread chart - needs cushing.xls and futures.xls")

    snapshot = build_snapshot(data, drivers, driver_date, stats, regimes, ts, binned, rho, cushing)
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "market_snapshot.md").write_text(snapshot, encoding="utf-8")
    data.to_csv(OUT_DIR / "weekly_dataset.csv", index=False)
    print(f"  Charts in {FIG_DIR.name}/, summary in {OUT_DIR.name}/market_snapshot.md\n")
    print(snapshot)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
