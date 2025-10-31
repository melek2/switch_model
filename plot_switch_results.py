# %% /usr/bin/env python3
"""
Standalone Switch plotting script (no switch_model imports).
- Configure inputs_dir, outputs_dir, scenario_name, and PLOTS at the top.
- Color mapping comes from inputs_dir/graph_tech_colors.csv and graph_tech_types.csv.
- Each plotting function is pure-matplotlib and self-contained.
"""

import os
import math
import time
import textwrap
import numpy as np
import pandas as pd
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ========== USER CONFIGURATION ==========
inputs_dir = "/Users/melek/Desktop/Research/Air quality/Inputs/switch_inputs_foresight/base_10_week_2050/2050/base_short"  # "/path/to/switch_inputs"
outputs_dir = "/Users/melek/Desktop/Research/Air quality/switch_outputs/base_short_10_weeks_graph"  # "/path/to/switch_outputs"
scenario_name = "base_short"
shape_file = "/Users/melek/Documents/Data/ShapeFiles/conus26/conus26_precise_auto.shp"

# Save figures here: <outputs_dir>/<SAVE_SUBDIR>
SAVE_SUBDIR = "plots_standalone"

# Global style/resolution controls
FIG_DPI_BASE = 150
FIG_DPI_SCALE = 1.2  # 1.2 = 20% higher resolution
FIG_FONT_SIZE = 11

# Turn plots on/off here.
PLOTS = {
    "dispatch": {
        "dispatch_matrix": True,
        "dispatch_matrix_per_period": False,
        "dispatch_by_region": False,
    },
    "generation": {
        "generation_mix_stacked_bars": True,
        "generation_mix_stacked_bars_by_zone": False,
    },
    "curtailment": {
        "curtailment_by_region": True,
        "curtailment_matrix": False,
    },
}

# ========== GLOBAL INIT ==========
os.makedirs(os.path.join(outputs_dir, SAVE_SUBDIR), exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": int(FIG_DPI_BASE * FIG_DPI_SCALE),
        "savefig.dpi": int(FIG_DPI_BASE * FIG_DPI_SCALE),
        "font.size": FIG_FONT_SIZE,
        "axes.titlesize": FIG_FONT_SIZE + 1,
        "axes.labelsize": FIG_FONT_SIZE,
        "legend.fontsize": FIG_FONT_SIZE - 1,
        "xtick.labelsize": FIG_FONT_SIZE - 1,
        "ytick.labelsize": FIG_FONT_SIZE - 1,
    }
)


# ========== UTILS ==========
def _csv(path, **kwargs) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing CSV: {path}")
    return pd.read_csv(path, **kwargs)


def _timestamp_suffix():
    return time.strftime("%Y%m%d-%H%M%S")


def _safe_filename(name):
    base = name.replace(" ", "_").replace("/", "_")
    return f"{base}_{_timestamp_suffix()}.png"


def _wrap(text, width=60):
    return "\n".join(textwrap.wrap(text, width=width))


def _ensure_numeric_period(order):
    # best effort to sort periods numerically if possible
    try:
        as_int = sorted(map(int, map(str, order)))
        return as_int
    except Exception:
        try:
            as_float = sorted(map(float, map(str, order)))
            return as_float
        except Exception:
            return sorted(order)


def _tp_in_week(name) -> int | float:
    try:
        return int(str(name).rsplit("_", 1)[-1])
    except Exception:
        return np.nan


def _choose_unit(vmax_mw: float, unit: str, pretty_min=0.1, pretty_max=1000.0):
    """Uniform unit picker used by any plot that scales MW."""
    if unit != "auto":
        table = {"TW": 1e-6, "GW": 1e-3, "MW": 1.0, "kW": 1e3, "W": 1e6}
        if unit not in table:
            raise ValueError("unit must be 'auto','TW','GW','MW','kW','W'.")
        return unit, table[unit]
    for name, mult in [
        ("TW", 1e-6),
        ("GW", 1e-3),
        ("MW", 1.0),
        ("kW", 1e3),
        ("W", 1e6),
    ]:
        if pretty_min <= vmax_mw * mult < pretty_max:
            return name, mult
    return "MW", 1.0


def _outdir(domain: str, sub: str | None = None) -> str:
    """Consistent save location per domain."""
    base = os.path.join(outputs_dir, SAVE_SUBDIR, domain)
    if sub:
        base = os.path.join(base, sub)
    os.makedirs(base, exist_ok=True)
    return base


# ----- Color and type mapping from inputs_dir -----
def load_color_map_from_inputs(inputs_dir):
    """
    Expect a CSV with something like:
      gen_type,color
    or:
      category,color
    Returns dict {gen_type: color}
    """
    colors_csv = os.path.join(inputs_dir, "graph_tech_colors.csv")
    if not os.path.exists(colors_csv):
        print(
            "[warn] graph_tech_colors.csv not found; colors will use matplotlib defaults."
        )
        return {}

    df = _csv(colors_csv)
    # flexible column discovery
    lower = {c.lower(): c for c in df.columns}
    type_col = None
    for k in ("gen_type", "type", "category", "tech_type"):
        if k in lower:
            type_col = lower[k]
            break
    color_col = None
    for k in ("color", "colour", "hex"):
        if k in lower:
            color_col = lower[k]
            break
    if type_col is None or color_col is None:
        print(
            "[warn] graph_tech_colors.csv lacks recognizable columns; using defaults."
        )
        return {}

    mapping = (
        df[[type_col, color_col]].dropna().astype(str).drop_duplicates().values.tolist()
    )
    return {k: v for k, v in mapping}


def load_tech_to_type_map(inputs_dir):
    """
    Expect a CSV with something like:
      gen_tech,gen_type
    or equivalent naming (tech -> type).
    Returns dict {gen_tech: gen_type}
    """
    types_csv = os.path.join(inputs_dir, "graph_tech_types.csv")
    if not os.path.exists(types_csv):
        print(
            "[warn] graph_tech_types.csv not found; will fall back to gen_tech as category."
        )
        return {}

    df = _csv(types_csv)
    lower = {c.lower(): c for c in df.columns}
    tech_col = None
    for k in ("gen_tech", "technology", "tech"):
        if k in lower:
            tech_col = lower[k]
            break
    type_col = None
    for k in ("gen_type", "type", "category", "tech_type"):
        if k in lower:
            type_col = lower[k]
            break

    if tech_col is None or type_col is None:
        print("[warn] graph_tech_types.csv lacks recognizable columns; ignoring.")
        return {}

    mapping = (
        df[[tech_col, type_col]].dropna().astype(str).drop_duplicates().values.tolist()
    )
    return {k: v for k, v in mapping}


GEN_TYPE_COLORS = load_color_map_from_inputs(inputs_dir)
TECH_TO_TYPE = load_tech_to_type_map(inputs_dir)


def map_gen_type(df):
    """
    Add 'gen_type' to df if missing, using TECH_TO_TYPE on 'gen_tech'.
    If both missing, returns df unchanged.
    """
    if "gen_type" in df.columns:
        return df
    if "gen_tech" in df.columns and TECH_TO_TYPE:
        df = df.copy()
        df["gen_type"] = df["gen_tech"].map(TECH_TO_TYPE).fillna(df["gen_tech"])
        return df
    return df


def color_for(category, fallback_cycle):
    c = GEN_TYPE_COLORS.get(str(category))
    if c:
        return c
    # fallback to default cycle if needed
    idx = abs(hash(str(category))) % len(fallback_cycle) if fallback_cycle else None
    return fallback_cycle[idx] if idx is not None else None


def order_categories(categories):
    """
    Order categories by the order in GEN_TYPE_COLORS if available, then append others alphabetically.
    """
    preferred = list(GEN_TYPE_COLORS.keys())
    seen = set(categories)
    ordered = [c for c in preferred if c in seen]
    extras = sorted(list(seen - set(preferred)))
    return ordered + extras


# %% ***********************************
# *********** DISPATCH PLOTS ***********
# **************************************
# ========== PLOT 1a ==========
def dispatch_matrix():
    """
    Average daily dispatch (GW) heatmap over hour x day.

    Inputs:
        - <outputs_dir>/dispatch.csv  (needs 'timestamp', 'DispatchGen_MW')
        - <inputs_dir>/timepoints.csv (columns: 'timestamp','timeseries')
        - <inputs_dir>/timeseries.csv (columns: 'timeseries','ts_scale_to_period', ...)
    Output:
        - <outputs_dir>/<SAVE_SUBDIR>/dispatch_matrix_<scenario>_<ts>.png
    """
    # ----- Load & aggregate total dispatch per timestamp -----
    disp = _csv(
        os.path.join(outputs_dir, "dispatch.csv"),
        usecols=["timestamp", "DispatchGen_MW"],
    )
    # Sum across all projects for each timestamp; convert MW → GW
    disp = disp.groupby("timestamp", as_index=False)["DispatchGen_MW"].sum()
    disp["DispatchGen_MW"] = disp["DispatchGen_MW"] / 1e3  # GW

    # ----- Parse hour index within each 168-hr week -----

    disp["tp_in_week"] = disp["timestamp"].map(_tp_in_week)
    disp = disp.dropna(subset=["tp_in_week"]).copy()
    disp["tp_in_week"] = disp["tp_in_week"].astype(int)
    disp["day_index"] = disp["tp_in_week"] // 24  # 0..6
    disp["hour_of_day"] = disp["tp_in_week"] % 24  # 0..23

    # ----- Map timestamps → timeseries -----
    tps = _csv(os.path.join(inputs_dir, "timepoints.csv"))[["timestamp", "timeseries"]]
    disp = disp.merge(tps, on="timestamp", how="left", validate="many_to_one")

    # ----- Bring in series weights (how many weeks each represents) -----
    ts = _csv(os.path.join(inputs_dir, "timeseries.csv"))
    # Be robust to slight column naming differences
    ts_lower = {c.lower(): c for c in ts.columns}
    series_col = ts_lower.get("timeseries", "timeseries")
    scale_col = ts_lower.get("ts_scale_to_period", "ts_scale_to_period")

    weights = ts[[series_col, scale_col]].rename(
        columns={series_col: "timeseries", scale_col: "ts_scale_to_period"}
    )
    disp = disp.merge(weights, on="timeseries", how="left", validate="many_to_one")
    disp["ts_scale_to_period"] = disp["ts_scale_to_period"].fillna(1.0)

    # ----- Weighted average across all representative weeks -----
    # Vectorized weighted mean: sum(x*w) / sum(w) for each day×hour
    disp = disp.copy()
    disp["w"] = disp["ts_scale_to_period"]
    disp["mw_w"] = disp["DispatchGen_MW"] * disp["w"]

    prof = disp.groupby(["day_index", "hour_of_day"], as_index=False).agg(
        sum_mw_w=("mw_w", "sum"), sum_w=("w", "sum")
    )
    prof["Dispatch_GW"] = prof["sum_mw_w"] / prof["sum_w"]
    prof = prof[["day_index", "hour_of_day", "Dispatch_GW"]]

    # Pivot to day × hour matrix
    pvt = prof.pivot(
        index="day_index", columns="hour_of_day", values="Dispatch_GW"
    ).sort_index()
    # Ensure full 7×24 shape if anything is missing
    pvt = pvt.reindex(index=range(0, 7), columns=range(0, 24))

    # ----- Plot -----
    fig = plt.figure()
    # Wider than tall; respects global DPI scaling
    fig.set_size_inches(14, 6)
    ax = fig.add_subplot(111)

    im = ax.imshow(pvt.values, aspect="auto", origin="lower")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("GW")

    ax.set_title(_wrap(f"{scenario_name}: Average daily dispatch (GW)"))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Day in representative week")

    # Ticks
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels(list(pvt.columns)[::2])
    ax.set_yticks(range(0, 7))
    ax.set_yticklabels([f"Day {i+1}" for i in range(7)])

    # Save
    # out_dir = os.path.join(outputs_dir, SAVE_SUBDIR)
    # os.makedirs(out_dir, exist_ok=True)
    # fname = _safe_filename(f"dispatch_matrix_{scenario_name}")
    # fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight")
    # plt.close(fig)
    # print(f"[saved] {fname} -> {out_dir}")
    out_dir = _outdir("dispatch")
    fname = _safe_filename(f"dispatch_matrix_{scenario_name}")
    fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight")


# ========== PLOT 1b ==========
def dispatch_matrix_per_period():
    """
    One heatmap per representative week (timeseries), stacked vertically.
    Titles include the weight of each representative week.

    Inputs:
      - <outputs_dir>/dispatch.csv  (needs 'timestamp','DispatchGen_MW')
      - <inputs_dir>/timepoints.csv (timestamp,timeseries)
      - <inputs_dir>/timeseries.csv (timeseries, ts_scale_to_period, ...)
    Output:
      - <outputs_dir>/<SAVE_SUBDIR>/dispatch_matrix_1a_<scenario>_<ts>.png
    """
    # Load and total dispatch per timestamp; MW -> GW
    disp = _csv(
        os.path.join(outputs_dir, "dispatch.csv"),
        usecols=["timestamp", "DispatchGen_MW"],
    )
    disp = disp.groupby("timestamp", as_index=False)["DispatchGen_MW"].sum()
    disp["DispatchGen_MW"] = disp["DispatchGen_MW"] / 1e3  # GW

    # Map timestamps -> timeseries
    tps = _csv(os.path.join(inputs_dir, "timepoints.csv"))[["timestamp", "timeseries"]]
    disp = disp.merge(tps, on="timestamp", how="left", validate="many_to_one")

    # Bring in series weights
    ts = _csv(os.path.join(inputs_dir, "timeseries.csv"))
    ts_lower = {c.lower(): c for c in ts.columns}
    series_col = ts_lower.get("timeseries", "timeseries")
    scale_col = ts_lower.get("ts_scale_to_period", "ts_scale_to_period")
    weights = ts[[series_col, scale_col]].rename(
        columns={series_col: "timeseries", scale_col: "ts_scale_to_period"}
    )

    # Parse hour within week
    # def _tp_in_week(ts_name):
    #     try:
    #         return int(str(ts_name).rsplit("_", 1)[-1])
    #     except Exception:
    #         return np.nan

    disp["tp_in_week"] = disp["timestamp"].map(_tp_in_week)
    disp = disp.dropna(subset=["tp_in_week"]).copy()
    disp["tp_in_week"] = disp["tp_in_week"].astype(int)
    disp["day_index"] = disp["tp_in_week"] // 24  # 0..6
    disp["hour_of_day"] = disp["tp_in_week"] % 24  # 0..23

    # Compute matrix per timeseries (no weighting across series here)
    series_list = disp["timeseries"].dropna().drop_duplicates().tolist()

    # Prepare all matrices and determine global color scale
    mats = []
    series_info = []  # (name, weight, pct)
    weight_map = weights.set_index("timeseries")["ts_scale_to_period"].to_dict()
    total_w = sum(weight_map.get(s, 1.0) for s in series_list) or 1.0

    vmin, vmax = np.inf, -np.inf
    for s in series_list:
        g = disp[disp["timeseries"] == s]
        pvt = g.pivot(
            index="day_index", columns="hour_of_day", values="DispatchGen_MW"
        ).reindex(index=range(0, 7), columns=range(0, 24))
        pvt = pvt.fillna(0.0)
        mats.append(pvt.values)
        w = float(weight_map.get(s, 1.0))
        pct = 100.0 * w / total_w
        series_info.append((s, w, pct))
        # update color range
        vmin = min(vmin, np.nanmin(pvt.values))
        vmax = max(vmax, np.nanmax(pvt.values))

    # Fallback if all zero
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0
    if vmax == vmin:
        vmax = vmin + 1e-6

    # Figure: one row per timeseries
    n = len(series_list)
    # Height scales with number of subplots
    fig_h = max(2.0 * n, 4.0)
    fig = plt.figure()
    fig.set_size_inches(14, fig_h)
    axes = [fig.add_subplot(n, 1, i + 1) for i in range(n)]

    for ax, mat, (s, w, pct) in zip(axes, mats, series_info):
        im = ax.imshow(mat, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
        # Titles emphasize weight
        ax.set_title(_wrap(f"{s} | weight {int(w):.1f} weeks ({pct:.1f}% of year)"))
        ax.set_ylabel("Day")
        ax.set_yticks(range(0, 7))
        ax.set_yticklabels([f"{d+1}" for d in range(7)])
        ax.set_xticks(range(0, 24, 2))
        ax.set_xticklabels([str(h) for h in range(0, 24, 2)])
        # Hide x label for all but bottom
        if ax is not axes[-1]:
            ax.set_xlabel("")
            ax.tick_params(axis="x", which="both", labelbottom=False)
        else:
            ax.set_xlabel("Hour of day")

    # Shared colorbar on the right
    # Use the last Axes' mappable (all share vmin/vmax)
    mappable = axes[-1].images[0]
    cbar = fig.colorbar(
        mappable, ax=axes, orientation="vertical", pad=0.01, fraction=0.02
    )
    cbar.set_label("GW")

    fig.suptitle(
        _wrap(f"{scenario_name}: Average daily dispatch by representative week"),
        y=0.995,
    )

    # out_dir = os.path.join(outputs_dir, SAVE_SUBDIR)
    # os.makedirs(out_dir, exist_ok=True)
    # fname = _safe_filename(f"dispatch_matrix_1a_{scenario_name}")
    out_dir = _outdir("dispatch")
    fname = _safe_filename(f"dispatch_matrix_per_period_{scenario_name}")

    fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {fname} -> {out_dir}")


# ========== PLOT 1c ==========
def dispatch_by_region(
    zones: "list[str] | str" = "all",
    split_weeks: bool = True,  # True → one row per timeseries; False → weighted-average week
    plot_style: str = "stacked",  # "line" or "stacked"
    unit: str = "GW",  # "GW","MW","TW","auto"
    pretty_min: float = 0.1,
    pretty_max: float = 1000.0,
    legend_top_to_bottom: bool = False,  # reverse legend order
    max_rows_per_col: int = 10,  # max # of rows per column before wrapping
):
    """
    Plots hourly dispatch by region and generation type.

    - Output: <outputs_dir>/<SAVE_SUBDIR>/dispatch_by_region/zone_<ZONE>_dispatch_<timestamp>.png
    - split_weeks=True → one subplot per representative week (timeseries)
      split_weeks=False → single averaged representative week
    - plot_style: "line" (per-type lines) or "stacked" (stacked area composition)
    - max_rows_per_col: number of rows per column before wrapping to new column
    """

    # ---------- Load and preprocess ----------
    df = _csv(os.path.join(outputs_dir, "dispatch.csv"))
    expected_cols = ["timestamp", "DispatchGen_MW", "gen_load_zone"]

    tech_col = (
        "gen_type"
        if "gen_type" in df.columns
        else "gen_tech" if "gen_tech" in df.columns else None
    )
    if tech_col is None:
        raise ValueError("dispatch.csv must contain either 'gen_type' or 'gen_tech'.")

    disp = df[expected_cols + [tech_col]].copy()

    # Map gen_tech → gen_type if needed
    if tech_col == "gen_tech" and "gen_type" not in disp.columns:
        type_path = os.path.join(inputs_dir, "graph_tech_types.csv")
        if os.path.exists(type_path):
            mapping_df = pd.read_csv(type_path)
            lower = {c.lower(): c for c in mapping_df.columns}
            tech_c = (
                lower.get("gen_tech") or lower.get("technology") or lower.get("tech")
            )
            type_c = lower.get("gen_type") or lower.get("type") or lower.get("category")
            if tech_c and type_c:
                tech_to_type = dict(zip(mapping_df[tech_c], mapping_df[type_c]))
                disp["gen_type"] = (
                    disp["gen_tech"].map(tech_to_type).fillna(disp["gen_tech"])
                )
            else:
                disp["gen_type"] = disp["gen_tech"]
        else:
            disp["gen_type"] = disp["gen_tech"]
    tech_col = "gen_type"

    disp["DispatchGen_MW"] = pd.to_numeric(
        disp["DispatchGen_MW"], errors="coerce"
    ).fillna(0.0)

    # Timestamp → timeseries
    tps = _csv(os.path.join(inputs_dir, "timepoints.csv"))[["timestamp", "timeseries"]]
    disp = disp.merge(tps, on="timestamp", how="left", validate="many_to_one")

    # Timeseries weights
    ts = _csv(os.path.join(inputs_dir, "timeseries.csv"))
    ts_lower = {c.lower(): c for c in ts.columns}
    series_col = ts_lower.get("timeseries", "timeseries")
    scale_col = ts_lower.get("ts_scale_to_period", "ts_scale_to_period")
    weights = ts[[series_col, scale_col]].rename(
        columns={series_col: "timeseries", scale_col: "ts_scale_to_period"}
    )
    weight_map = weights.set_index("timeseries")["ts_scale_to_period"].to_dict()

    # Color mapping
    color_path = os.path.join(inputs_dir, "graph_tech_colors.csv")
    if os.path.exists(color_path):
        color_df = pd.read_csv(color_path)
        lower = {c.lower(): c for c in color_df.columns}
        key_col = lower.get("gen_type") or lower.get("type") or list(lower.values())[0]
        col_col = lower.get("color") or lower.get("hex") or list(lower.values())[-1]
        color_map = dict(
            zip(color_df[key_col].astype(str), color_df[col_col].astype(str))
        )
    else:
        color_map = {}

    def color_for_type(t):
        if str(t) in color_map:
            return color_map[str(t)]
        cycle = plt.rcParams.get("axes.prop_cycle").by_key().get("color", [])
        return cycle[hash(str(t)) % len(cycle)] if cycle else "gray"

    # Parse hour within week
    def _tp_in_week(ts_name):
        try:
            return int(str(ts_name).rsplit("_", 1)[-1])
        except Exception:
            return np.nan

    disp["tp_in_week"] = disp["timestamp"].map(_tp_in_week)
    disp = disp.dropna(subset=["tp_in_week"]).copy()
    disp["tp_in_week"] = disp["tp_in_week"].astype(int)
    disp["day_index"] = disp["tp_in_week"] // 24
    disp["hour_of_day"] = disp["tp_in_week"] % 24
    disp["hour_abs"] = disp["day_index"] * 24 + disp["hour_of_day"]

    # ---------- Zones ----------
    all_zones = sorted(disp["gen_load_zone"].dropna().unique().tolist())
    target_zones = (
        all_zones
        if zones in ("all", None)
        else [
            z for z in ([zones] if isinstance(zones, str) else zones) if z in all_zones
        ]
    )

    # ---------- Output ----------
    # out_dir = os.path.join(outputs_dir, SAVE_SUBDIR, "dispatch_by_region")
    # os.makedirs(out_dir, exist_ok=True)
    out_dir = _outdir("dispatch", "by_region")

    # ---------- Helper ----------
    def _set_week_ticks(ax):
        ax.set_xlim(0, 168)
        ticks = np.arange(0, 168, 24)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(d) for d in range(1, 8)])

    def pivot_hourly(df_in):
        p = (
            df_in.groupby(["hour_abs", "gen_type"], as_index=False)["Dispatch_units"]
            .sum()
            .pivot(index="hour_abs", columns="gen_type", values="Dispatch_units")
            .reindex(index=np.arange(0, 168))
            .fillna(0.0)
        )
        return p.reindex(columns=sorted(p.columns)).fillna(0.0)

    # ---------- Loop through zones ----------
    for zone in target_zones:
        d_zone = disp[disp["gen_load_zone"] == zone].copy()
        if d_zone.empty:
            print(f"[info] Zone {zone}: no dispatch data.")
            continue

        # vmax_mw = d_zone["DispatchGen_MW"].max()
        # if unit != "auto":
        #     scale = {"TW": 1e-6, "GW": 1e-3, "MW": 1.0}[unit]
        #     out_unit = unit
        # else:
        #     out_unit, scale = ("GW", 1e-3) if 0.1 <= vmax_mw * 1e-3 < 1000 else ("MW", 1.0)
        # d_zone["Dispatch_units"] = d_zone["DispatchGen_MW"] * scale
        vmax_mw = d_zone["DispatchGen_MW"].max()
        out_unit, mult = _choose_unit(vmax_mw, unit, pretty_min, pretty_max)
        d_zone["Dispatch_units"] = d_zone["DispatchGen_MW"] * mult

        # ---------------- Split weeks (multi-panel faceting) ----------------
        if split_weeks:
            series_list = d_zone["timeseries"].dropna().unique().tolist()
            total_w = sum(weight_map.get(s, 1.0) for s in series_list) or 1.0
            n_ts = len(series_list)

            ncols = math.ceil(n_ts / max_rows_per_col)
            nrows = min(max_rows_per_col, n_ts)
            fig_w = 7 * ncols
            fig_h = 2.5 * nrows
            fig, axes = plt.subplots(
                nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), squeeze=False
            )

            # Fill column-major order
            axes_flat = [axes[r, c] for c in range(ncols) for r in range(nrows)]

            used_types = set()
            for i, (ax, s) in enumerate(zip(axes_flat, series_list)):
                g = d_zone[d_zone["timeseries"] == s]
                p = pivot_hourly(g)
                if plot_style == "stacked":
                    X = p.index.values
                    Y = [p[c].values for c in p.columns]
                    ax.stackplot(X, Y, colors=[color_for_type(c) for c in p.columns])
                    used_types.update(p.columns)
                else:
                    for c in p.columns:
                        ax.plot(
                            p.index.values,
                            p[c].values,
                            lw=1.2,
                            color=color_for_type(c),
                            label=c,
                        )
                        used_types.add(c)
                w = float(weight_map.get(s, 1.0))
                pct = 100.0 * w / total_w
                ax.set_title(f"{s} | {int(w)} wk ({pct:.1f}%)", fontsize=9)
                ax.set_ylabel(out_unit, fontsize=8)
                _set_week_ticks(ax)
                if (i % nrows) != nrows - 1:
                    ax.tick_params(axis="x", labelbottom=False)
                else:
                    ax.set_xlabel("Day of representative week")

            for j in range(n_ts, len(axes_flat)):
                axes_flat[j].set_visible(False)

            plt.tight_layout()

            # Dynamic legend with only used types
            legend_types = sorted(list(used_types))
            if legend_top_to_bottom:
                legend_types = legend_types[::-1]
            handles = [
                Patch(facecolor=color_for_type(c), edgecolor="none")
                for c in legend_types
            ]
            labels = [str(c) for c in legend_types]
            fig.legend(
                handles,
                labels,
                loc="center right",
                bbox_to_anchor=(1.05, 0.5),
                frameon=False,
                title="Generation Type",
            )

            fig.suptitle(
                f"{scenario_name}: {zone} — Dispatch by representative week "
                f"({'stacked' if plot_style == 'stacked' else 'lines'})",
                y=0.995,
            )

        # ---------------- Weighted-average representative week ----------------
        else:
            g = d_zone.merge(weights, on="timeseries", how="left")
            g["w"] = g["ts_scale_to_period"].fillna(1.0)
            g["units_w"] = g["Dispatch_units"] * g["w"]
            prof = g.groupby(["hour_abs", "gen_type"], as_index=False).agg(
                sum_u_w=("units_w", "sum"), sum_w=("w", "sum")
            )
            prof["Dispatch_units"] = np.divide(
                prof["sum_u_w"],
                prof["sum_w"],
                out=np.zeros_like(prof["sum_u_w"]),
                where=prof["sum_w"] > 0,
            )
            p = (
                prof.pivot(
                    index="hour_abs", columns="gen_type", values="Dispatch_units"
                )
                .reindex(index=np.arange(0, 168))
                .fillna(0.0)
            )
            p = p.reindex(columns=sorted(p.columns)).fillna(0.0)

            fig, ax = plt.subplots(figsize=(14, 5))
            used_types = list(p.columns)
            if plot_style == "stacked":
                X = p.index.values
                Y = [p[c].values for c in p.columns]
                ax.stackplot(X, Y, colors=[color_for_type(c) for c in p.columns])
            else:
                for c in p.columns:
                    ax.plot(
                        p.index.values,
                        p[c].values,
                        lw=1.6,
                        color=color_for_type(c),
                        label=c,
                    )

            _set_week_ticks(ax)
            ax.set_xlabel("Day of representative week")
            ax.set_ylabel(out_unit)
            ax.set_title(
                _wrap(
                    f"[Dispatch] {scenario_name}: {zone} — Average daily dispatch ({out_unit})"
                )
            )

            legend_types = used_types[::-1] if legend_top_to_bottom else used_types
            handles = [
                Patch(facecolor=color_for_type(c), edgecolor="none")
                for c in legend_types
            ]
            labels = [str(c) for c in legend_types]
            fig.legend(
                handles,
                labels,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=False,
                title="Generation Type",
            )
            fig.tight_layout()

        # ---------- Save ----------
        fname = _safe_filename(f"zone_{zone}_dispatch")
        fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {fname} -> {out_dir}")


# %% *************************************
# *********** GENERATION PLOTS ***********
# ****************************************
# ========== PLOT 2a ==========
def generation_mix_stacked_bars(
    to_twh: bool = True,
    bucket_small: bool = True,
    bucket_threshold: float = 0.005,
    show_totals_in_legend: bool = True,
    annotate_bars: bool = True,
    annotate_threshold: float = 0.001,
    order_by_totals: bool = False,
    legend_top_to_bottom: bool = True,
):
    """
    Generation mix by period as stacked bars by gen_type.

    Sources (first available is used):
      1) <outputs_dir>/dispatch_annual_summary.csv
      2) <outputs_dir>/dispatch_gen_annual_summary.csv

    Behavior:
      - Maps gen_tech → gen_type via inputs_dir/graph_tech_types.csv when needed.
      - Uses colors from inputs_dir/graph_tech_colors.csv.
      - Buckets small categories per period into 'Other' (optional).
      - Drops all-zero categories post-aggregation (prevents empty 'Other').
      - Legend entries include total energy and share (optional).
      - Legend order can mirror the visible stack top→bottom.
    """
    # ---------- Load source ----------
    path1 = os.path.join(outputs_dir, "dispatch_annual_summary.csv")
    path2 = os.path.join(outputs_dir, "dispatch_gen_annual_summary.csv")
    if os.path.exists(path1):
        df = _csv(path1)
    elif os.path.exists(path2):
        df = _csv(path2)
    else:
        raise FileNotFoundError(
            "dispatch_annual_summary.csv or dispatch_gen_annual_summary.csv not found in outputs_dir."
        )

    energy_col = "Energy_GWh_typical_yr"
    if energy_col not in df.columns:
        raise ValueError(f"Expected '{energy_col}' in annual summary file.")

    # ---------- Ensure gen_type ----------
    if "gen_type" not in df.columns:
        if "gen_tech" not in df.columns:
            raise ValueError(
                "Missing 'gen_type' and 'gen_tech'; cannot map technologies to types."
            )
        df = map_gen_type(df)

    # Remove duplicated capacity columns that are irrelevant here
    dup_cols = [
        c
        for c in df.columns
        if c.startswith("GenCapacity_MW") and c != "GenCapacity_MW"
    ]
    if dup_cols:
        df = df.drop(columns=dup_cols)

    # ---------- Aggregate to period × gen_type ----------
    agg = df.groupby(["period", "gen_type"], as_index=False)[energy_col].sum()

    # Units
    unit = "TWh" if to_twh else "GWh"
    if to_twh:
        agg[energy_col] = agg[energy_col] / 1e3  # GWh → TWh

    # Period order
    periods = _ensure_numeric_period(agg["period"].unique().tolist())

    # ---------- Optional bucketing (per period) ----------
    if bucket_small and len(agg) > 0:
        rows = []
        for p in periods:
            sub = agg[agg["period"] == p].copy()
            total = sub[energy_col].sum()
            if total <= 0:
                rows.append(sub[["period", "gen_type", energy_col]])
                continue
            sub["share"] = sub[energy_col] / total
            small_mask = sub["share"] < float(bucket_threshold)
            big = sub.loc[~small_mask, ["period", "gen_type", energy_col]]
            if small_mask.any():
                other_val = sub.loc[small_mask, energy_col].sum()
                rows.append(
                    pd.concat(
                        [
                            big,
                            pd.DataFrame(
                                [
                                    {
                                        "period": p,
                                        "gen_type": "Other",
                                        energy_col: other_val,
                                    }
                                ]
                            ),
                        ],
                        ignore_index=True,
                    )
                )
            else:
                rows.append(big)
        agg = pd.concat(rows, ignore_index=True)

    # ---------- Pivot to period × gen_type ----------
    pivot = agg.pivot(index="period", columns="gen_type", values=energy_col).fillna(0.0)
    pivot = pivot.reindex(index=periods)

    # Base category order from color CSV (then alphabetical for extras)
    cats = order_categories(pivot.columns.tolist())
    pivot = pivot.reindex(columns=cats)

    # Optional: reorder by total energy descending (before colors/legend)
    if order_by_totals and pivot.shape[1] > 0:
        totals = pivot.sum(axis=0).sort_values(ascending=False)
        pivot = pivot[totals.index]

    # Drop all-zero categories (prevents empty 'Other')
    tol = 1e-12
    keep_cols = pivot.columns[(pivot.sum(axis=0).abs() > tol)]
    pivot = pivot.loc[:, keep_cols]

    # If all categories are zero, emit an empty figure with a note
    if pivot.shape[1] == 0:
        fig = plt.figure()
        fig.set_size_inches(12, 4)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No nonzero categories to display.", ha="center", va="center")
        ax.axis("off")
        # out_dir = os.path.join(outputs_dir, SAVE_SUBDIR)
        # os.makedirs(out_dir, exist_ok=True)
        # fname = _safe_filename(f"generation_mix_stacked_bars_{scenario_name}")
        # fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight")
        out_dir = _outdir("generation")
        fname = _safe_filename(f"generation_mix_stacked_bars_{scenario_name}")
        fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight")

        plt.close(fig)
        print(f"[saved] {fname} -> {out_dir}")
        return

    # ---------- Colors and legend labels (match final columns) ----------
    cycle = plt.rcParams.get("axes.prop_cycle").by_key().get("color", [])
    colors = [
        ("lightgray" if c == "Other" else color_for(c, cycle)) for c in pivot.columns
    ]

    totals_by_cat = pivot.sum(axis=0)
    grand_total = float(totals_by_cat.sum()) if float(totals_by_cat.sum()) > 0 else 1.0

    def _fmt_amount(x: float) -> str:
        if unit == "TWh":
            return f"{x:,.1f}"
        return f"{x:,.1f}" if abs(x) < 1e4 else f"{x:,.0f}"

    legend_labels = []
    for cat in pivot.columns:
        if show_totals_in_legend:
            tot = float(totals_by_cat.get(cat, 0.0))
            pct = 100.0 * tot / grand_total
            legend_labels.append(f"{cat} · {_fmt_amount(tot)} {unit} ({pct:.1f}%)")
        else:
            legend_labels.append(str(cat))

    # ---------- Plot ----------
    fig = plt.figure()
    fig.set_size_inches(12, 6)
    ax = fig.add_subplot(111)

    x = np.arange(len(pivot))
    width = 0.6
    bottom = np.zeros(len(pivot), dtype=float)
    bars_by_cat = []

    # Precompute period totals for annotation shares
    period_totals = pivot.sum(axis=1).values

    for col, col_color in zip(pivot.columns, colors):
        vals = pivot[col].values
        bars = ax.bar(
            x, vals, width=width, bottom=bottom, color=col_color, edgecolor="none"
        )
        bars_by_cat.append(bars)

        if annotate_bars:
            shares = np.divide(
                vals, period_totals, out=np.zeros_like(vals), where=period_totals > 0
            )
            for xi, v, btm, sh in zip(x, vals, bottom, shares):
                if v > 0 and sh >= annotate_threshold:
                    ax.text(
                        xi,
                        btm + v * 0.5,
                        f"{sh*100:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=FIG_FONT_SIZE - 2,
                    )

        bottom += vals

    # Labels, ticks, legend
    ax.set_title(_wrap(f"{scenario_name}: Generation mix by period"))
    ax.set_ylabel(f"Energy ({unit})")
    ax.set_xlabel("Period")
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in pivot.index])

    # Legend handles in same order as plotted; optionally reverse to match top-of-stack first
    handles = []
    labels = []
    for i, bc in enumerate(bars_by_cat):
        rects = getattr(bc, "patches", list(bc))
        if len(rects) == 0:
            continue
        handles.append(rects[0])
        labels.append(legend_labels[i])

    if legend_top_to_bottom:
        handles = handles[::-1]
        labels = labels[::-1]

    ncols_legend = 2 if len(labels) > 12 else 1
    ax.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=ncols_legend,
        title="Generation Type · Energy Share",
        frameon=False,
    )

    fig.tight_layout()
    out_dir = os.path.join(outputs_dir, SAVE_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    fname = _safe_filename(f"generation_mix_stacked_bars_{scenario_name}")
    fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {fname} -> {out_dir}")


# ========== PLOT 2b ==========
def generation_mix_stacked_bars_by_zone(
    periods: "int|list[int]|None" = None,
    to_twh: bool = True,
    bucket_small: bool = True,
    bucket_threshold: float = 0.005,
    show_totals_in_legend: bool = True,
    annotate_bars: bool = True,
    annotate_threshold: float = 0.05,
    order_zones_by_total: bool = True,
    legend_top_to_bottom: bool = True,
    xtick_rotation: int = 45,
    legend_total_scope: str = "all_periods",  # "all_periods" or "this_period"
    facet_periods: bool = False,
    max_cols: int = 2,
):
    """
    Generation mix as stacked bars by load zone (one bar per zone) within a period.
    Supports faceting multiple periods into a single figure.

    Legend totals and shares are computed over `legend_total_scope`:
      - "all_periods": system totals across every period and zone in the file
      - "this_period": totals across the periods being displayed (for faceting: all shown periods combined)
    """
    path = os.path.join(outputs_dir, "dispatch_gen_annual_summary.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "dispatch_gen_annual_summary.csv not found in outputs_dir."
        )
    df = _csv(path)

    energy_col = "Energy_GWh_typical_yr"
    required = {"gen_load_zone", "period", energy_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"dispatch_gen_annual_summary.csv is missing columns: {sorted(missing)}"
        )

    # Ensure gen_type
    if "gen_type" not in df.columns:
        if "gen_tech" not in df.columns:
            raise ValueError(
                "Missing 'gen_type' and 'gen_tech'; cannot map technologies to types."
            )
        df = map_gen_type(df)

    # ---- Unit handling ----
    unit = "TWh" if to_twh else "GWh"
    df = df.copy()
    if to_twh:
        df[energy_col] = df[energy_col] / 1e3  # GWh → TWh

    # ---- Period selection ----
    all_periods = _ensure_numeric_period(df["period"].dropna().unique().tolist())
    if periods is None:
        sel_periods = all_periods
    else:
        sel_periods = [periods] if isinstance(periods, (int, str)) else list(periods)
        sel_periods = [p for p in sel_periods if p in all_periods]
        if not sel_periods:
            raise ValueError(f"No matching periods in data. Available: {all_periods}")

    # ---- Legend totals scope ----
    if legend_total_scope not in {"all_periods", "this_period"}:
        raise ValueError("legend_total_scope must be 'all_periods' or 'this_period'.")

    # Aggregate to (period, zone, type)
    agg_all = df.groupby(["period", "gen_load_zone", "gen_type"], as_index=False)[
        energy_col
    ].sum()

    # ----- Color helper: use global mapping consistently -----
    cycle = plt.rcParams.get("axes.prop_cycle").by_key().get("color", [])

    def _col_for(cat: str) -> str:
        return "lightgray" if str(cat) == "Other" else color_for(cat, cycle)

    # Helper: compute category order + legend labels for a given scope
    def _prep_legend_and_order(scope_df):
        cats = order_categories(scope_df["gen_type"].dropna().unique().tolist())
        # keep "Other" last
        if "Other" in cats:
            cats = [c for c in cats if c != "Other"] + ["Other"]

        totals_ser = (
            scope_df.groupby("gen_type", as_index=False)[energy_col]
            .sum()
            .set_index("gen_type")[energy_col]
        )
        # drop categories with truly zero total in this scope
        cats = [c for c in cats if float(totals_ser.get(c, 0.0)) > 1e-12]

        grand_total = float(totals_ser.reindex(cats).sum()) or 1.0

        def _fmt_amount(x: float) -> str:
            if unit == "TWh":
                return f"{x:,.1f}"
            return f"{x:,.1f}" if abs(x) < 1e4 else f"{x:,.0f}"

        if show_totals_in_legend:
            legend_labels = [
                f"{c} · {_fmt_amount(float(totals_ser.get(c, 0.0)))} {unit} "
                f"({100.0*float(totals_ser.get(c, 0.0))/grand_total:.1f}%)"
                for c in cats
            ]
        else:
            legend_labels = [str(c) for c in cats]

        return cats, legend_labels

    # Legend/order scope DF
    if legend_total_scope == "all_periods":
        legend_scope_df = agg_all.copy()
    else:
        legend_scope_df = agg_all[agg_all["period"].isin(sel_periods)].copy()

    cats_global, legend_labels_global = _prep_legend_and_order(legend_scope_df)

    # ---------- Faceted multi-period figure ----------
    if facet_periods and len(sel_periods) > 1:
        n_per = len(sel_periods)
        ncols = max(1, int(max_cols))
        nrows = int(math.ceil(n_per / ncols))

        zones_in_sel = sorted(
            agg_all[agg_all["period"].isin(sel_periods)]["gen_load_zone"]
            .unique()
            .tolist()
        )
        per_subplot_height = max(3.6, 0.28 * len(zones_in_sel))
        fig_h = per_subplot_height * nrows
        fig_w = 12 * ncols

        fig = plt.figure()
        fig.set_size_inches(fig_w, fig_h)
        axes = [fig.add_subplot(nrows, ncols, i + 1) for i in range(n_per)]

        for ax, p in zip(axes, sel_periods):
            sub = agg_all[agg_all["period"] == p].copy()
            if sub.empty:
                ax.axis("off")
                continue

            # Optional bucketing per zone for this period
            if bucket_small:
                rows = []
                for z, zdf in sub.groupby("gen_load_zone", as_index=False):
                    total = zdf[energy_col].sum()
                    if total <= 0:
                        rows.append(
                            zdf[["period", "gen_load_zone", "gen_type", energy_col]]
                        )
                        continue
                    zdf = zdf.copy()
                    zdf["share"] = zdf[energy_col] / total
                    small = zdf["share"] < float(bucket_threshold)
                    big = zdf.loc[
                        ~small, ["period", "gen_load_zone", "gen_type", energy_col]
                    ]
                    if small.any():
                        other_val = zdf.loc[small, energy_col].sum()
                        rows.append(
                            pd.DataFrame(
                                [
                                    {
                                        "period": p,
                                        "gen_load_zone": z,
                                        "gen_type": "Other",
                                        energy_col: other_val,
                                    }
                                ]
                            ).append(big, ignore_index=True)
                        )
                    else:
                        rows.append(big)
                sub = pd.concat(rows, ignore_index=True)

            # Build pivot and enforce global category order (drop zero cols for cleanliness)
            pivot = (
                sub.pivot(index="gen_load_zone", columns="gen_type", values=energy_col)
                .reindex(columns=cats_global)
                .fillna(0.0)
            )
            keep_cols = pivot.columns[(pivot.sum(axis=0).abs() > 1e-12)]
            pivot = pivot.loc[:, keep_cols]

            if order_zones_by_total and pivot.shape[0] > 0:
                zone_totals = pivot.sum(axis=1).sort_values(ascending=False)
                pivot = pivot.reindex(index=zone_totals.index)

            # Plot bars
            x = np.arange(len(pivot))
            width = 0.8
            bottom = np.zeros(len(pivot), dtype=float)
            zone_totals_for_share = pivot.sum(axis=1).values

            for col in pivot.columns:
                vals = pivot[col].values
                ax.bar(
                    x,
                    vals,
                    width=width,
                    bottom=bottom,
                    color=_col_for(col),
                    edgecolor="none",
                )
                if annotate_bars:
                    shares = np.divide(
                        vals,
                        zone_totals_for_share,
                        out=np.zeros_like(vals),
                        where=zone_totals_for_share > 0,
                    )
                    for xi, v, btm, sh in zip(x, vals, bottom, shares):
                        if v > 0 and sh >= annotate_threshold:
                            ax.text(
                                xi,
                                btm + v * 0.5,
                                f"{sh*100:.0f}%",
                                ha="center",
                                va="center",
                                fontsize=FIG_FONT_SIZE - 2,
                            )
                bottom += vals

            ax.set_title(f"Period {p}")
            ax.set_ylabel(f"Energy ({unit})")
            ax.set_xticks(x)
            ax.set_xticklabels(list(pivot.index), rotation=xtick_rotation, ha="right")

        # Hide extras
        for k in range(len(sel_periods), len(axes)):
            axes[k].axis("off")

        # Figure-level legend with Patch handles; only legend order reversed if requested
        leg_cats = cats_global.copy()
        leg_labels = legend_labels_global.copy()
        if legend_top_to_bottom:
            leg_cats = leg_cats[::-1]
            leg_labels = leg_labels[::-1]
        handles = [
            matplotlib.patches.Patch(facecolor=_col_for(c), edgecolor="none")
            for c in leg_cats
        ]
        fig.legend(
            handles,
            leg_labels,
            loc="center right",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            title="Generation Type · Totals",
        )

        fig.suptitle(
            _wrap(
                f"{scenario_name}: Zonal generation mix ({unit}) — Periods {sel_periods[0]}–{sel_periods[-1]}"
            ),
            y=0.995,
        )

        # out_dir = os.path.join(outputs_dir, SAVE_SUBDIR)
        out_dir = _outdir("generation", "by_zone")
        os.makedirs(out_dir, exist_ok=True)
        fname = _safe_filename(
            f"generation_mix_by_zone_periods_{sel_periods[0]}_{sel_periods[-1]}_{scenario_name}"
        )
        # fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight")
        fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight")

        plt.close(fig)
        print(f"[saved] {fname} -> {out_dir}")
        return

    # ---------- One file per period ----------
    for p in sel_periods:
        sub = agg_all[agg_all["period"] == p].copy()
        if sub.empty:
            fig = plt.figure()
            fig.set_size_inches(12, 4)
            ax = fig.add_subplot(111)
            ax.text(
                0.5, 0.5, f"No energy by zone for period {p}.", ha="center", va="center"
            )
            ax.axis("off")
            # out_dir = os.path.join(outputs_dir, SAVE_SUBDIR); os.makedirs(out_dir, exist_ok=True)
            out_dir = _outdir("generation", "by_zone")
            fname = _safe_filename(f"generation_mix_by_zone_period{p}_{scenario_name}")
            # fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight"); plt.close(fig)
            fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight")
            plt.close(fig)
            print(f"[saved] {fname} -> {out_dir}")
            continue

        # Optional bucketing per zone for this period
        if bucket_small:
            rows = []
            for z, zdf in sub.groupby("gen_load_zone", as_index=False):
                total = zdf[energy_col].sum()
                if total <= 0:
                    rows.append(
                        zdf[["period", "gen_load_zone", "gen_type", energy_col]]
                    )
                    continue
                zdf = zdf.copy()
                zdf["share"] = zdf[energy_col] / total
                small = zdf["share"] < float(bucket_threshold)
                big = zdf.loc[
                    ~small, ["period", "gen_load_zone", "gen_type", energy_col]
                ]
                if small.any():
                    other_val = zdf.loc[small, energy_col].sum()
                    rows.append(
                        pd.concat(
                            [
                                big,
                                pd.DataFrame(
                                    [
                                        {
                                            "period": p,
                                            "gen_load_zone": z,
                                            "gen_type": "Other",
                                            energy_col: other_val,
                                        }
                                    ]
                                ),
                            ],
                            ignore_index=True,
                        )
                    )
                else:
                    rows.append(big)
            sub = pd.concat(rows, ignore_index=True)

        # Pivot: zones × gen_type for this period
        pivot = sub.pivot(
            index="gen_load_zone", columns="gen_type", values=energy_col
        ).fillna(0.0)

        # Order zones by total
        if order_zones_by_total and pivot.shape[0] > 0:
            zone_totals = pivot.sum(axis=1).sort_values(ascending=False)
            pivot = pivot.reindex(index=zone_totals.index)

        # Category order from global scope; keep Other last; drop zeros
        cats = [c for c in cats_global if c in pivot.columns]
        if "Other" in cats:
            cats = [c for c in cats if c != "Other"] + ["Other"]
        pivot = pivot.reindex(columns=cats).fillna(0.0)
        pivot = pivot.loc[:, pivot.sum(axis=0).abs() > 1e-12]

        # Legend labels aligned to current columns (order-only change for display)
        labels_map = dict(zip(cats_global, legend_labels_global))
        legend_labels = [labels_map[c] for c in pivot.columns]
        leg_cats = list(pivot.columns)
        if legend_top_to_bottom:
            legend_labels = legend_labels[::-1]
            leg_cats = leg_cats[::-1]

        # ----- Plot -----
        fig = plt.figure()
        fig.set_size_inches(max(12, 0.45 * len(pivot)), 6)
        ax = fig.add_subplot(111)

        x = np.arange(len(pivot))
        width = 0.8
        bottom = np.zeros(len(pivot), dtype=float)
        zone_totals_for_share = pivot.sum(axis=1).values

        for col in pivot.columns:
            vals = pivot[col].values
            ax.bar(
                x,
                vals,
                width=width,
                bottom=bottom,
                color=_col_for(col),
                edgecolor="none",
            )
            if annotate_bars:
                shares = np.divide(
                    vals,
                    zone_totals_for_share,
                    out=np.zeros_like(vals),
                    where=zone_totals_for_share > 0,
                )
                for xi, v, btm, sh in zip(x, vals, bottom, shares):
                    if v > 0 and sh >= annotate_threshold:
                        ax.text(
                            xi,
                            btm + v * 0.5,
                            f"{sh*100:.0f}%",
                            ha="center",
                            va="center",
                            fontsize=FIG_FONT_SIZE - 2,
                        )
            bottom += vals

        ax.set_title(_wrap(f"{scenario_name}: Generation mix by zone (period {p})"))
        ax.set_ylabel(f"Energy ({unit})")
        ax.set_xlabel("Load zone")
        ax.set_xticks(x)
        ax.set_xticklabels(list(pivot.index), rotation=xtick_rotation, ha="right")

        # Legend (Patch handles to match bar fills exactly)
        handles = [
            matplotlib.patches.Patch(facecolor=_col_for(c), edgecolor="none")
            for c in leg_cats
        ]
        ncols_legend = 2 if len(legend_labels) > 12 else 1
        ax.legend(
            handles,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            ncol=ncols_legend,
            title="Generation Type · Totals",
            frameon=False,
        )

        fig.tight_layout()
        out_dir = os.path.join(outputs_dir, SAVE_SUBDIR)
        os.makedirs(out_dir, exist_ok=True)
        fname = _safe_filename(f"generation_mix_by_zone_period{p}_{scenario_name}")
        fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {fname} -> {out_dir}")


# %%
# %% **************************************
# *********** CURTAILMENT PLOTS ***********
# *****************************************
# ========== PLOT 3 ==========
def curtailment_matrix(
    aggregate: str = "week",  # "week" or "split_week"
    split_by_source: bool = False,
    unit: str = "auto",  # "auto","TW","GW","MW","kW","W"
    pretty_min: float = 0.1,  # target lower bound for vmax after scaling
    pretty_max: float = 1000.0,
    max_rows_per_col: int = 10,  # NEW: wrap after this many rows per column (split_week only)
):
    """
    Curtailment heatmaps for Solar + Wind.

    aggregate:
      - "week": average daily curtailment over hour × day, weighted by ts_scale_to_period
                (single panel or split into Solar | Wind rows if split_by_source=True)
      - "split_week": one heatmap per representative week (timeseries). If split_by_source=True,
                there are 2 columns per week: Solar | Wind. Subplots wrap column-wise after
                'max_rows_per_col' rows.

    Inputs:
      - <outputs_dir>/dispatch.csv    [timestamp, gen_energy_source, Curtailment_MW]
      - <inputs_dir>/timepoints.csv   [timestamp, timeseries]
      - <inputs_dir>/timeseries.csv   [timeseries, ts_scale_to_period]
    """
    # ---------- Load & filter ----------
    need_cols = ["timestamp", "gen_energy_source", "Curtailment_MW"]
    disp = _csv(os.path.join(outputs_dir, "dispatch.csv"), usecols=need_cols)
    disp = disp[disp["gen_energy_source"].isin(["Solar", "Wind"])].copy()

    if disp.empty:
        fig = plt.figure()
        fig.set_size_inches(10, 4)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No Solar/Wind curtailment found.", ha="center", va="center")
        ax.axis("off")
        # out_dir = os.path.join(outputs_dir, SAVE_SUBDIR); os.makedirs(out_dir, exist_ok=True)
        out_dir = _outdir("curtailment")

        # fname = _safe_filename(f"curtailment_matrix_{aggregate}_{scenario_name}")
        # fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight"); plt.close(fig)
        fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight")
        plt.close(fig)

        print(f"[saved] {fname} -> {out_dir}")
        return

    # Keep native MW; convert at the end
    disp["Curtailment_MW"] = disp["Curtailment_MW"].astype(float)

    # ---------- Map timestamp → timeseries ----------
    tps = _csv(os.path.join(inputs_dir, "timepoints.csv"))[["timestamp", "timeseries"]]
    disp = disp.merge(tps, on="timestamp", how="left", validate="many_to_one")

    # ---------- Timeseries weights ----------
    ts = _csv(os.path.join(inputs_dir, "timeseries.csv"))
    lower = {c.lower(): c for c in ts.columns}
    series_col = lower.get("timeseries", "timeseries")
    scale_col = lower.get("ts_scale_to_period", "ts_scale_to_period")
    weights_df = ts[[series_col, scale_col]].rename(
        columns={series_col: "timeseries", scale_col: "ts_scale_to_period"}
    )
    weight_map = weights_df.set_index("timeseries")["ts_scale_to_period"].to_dict()

    # ---------- Parse hour within week ----------
    def _tp_in_week(ts_str):
        try:
            return int(str(ts_str).rsplit("_", 1)[-1])
        except Exception:
            return np.nan

    disp["tp_in_week"] = disp["timestamp"].map(_tp_in_week)
    disp = disp.dropna(subset=["tp_in_week"]).copy()
    disp["tp_in_week"] = disp["tp_in_week"].astype(int)
    disp["day_index"] = disp["tp_in_week"] // 24
    disp["hour_of_day"] = disp["tp_in_week"] % 24

    # ---------- Helpers for unit scaling ----------
    # unit_table = [("TW", 1e-6), ("GW", 1e-3), ("MW", 1.0), ("kW", 1e3), ("W", 1e6)]
    # def choose_unit(vmax_mw: float) -> tuple[str, float]:
    #     if unit != "auto":
    #         explicit = {"TW": 1e-6, "GW": 1e-3, "MW": 1.0, "kW": 1e3, "W": 1e6}
    #         if unit not in explicit:
    #             raise ValueError("unit must be 'auto', 'TW', 'GW', 'MW', 'kW', or 'W'.")
    #         return unit, explicit[unit]
    #     for name, mult in unit_table:
    #         if pretty_min <= vmax_mw * mult < pretty_max:
    #             return name, mult
    #     return "MW", 1.0

    # =====================================================================
    # MODE A: aggregate == "week"  (weighted across weeks)
    # =====================================================================
    if aggregate == "week":
        # attach weights at timeseries level
        disp = disp.merge(
            weights_df, on="timeseries", how="left", validate="many_to_one"
        )
        disp["ts_scale_to_period"] = disp["ts_scale_to_period"].fillna(1.0)

        panels = ["Solar", "Wind"] if split_by_source else ["Solar+Wind"]
        mats_mw = []
        vmin_mw, vmax_mw = np.inf, -np.inf

        for src in panels:
            d = disp if src == "Solar+Wind" else disp[disp["gen_energy_source"] == src]
            # sum across projects per timestamp
            d = d.groupby(
                ["timeseries", "timestamp", "day_index", "hour_of_day"], as_index=False
            )["Curtailment_MW"].sum()
            # weighted average across representative weeks
            d = d.merge(weights_df, on="timeseries", how="left")
            d["w"] = d["ts_scale_to_period"].fillna(1.0)
            d["mw_w"] = d["Curtailment_MW"] * d["w"]

            prof = d.groupby(["day_index", "hour_of_day"], as_index=False).agg(
                sum_mw_w=("mw_w", "sum"), sum_w=("w", "sum")
            )
            prof["Curtailment_MW"] = np.divide(
                prof["sum_mw_w"],
                prof["sum_w"],
                out=np.zeros_like(prof["sum_mw_w"]),
                where=prof["sum_w"] > 0,
            )

            pvt = (
                prof.pivot(
                    index="day_index", columns="hour_of_day", values="Curtailment_MW"
                ).reindex(index=range(0, 7), columns=range(0, 24))
            ).fillna(0.0)
            mat = pvt.values
            mats_mw.append((src, mat))
            vmin_mw = min(vmin_mw, float(np.nanmin(mat)))
            vmax_mw = max(vmax_mw, float(np.nanmax(mat)))

        if not np.isfinite(vmin_mw) or not np.isfinite(vmax_mw):
            vmin_mw, vmax_mw = 0.0, 1.0
        if vmax_mw == vmin_mw:
            vmax_mw = vmin_mw + 1e-9

        # out_unit, mult = choose_unit(vmax_mw)
        out_unit, mult = _choose_unit(vmax_mw, unit, pretty_min, pretty_max)
        mats = [(src, mat * mult) for (src, mat) in mats_mw]
        vmin, vmax = vmin_mw * mult, vmax_mw * mult

        # plot (no wrapping needed here)
        nrows = len(panels)
        fig = plt.figure()
        fig.set_size_inches(14, 4.5 * nrows)
        axes = [fig.add_subplot(nrows, 1, i + 1) for i in range(nrows)]

        for ax, (label, mat) in zip(axes, mats):
            ax.set_title(
                _wrap(
                    f"[Curtailment] {scenario_name}: Average daily curtailment ({out_unit})"
                )
            )
            im = ax.imshow(mat, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
            ttl = (
                f"{scenario_name}: Average daily curtailment ({out_unit})"
                if label == "Solar+Wind"
                else f"{scenario_name}: {label} curtailment ({out_unit})"
            )
            ax.set_title(_wrap(ttl))
            ax.set_ylabel("Day")
            ax.set_yticks(range(0, 7))
            ax.set_yticklabels([f"{d+1}" for d in range(7)])
            ax.set_xticks(range(0, 24, 2))
            ax.set_xticklabels([str(h) for h in range(0, 24, 2)])
            if ax is not axes[-1]:
                ax.tick_params(axis="x", which="both", labelbottom=False)
            else:
                ax.set_xlabel("Hour of day")

        # shared colorbar
        mappable = axes[-1].images[0]
        cbar = fig.colorbar(
            mappable, ax=axes, orientation="vertical", pad=0.01, fraction=0.02
        )
        cbar.set_label(out_unit)

        # save
        # out_dir = os.path.join(outputs_dir, SAVE_SUBDIR); os.makedirs(out_dir, exist_ok=True)
        out_dir = _outdir("curtailment")

        suffix = "_split" if split_by_source else ""
        fname = _safe_filename(f"curtailment_matrix_week_{scenario_name}{suffix}")
        fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {fname} -> {out_dir}")
        return

    # =====================================================================
    # MODE B: aggregate == "split_week"  (per-week panels with wrapping)
    # =====================================================================
    if aggregate == "split_week":
        series_list = disp["timeseries"].dropna().drop_duplicates().tolist()
        total_w = sum(weight_map.get(s, 1.0) for s in series_list) or 1.0
        panel_specs = ["Solar", "Wind"] if split_by_source else ["Solar+Wind"]

        mats_mw = []  # (timeseries, src, mat)
        vmin_mw, vmax_mw = np.inf, -np.inf

        for s in series_list:
            g_ts = disp[disp["timeseries"] == s]
            for src in panel_specs:
                d = (
                    g_ts
                    if src == "Solar+Wind"
                    else g_ts[g_ts["gen_energy_source"] == src]
                )
                d = d.groupby(
                    ["timestamp", "day_index", "hour_of_day"], as_index=False
                )["Curtailment_MW"].sum()
                pvt = (
                    d.pivot(
                        index="day_index",
                        columns="hour_of_day",
                        values="Curtailment_MW",
                    ).reindex(index=range(0, 7), columns=range(0, 24))
                ).fillna(0.0)
                mat = pvt.values
                mats_mw.append((s, src, mat))
                vmin_mw = min(vmin_mw, float(np.nanmin(mat)))
                vmax_mw = max(vmax_mw, float(np.nanmax(mat)))

        if not np.isfinite(vmin_mw) or not np.isfinite(vmax_mw):
            vmin_mw, vmax_mw = 0.0, 1.0
        if vmax_mw == vmin_mw:
            vmax_mw = vmin_mw + 1e-9

        # out_unit, mult = choose_unit(vmax_mw)
        out_unit, mult = _choose_unit(vmax_mw, unit, pretty_min, pretty_max)

        mats = [(s, src, mat * mult) for (s, src, mat) in mats_mw]
        vmin, vmax = vmin_mw * mult, vmax_mw * mult

        n_ts = len(series_list)

        if not split_by_source:
            # One panel per timeseries; wrap after max_rows_per_col (column-major)
            ncols = math.ceil(n_ts / max_rows_per_col)
            nrows = min(max_rows_per_col, n_ts)
            fig_w = 7 * ncols
            fig_h = 2.3 * nrows
            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(fig_w, fig_h),
                squeeze=False,
                sharex=True,
                sharey=True,
            )

            last_im = None
            for i, s in enumerate(series_list):
                col_group = i // max_rows_per_col
                row = i % max_rows_per_col
                ax = axes[row, col_group]
                mat = next(
                    m
                    for (ts_name, src_name, m) in mats
                    if ts_name == s and src_name == "Solar+Wind"
                )
                im = ax.imshow(mat, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
                last_im = im
                w = float(weight_map.get(s, 1.0))
                pct = 100.0 * w / total_w
                ax.set_ylabel(
                    f"{s}\n{int(w)} wk ({pct:.1f}%)",
                    rotation=0,
                    labelpad=45,
                    ha="right",
                    va="center",
                    fontsize=max(8, FIG_FONT_SIZE - 1),
                )
                ax.set_yticks(range(0, 7))
                ax.set_yticklabels([f"{d+1}" for d in range(7)])
                ax.set_xticks(range(0, 24, 4))
                ax.set_xticklabels([str(h) for h in range(0, 24, 4)])
                if (row != nrows - 1) and (i != n_ts - 1):
                    ax.tick_params(axis="x", which="both", labelbottom=False)
                else:
                    ax.set_xlabel("Hour of day")

            # Hide unused axes
            total_axes = nrows * ncols
            for j in range(n_ts, total_axes):
                r = j % nrows
                c = j // nrows
                axes[r, c].set_visible(False)

            # Shared colorbar
            cbar = fig.colorbar(
                last_im,
                ax=axes.ravel().tolist(),
                orientation="vertical",
                pad=0.01,
                fraction=0.02,
            )
            cbar.set_label(out_unit)

            fig.suptitle(
                _wrap(
                    f"{scenario_name}: Average daily curtailment by representative week"
                ),
                y=0.995,
            )
        else:
            # Two panels per timeseries (Solar|Wind) side-by-side; wrap after max_rows_per_col
            ncols_groups = math.ceil(
                n_ts / max_rows_per_col
            )  # number of TS column groups
            nrows = min(max_rows_per_col, n_ts)
            ncols = 2 * ncols_groups  # 2 columns per group
            fig_w = 7.8 * ncols
            fig_h = 2.3 * nrows
            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(fig_w, fig_h),
                squeeze=False,
                sharex=True,
                sharey=True,
            )

            last_im = None
            for i, s in enumerate(series_list):
                group = i // max_rows_per_col
                row = i % max_rows_per_col
                col_solar = 2 * group
                col_wind = col_solar + 1

                # Solar
                mat_s = next(
                    m
                    for (ts_name, src_name, m) in mats
                    if ts_name == s and src_name == "Solar"
                )
                ax_s = axes[row, col_solar]
                im = ax_s.imshow(
                    mat_s, aspect="auto", origin="lower", vmin=vmin, vmax=vmax
                )
                last_im = im

                # Wind
                mat_w = next(
                    m
                    for (ts_name, src_name, m) in mats
                    if ts_name == s and src_name == "Wind"
                )
                ax_w = axes[row, col_wind]
                ax_w.imshow(mat_w, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)

                w = float(weight_map.get(s, 1.0))
                pct = 100.0 * w / total_w
                ax_s.set_ylabel(
                    f"{s}\n{int(w)} wk ({pct:.1f}%)",
                    rotation=0,
                    labelpad=45,
                    ha="right",
                    va="center",
                    fontsize=max(8, FIG_FONT_SIZE - 1),
                )

                if row == 0:  # titles on the top row of each group
                    ax_s.set_title("Solar", fontsize=FIG_FONT_SIZE + 1)
                    ax_w.set_title("Wind", fontsize=FIG_FONT_SIZE + 1)

                for ax in (ax_s, ax_w):
                    ax.set_yticks(range(0, 7))
                    ax.set_yticklabels([f"{d+1}" for d in range(7)])
                    ax.set_xticks(range(0, 24, 4))
                    ax.set_xticklabels([str(h) for h in range(0, 24, 4)])

                # X labels only on bottom-most used row in each group
                last_row_in_group = (
                    row == min(nrows - 1, (n_ts - 1) % max_rows_per_col)
                    if (group == ncols_groups - 1)
                    else row == nrows - 1
                )
                if not last_row_in_group:
                    ax_s.tick_params(axis="x", which="both", labelbottom=False)
                    ax_w.tick_params(axis="x", which="both", labelbottom=False)
                else:
                    ax_s.set_xlabel("Hour of day")
                    ax_w.set_xlabel("Hour of day")

            # --- Correctly hide only the unused cells in the FINAL group ---
            remainder = n_ts % max_rows_per_col
            if remainder == 0:
                remainder = nrows  # final group full

            last_group = ncols_groups - 1
            # hide all rows BELOW the remainder in the final group's Solar & Wind columns
            for r in range(remainder, nrows):
                axes[r, 2 * last_group].set_visible(False)  # Solar col of last group
                axes[r, 2 * last_group + 1].set_visible(False)  # Wind col of last group

            # (No other groups need hiding; they are fully filled.)

            # Shared colorbar
            cbar = fig.colorbar(
                last_im,
                ax=axes.ravel().tolist(),
                orientation="vertical",
                pad=0.01,
                fraction=0.02,
            )
            cbar.set_label(out_unit)

            fig.suptitle(
                _wrap(
                    f"{scenario_name}: Average daily curtailment by representative week (Solar | Wind)"
                ),
                y=0.995,
            )

        # save
        out_dir = os.path.join(outputs_dir, SAVE_SUBDIR)
        os.makedirs(out_dir, exist_ok=True)
        suffix = "_split" if split_by_source else ""
        fname = _safe_filename(f"curtailment_matrix_split_week_{scenario_name}{suffix}")
        fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {fname} -> {out_dir}")
        return

    # ---------- Fallback if bad option ----------
    else:
        raise ValueError("aggregate must be 'week' or 'split_week'.")


# ========== PLOT 3a ==========
def curtailment_by_region(
    zones: "list[str] | str" = "all",
    aggregate: str = "week",  # "week" or "split_week"
    split_by_source: bool = False,  # if True: Solar | Wind separate columns
    unit: str = "auto",  # "auto","TW","GW","MW","kW","W"
    pretty_min: float = 0.1,  # target lower bound for vmax after scaling
    pretty_max: float = 1000.0,
    max_rows_per_col: int = 10,  # wrap after this many rows per column
):
    """
    Produce curtailment heatmaps by region (gen_load_zone).

    • aggregate='week': one weighted representative-week heatmap per zone
                        (two columns if split_by_source=True: Solar | Wind).
    • aggregate='split_week': one row per timeseries; wraps column-wise after
                              max_rows_per_col rows. If split_by_source=True,
                              each timeseries uses a Solar|Wind column pair.
    """

    # ---------- Load & filter base data ----------
    need_cols = ["timestamp", "gen_energy_source", "Curtailment_MW", "gen_load_zone"]
    disp = _csv(os.path.join(outputs_dir, "dispatch.csv"), usecols=need_cols)
    disp = disp[disp["gen_energy_source"].isin(["Solar", "Wind"])].copy()
    if disp.empty:
        print(
            "[info] No Solar/Wind curtailment rows found in dispatch.csv; nothing to plot."
        )
        return

    disp["Curtailment_MW"] = pd.to_numeric(
        disp["Curtailment_MW"], errors="coerce"
    ).fillna(0.0)

    # Timestamp → timeseries
    tps = _csv(os.path.join(inputs_dir, "timepoints.csv"))[["timestamp", "timeseries"]]
    disp = disp.merge(tps, on="timestamp", how="left", validate="many_to_one")

    # Timeseries weights
    ts_df = _csv(os.path.join(inputs_dir, "timeseries.csv"))
    ts_lower = {c.lower(): c for c in ts_df.columns}
    series_col = ts_lower.get("timeseries", "timeseries")
    scale_col = ts_lower.get("ts_scale_to_period", "ts_scale_to_period")
    weights_df = ts_df[[series_col, scale_col]].rename(
        columns={series_col: "timeseries", scale_col: "ts_scale_to_period"}
    )
    weight_map = weights_df.set_index("timeseries")["ts_scale_to_period"].to_dict()

    # Parse hour within week → day/hour
    def _tp_in_week(ts_str):
        try:
            return int(str(ts_str).rsplit("_", 1)[-1])
        except Exception:
            return np.nan

    disp["tp_in_week"] = disp["timestamp"].map(_tp_in_week)
    disp = disp.dropna(subset=["tp_in_week"]).copy()
    disp["tp_in_week"] = disp["tp_in_week"].astype(int)
    disp["day_index"] = disp["tp_in_week"] // 24
    disp["hour_of_day"] = disp["tp_in_week"] % 24

    # Determine zones to render
    all_zones = sorted(disp["gen_load_zone"].dropna().unique().tolist())
    if zones == "all" or zones is None:
        target_zones = all_zones
    else:
        zlist = zones if isinstance(zones, list) else [zones]
        target_zones = [z for z in zlist if z in all_zones]
        missing = [z for z in zlist if z not in all_zones]
        if missing:
            print(f"[warn] Skipping zones not found in data: {missing}")

    # ---------- Helpers ----------
    def choose_unit(vmax_mw_val: float) -> tuple[str, float]:
        if unit != "auto":
            explicit = {"TW": 1e-6, "GW": 1e-3, "MW": 1.0, "kW": 1e3, "W": 1e6}
            if unit not in explicit:
                raise ValueError("unit must be 'auto', 'TW', 'GW', 'MW', 'kW', or 'W'.")
            return unit, explicit[unit]
        unit_table = [("TW", 1e-6), ("GW", 1e-3), ("MW", 1.0), ("kW", 1e3), ("W", 1e6)]
        for name, mult in unit_table:
            if pretty_min <= vmax_mw_val * mult < pretty_max:
                return name, mult
        return "MW", 1.0

    def zone_mats_aggregate_week(zone: str, split: bool):
        d0 = disp[disp["gen_load_zone"] == zone].copy()
        if d0.empty:
            return [], 0.0, 0.0

        labels = ["Solar", "Wind"] if split else ["Solar+Wind"]
        mats_mw, vmin_mw, vmax_mw = [], np.inf, -np.inf
        for lab in labels:
            d = d0 if lab == "Solar+Wind" else d0[d0["gen_energy_source"] == lab]
            d = d.groupby(
                ["timeseries", "timestamp", "day_index", "hour_of_day"], as_index=False
            )["Curtailment_MW"].sum()
            d = d.merge(weights_df, on="timeseries", how="left")
            d["w"] = d["ts_scale_to_period"].fillna(1.0)
            d["mw_w"] = d["Curtailment_MW"] * d["w"]
            prof = d.groupby(["day_index", "hour_of_day"], as_index=False).agg(
                sum_mw_w=("mw_w", "sum"), sum_w=("w", "sum")
            )
            prof["Curtailment_MW"] = np.divide(
                prof["sum_mw_w"],
                prof["sum_w"],
                out=np.zeros_like(prof["sum_mw_w"]),
                where=prof["sum_w"] > 0,
            )
            pvt = (
                prof.pivot(
                    index="day_index", columns="hour_of_day", values="Curtailment_MW"
                ).reindex(index=range(0, 7), columns=range(0, 24))
            ).fillna(0.0)
            mat = pvt.values
            mats_mw.append((lab, mat))
            vmin_mw = min(vmin_mw, float(np.nanmin(mat)))
            vmax_mw = max(vmax_mw, float(np.nanmax(mat)))
        if not np.isfinite(vmin_mw) or not np.isfinite(vmax_mw):
            vmin_mw, vmax_mw = 0.0, 1.0
        if vmax_mw == vmin_mw:
            vmax_mw = vmin_mw + 1e-9
        return mats_mw, vmin_mw, vmax_mw

    def zone_mats_split_week(zone: str, split: bool):
        d0 = disp[disp["gen_load_zone"] == zone].copy()
        if d0.empty:
            return [], [], 0.0, 0.0

        series_list = d0["timeseries"].dropna().drop_duplicates().tolist()
        labels = ["Solar", "Wind"] if split else ["Solar+Wind"]
        mats_mw, vmin_mw, vmax_mw = [], np.inf, -np.inf
        for s in series_list:
            g_ts = d0[d0["timeseries"] == s]
            for lab in labels:
                d = (
                    g_ts
                    if lab == "Solar+Wind"
                    else g_ts[g_ts["gen_energy_source"] == lab]
                )
                d = d.groupby(
                    ["timestamp", "day_index", "hour_of_day"], as_index=False
                )["Curtailment_MW"].sum()
                pvt = (
                    d.pivot(
                        index="day_index",
                        columns="hour_of_day",
                        values="Curtailment_MW",
                    ).reindex(index=range(0, 7), columns=range(0, 24))
                ).fillna(0.0)
                mat = pvt.values
                mats_mw.append((s, lab, mat))
                vmin_mw = min(vmin_mw, float(np.nanmin(mat)))
                vmax_mw = max(vmax_mw, float(np.nanmax(mat)))
        if not np.isfinite(vmin_mw) or not np.isfinite(vmax_mw):
            vmin_mw, vmax_mw = 0.0, 1.0
        if vmax_mw == vmin_mw:
            vmax_mw = vmin_mw + 1e-9
        return series_list, mats_mw, vmin_mw, vmax_mw

    # ---------- Output dir ----------
    out_dir = os.path.join(outputs_dir, SAVE_SUBDIR, "curtailment_by_period")
    os.makedirs(out_dir, exist_ok=True)

    # ---------- Render ----------
    for zone in target_zones:
        if aggregate == "week":
            mats_mw, vmin_mw, vmax_mw = zone_mats_aggregate_week(zone, split_by_source)
            if not mats_mw:
                print(f"[info] Zone {zone}: no curtailment data to plot.")
                continue
            out_unit, mult = choose_unit(vmax_mw)
            mats = [(lab, mat * mult) for (lab, mat) in mats_mw]
            vmin, vmax = vmin_mw * mult, vmax_mw * mult

            ncols = len(mats)
            fig_w = 14 if ncols == 1 else 16
            fig_h = 4.8
            fig, axes = plt.subplots(
                1,
                ncols,
                figsize=(fig_w, fig_h),
                squeeze=False,
                sharex=True,
                sharey=True,
            )

            last_im = None
            for j, (lab, mat) in enumerate(mats):
                ax = axes[0, j]
                im = ax.imshow(mat, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
                last_im = im
                title_lab = lab if lab != "Solar+Wind" else "Solar+Wind"
                ax.set_title(_wrap(f"{scenario_name}: {zone} — {title_lab}"))
                ax.set_ylabel("Day")
                ax.set_yticks(range(0, 7))
                ax.set_yticklabels([f"{d+1}" for d in range(7)])
                ax.set_xticks(range(0, 24, 2))
                ax.set_xticklabels([str(h) for h in range(0, 24, 2)])
                ax.set_xlabel("Hour of day")

            cbar = fig.colorbar(
                last_im,
                ax=axes.ravel().tolist(),
                orientation="vertical",
                pad=0.01,
                fraction=0.02,
            )
            cbar.set_label(out_unit)

            fname = _safe_filename(f"zone_{zone}_curtailment")
            fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight")
            plt.close(fig)
            print(f"[saved] {fname} -> {out_dir}")

        elif aggregate == "split_week":
            series_list, mats_mw, vmin_mw, vmax_mw = zone_mats_split_week(
                zone, split_by_source
            )
            if not mats_mw:
                print(f"[info] Zone {zone}: no curtailment data to plot.")
                continue
            out_unit, mult = choose_unit(vmax_mw)
            mats = [(s, lab, mat * mult) for (s, lab, mat) in mats_mw]
            vmin, vmax = vmin_mw * mult, vmax_mw * mult

            total_w = sum(weight_map.get(s, 1.0) for s in series_list) or 1.0
            n_ts = len(series_list)

            if not split_by_source:
                ncols = math.ceil(n_ts / max_rows_per_col)
                nrows = min(max_rows_per_col, n_ts)
                fig_w = 7 * ncols
                fig_h = 2.3 * nrows
                fig, axes = plt.subplots(
                    nrows=nrows,
                    ncols=ncols,
                    figsize=(fig_w, fig_h),
                    squeeze=False,
                    sharex=True,
                    sharey=True,
                )

                last_im = None
                for i, s in enumerate(series_list):
                    group = i // max_rows_per_col
                    row = i % max_rows_per_col
                    ax = axes[row, group]
                    mat = next(
                        m
                        for (ts_name, lab, m) in mats
                        if ts_name == s and lab == "Solar+Wind"
                    )
                    im = ax.imshow(
                        mat, aspect="auto", origin="lower", vmin=vmin, vmax=vmax
                    )
                    last_im = im
                    w = float(weight_map.get(s, 1.0))
                    pct = 100.0 * w / total_w
                    ax.set_ylabel(
                        f"{s}\n{int(w)} wk ({pct:.1f}%)",
                        rotation=0,
                        labelpad=45,
                        ha="right",
                        va="center",
                        fontsize=FIG_FONT_SIZE - 1,
                    )
                    ax.set_yticks(range(0, 7))
                    ax.set_yticklabels([f"{d+1}" for d in range(7)])
                    ax.set_xticks(range(0, 24, 4))
                    ax.set_xticklabels([str(h) for h in range(0, 24, 4)])

                    # bottom labels only for last used row in each group
                    last_used_row = (
                        min(max_rows_per_col, n_ts - group * max_rows_per_col) - 1
                    )
                    if row != last_used_row:
                        ax.tick_params(axis="x", which="both", labelbottom=False)
                    else:
                        ax.set_xlabel("Hour of day")

                # Hide unused cells in final column
                total_axes = nrows * ncols
                for j in range(n_ts, total_axes):
                    r = j % nrows
                    c = j // nrows
                    axes[r, c].set_visible(False)

                cbar = fig.colorbar(
                    last_im,
                    ax=axes.ravel().tolist(),
                    orientation="vertical",
                    pad=0.01,
                    fraction=0.02,
                )
                cbar.set_label(out_unit)

                fig.suptitle(
                    _wrap(
                        f"{scenario_name}: {zone} — Curtailment by representative week"
                    ),
                    y=0.995,
                )

            else:
                # ---- Solar|Wind paired columns with correct hiding for last partial group ----
                ncol_groups = math.ceil(
                    n_ts / max_rows_per_col
                )  # number of TS column groups
                nrows = min(max_rows_per_col, n_ts)
                ncols = 2 * ncol_groups  # Solar & Wind per group
                fig_w = 7.8 * ncols
                fig_h = 2.3 * nrows
                fig, axes = plt.subplots(
                    nrows=nrows,
                    ncols=ncols,
                    figsize=(fig_w, fig_h),
                    squeeze=False,
                    sharex=True,
                    sharey=True,
                )

                last_im = None
                for i, s in enumerate(series_list):
                    group = i // max_rows_per_col
                    row = i % max_rows_per_col
                    col_solar = 2 * group
                    col_wind = col_solar + 1

                    # Solar
                    mat_s = next(
                        m
                        for (ts_name, lab, m) in mats
                        if ts_name == s and lab == "Solar"
                    )
                    ax_s = axes[row, col_solar]
                    im = ax_s.imshow(
                        mat_s, aspect="auto", origin="lower", vmin=vmin, vmax=vmax
                    )
                    last_im = im

                    # Wind
                    mat_w = next(
                        m
                        for (ts_name, lab, m) in mats
                        if ts_name == s and lab == "Wind"
                    )
                    ax_w = axes[row, col_wind]
                    ax_w.imshow(
                        mat_w, aspect="auto", origin="lower", vmin=vmin, vmax=vmax
                    )

                    w = float(weight_map.get(s, 1.0))
                    pct = 100.0 * w / total_w
                    ax_s.set_ylabel(
                        f"{s}\n{int(w)} wk ({pct:.1f}%)",
                        rotation=0,
                        labelpad=45,
                        ha="right",
                        va="center",
                        fontsize=FIG_FONT_SIZE - 1,
                    )

                    if row == 0:
                        ax_s.set_title("Solar", fontsize=FIG_FONT_SIZE + 1)
                        ax_w.set_title("Wind", fontsize=FIG_FONT_SIZE + 1)

                    for ax in (ax_s, ax_w):
                        ax.set_yticks(range(0, 7))
                        ax.set_yticklabels([f"{d+1}" for d in range(7)])
                        ax.set_xticks(range(0, 24, 4))
                        ax.set_xticklabels([str(h) for h in range(0, 24, 4)])

                    # Bottom labels only on the last used row in each group
                    if group == ncol_groups - 1:
                        remainder = n_ts % max_rows_per_col or nrows
                        last_used_row = remainder - 1
                    else:
                        last_used_row = nrows - 1

                    if row != last_used_row:
                        ax_s.tick_params(axis="x", which="both", labelbottom=False)
                        ax_w.tick_params(axis="x", which="both", labelbottom=False)
                    else:
                        ax_s.set_xlabel("Hour of day")
                        ax_w.set_xlabel("Hour of day")

                # Hide only the unused rows in the **final** Solar|Wind pair
                remainder = n_ts % max_rows_per_col
                if remainder != 0:
                    last_group = ncol_groups - 1
                    for r in range(remainder, nrows):
                        axes[r, 2 * last_group].set_visible(False)  # Solar cell
                        axes[r, 2 * last_group + 1].set_visible(False)  # Wind cell

                cbar = fig.colorbar(
                    last_im,
                    ax=axes.ravel().tolist(),
                    orientation="vertical",
                    pad=0.01,
                    fraction=0.02,
                )
                cbar.set_label(out_unit)

                fig.suptitle(
                    _wrap(
                        f"{scenario_name}: {zone} — Curtailment by representative week (Solar | Wind)"
                    ),
                    y=0.995,
                )

            # Save
            fname = _safe_filename(f"zone_{zone}_curtailment")
            fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight")
            plt.close(fig)
            print(f"[saved] {fname} -> {out_dir}")

        else:
            raise ValueError("aggregate must be 'week' or 'split_week'.")


# %%
# ========== TEMPLATE FOR NEW PLOTS ==========
def template_new_plot():
    """
    Copy this function, rename it, and add its toggle in PLOTS at the top.
    Use direct CSV reads from outputs_dir, and colors/types from the two input CSVs.
    Remember to save to <outputs_dir>/<SAVE_SUBDIR>/.
    """
    pass


# ========== DRIVER ==========
def main():
    if PLOTS.get("dispatch_matrix", False):
        dispatch_matrix()
    if PLOTS.get("generation_mix_stacked_bars", False):
        generation_mix_stacked_bars()
    if PLOTS.get("curtailment_matrix", False):
        curtailment_matrix()
    print("[done] all selected plots generated.")


if __name__ == "__main__":
    # Apply rcParams once (you already do this at top)
    # ---- DISPATCH ----
    if PLOTS["dispatch"]["dispatch_matrix"]:
        dispatch_matrix()
    if PLOTS["dispatch"]["dispatch_matrix_per_period"]:
        dispatch_matrix_per_period()
    if PLOTS["dispatch"]["dispatch_by_region"]:
        dispatch_by_region(
            zones="all",
            split_weeks=True,
            plot_style="line",
            unit="auto",
            legend_top_to_bottom=False,
        )

    # ---- GENERATION ----
    if PLOTS["generation"]["generation_mix_stacked_bars"]:
        generation_mix_stacked_bars(to_twh=True)
    if PLOTS["generation"]["generation_mix_stacked_bars_by_zone"]:
        generation_mix_stacked_bars_by_zone(
            periods=None, to_twh=True, facet_periods=False
        )

    # ---- CURTAILMENT ----
    if PLOTS["curtailment"]["curtailment_by_region"]:
        curtailment_by_region(
            zones="all", aggregate="split_week", split_by_source=False
        )
    if PLOTS["curtailment"]["curtailment_matrix"]:
        curtailment_matrix(aggregate="week", split_by_source=False)
