# Copyright (c) 2015-2024 The Switch Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0, which is in the LICENSE file.

"""
Defines model components to describe transmission dispatch for the
Switch model.
"""

from pyomo.environ import *
# from switch_model.tools.graph.main import graph, Figure

dependencies = (
    "switch_model.timescales",
    "switch_model.balancing.load_zones",
    "switch_model.financials",
    "switch_model.transmission.transport.build",
)


def define_components(mod):
    """

    Adds components to a Pyomo abstract model object to describe the
    dispatch of transmission resources in an electric grid. This
    includes parameters, dispatch decisions and constraints. Unless
    otherwise stated, all power capacity is specified in units of MW,
    all energy amounts are specified in units of MWh, and all sets and
    parameters are mandatory.

    TRANS_TIMEPOINTS describes the scope that transmission dispatch
    decisions must be made over. It is defined as the set of
    DIRECTIONAL_TX crossed with TIMEPOINTS. It is indexed as
    (load_zone_from, load_zone_to, timepoint) and may be abbreviated as
    [z_from, zone_to, tp] for brevity.

    DispatchTx[z_from, zone_to, tp] is the decision of how much power
    to send along each transmission line in a particular direction in
    each timepoint.

    Maximum_DispatchTx is a constraint that forces DispatchTx to
    stay below the bounds of installed capacity.

    TxPowerSent[z_from, zone_to, tp] is an expression that describes the
    power sent down a transmission line. This is completely determined by
    DispatchTx[z_from, zone_to, tp].

    TxPowerReceived[z_from, zone_to, tp] is an expression that describes the
    power sent down a transmission line. This is completely determined by
    DispatchTx[z_from, zone_to, tp] and trans_efficiency[tx].

    TXPowerNet[z, tp] is an expression that returns the net power from
    transmission for a load zone. This is the sum of TxPowerReceived by
    the load zone minus the sum of TxPowerSent by the load zone.

    """

    mod.TRANS_TIMEPOINTS = Set(
        dimen=3, initialize=lambda m: m.DIRECTIONAL_TX * m.TIMEPOINTS
    )
    mod.DispatchTx = Var(mod.TRANS_TIMEPOINTS, within=NonNegativeReals)

    mod.Maximum_DispatchTx = Constraint(
        mod.TRANS_TIMEPOINTS,
        rule=lambda m, zone_from, zone_to, tp: (
            m.DispatchTx[zone_from, zone_to, tp]
            <= m.TxCapacityNameplateAvailable[
                m.trans_d_line[zone_from, zone_to], m.tp_period[tp]
            ]
        ),
    )

    mod.TxPowerSent = Expression(
        mod.TRANS_TIMEPOINTS,
        rule=lambda m, zone_from, zone_to, tp: (m.DispatchTx[zone_from, zone_to, tp]),
    )
    mod.TxPowerReceived = Expression(
        mod.TRANS_TIMEPOINTS,
        rule=lambda m, zone_from, zone_to, tp: (
            m.DispatchTx[zone_from, zone_to, tp]
            * m.trans_efficiency[m.trans_d_line[zone_from, zone_to]]
        ),
    )

    def TXPowerNet_calculation(m, z, tp):
        return sum(
            m.TxPowerReceived[zone_from, z, tp]
            for zone_from in m.TX_CONNECTIONS_TO_ZONE[z]
        ) - sum(
            m.TxPowerSent[z, zone_to, tp] for zone_to in m.TX_CONNECTIONS_TO_ZONE[z]
        )

    mod.TXPowerNet = Expression(
        mod.LOAD_ZONES, mod.TIMEPOINTS, rule=TXPowerNet_calculation
    )
    # Register net transmission as contributing to zonal energy balance
    mod.Zone_Power_Injections.append("TXPowerNet")
# all plotting is now done in post processing in switch_model/plot_switch_results.py

# @graph(
#     "transmission_utilization_heatmap",
#     title="Transmission utilization (from × to) — mean over time",
#     supports_multi_scenario=False
# )
# def graph_transmission_utilization_heatmap(tools):
#     import numpy as np
#     import pandas as pd
#     import matplotlib.pyplot as plt

#     # 1) Load flows and recover endpoints + timepoint
#     flows = tools.get_dataframe("DispatchTx.csv").copy()
#     if "DispatchTx" not in flows.columns:
#         # Fallback common variants
#         cand = [c for c in flows.columns if c.lower() in {"dispatchtx", "tx_dispatch", "flow_mw", "tx_flow"}]
#         if not cand:
#             raise RuntimeError("Could not find transmission flow column (expected 'DispatchTx').")
#         flows.rename(columns={cand[0]: "DispatchTx"}, inplace=True)

#     tcols = [c for c in flows.columns if c.upper().startswith("TRANS_TIMEPOINTS_")]
#     if len(tcols) >= 3:
#         from_col, to_col, tp_col = tcols[:3]
#         flows["from_zone"] = flows[from_col].astype(str)
#         flows["to_zone"] = flows[to_col].astype(str)
#         flows["timepoint_id"] = flows[tp_col]
#     else:
#         # Best-effort: try to guess common names
#         guess_from = next((c for c in flows.columns if c.lower() in {"from", "from_zone", "lz1", "load_zone_from", "tx_lz1"}), None)
#         guess_to   = next((c for c in flows.columns if c.lower() in {"to", "to_zone", "lz2", "load_zone_to", "tx_lz2"}), None)
#         if guess_from and guess_to:
#             flows["from_zone"] = flows[guess_from].astype(str)
#             flows["to_zone"] = flows[guess_to].astype(str)
#         else:
#             # Single synthetic corridor (still builds a 1×1 matrix)
#             flows["from_zone"] = "all"
#             flows["to_zone"] = "all"
#         flows["timepoint_id"] = np.arange(len(flows))

#     # Normalize to numeric
#     flows["DispatchTx"] = pd.to_numeric(flows["DispatchTx"], errors="coerce").fillna(0.0)
#     flows["abs_flow"] = flows["DispatchTx"].abs()

#     # 2) Try to get nameplate capacities per corridor (direction-agnostic)
#     def _try_load_capacity():
#         # Most common input name is 'transmission_lines.csv'
#         for fname in ["transmission_lines.csv", "TxCapacity.csv", "TxLimits.csv"]:
#             try:
#                 df = tools.get_dataframe(fname, from_inputs=True).copy()
#             except Exception:
#                 continue
#             if df is None or df.empty:
#                 continue

#             cols_l = {c.lower(): c for c in df.columns}
#             # Heuristic: find from/to columns
#             from_candidates = [k for k in cols_l if k in {"from", "from_zone", "lz1", "load_zone_from", "tx_lz1"}]
#             to_candidates   = [k for k in cols_l if k in {"to", "to_zone", "lz2", "load_zone_to", "tx_lz2"}]
#             if not from_candidates or not to_candidates:
#                 # Try generic "lz" pairs
#                 lz_cols = [k for k in cols_l if k.startswith("lz")]
#                 if len(lz_cols) >= 2:
#                     from_candidates = [lz_cols[0]]
#                     to_candidates = [lz_cols[1]]
#             if not from_candidates or not to_candidates:
#                 continue

#             fcol = cols_l[from_candidates[0]]
#             tcol = cols_l[to_candidates[0]]

#             # Capacity-like column
#             cap_candidates = [c for c in df.columns if any(s in c.lower() for s in ["cap", "limit", "rating", "mw"])]
#             # Pick numeric one with largest mean value
#             cap_col = None
#             for c in cap_candidates:
#                 if pd.api.types.is_numeric_dtype(df[c]):
#                     cap_col = c if cap_col is None else (c if df[c].abs().mean() > df[cap_col].abs().mean() else cap_col)
#             if cap_col is None:
#                 continue

#             cap = df[[fcol, tcol, cap_col]].copy()
#             cap["from_zone"] = cap[fcol].astype(str)
#             cap["to_zone"] = cap[tcol].astype(str)
#             # Combine parallel lines if any
#             cap = cap.groupby(["from_zone", "to_zone"], as_index=False)[cap_col].sum()
#             cap.rename(columns={cap_col: "capacity_mw"}, inplace=True)
#             # Make direction-agnostic key for merging both ways
#             cap["pair_key"] = cap.apply(lambda r: tuple(sorted([r["from_zone"], r["to_zone"]])), axis=1)
#             return cap[["pair_key", "capacity_mw"]]
#         return None

#     cap_df = _try_load_capacity()

#     # 3) Build denominator per directed pair
#     flows["pair_key"] = flows.apply(lambda r: tuple(sorted([r["from_zone"], r["to_zone"]])), axis=1)
#     if cap_df is not None and not cap_df.empty:
#         denom = flows[["pair_key"]].merge(cap_df, on="pair_key", how="left")["capacity_mw"]
#         flows["denom"] = denom
#         # Fallback to observed peak if capacity missing/zero
#         miss = flows["denom"].isna() | (flows["denom"] <= 0)
#         if miss.any():
#             peak = flows.groupby("pair_key")["abs_flow"].max().rename("peak_obs")
#             flows.loc[miss, "denom"] = flows.loc[miss, "pair_key"].map(peak).fillna(1.0)
#     else:
#         # No capacity file; use observed peak
#         peak = flows.groupby("pair_key")["abs_flow"].max().rename("peak_obs")
#         flows["denom"] = flows["pair_key"].map(peak).fillna(1.0)

#     # 4) Aggregate over time to a single utilization statistic (mean of |flow|/denom)
#     flows["util"] = flows["abs_flow"] / flows["denom"].replace(0, np.nan)
#     util = (flows
#             .groupby(["from_zone", "to_zone"], as_index=False)["util"]
#             .mean())
#     # Optional: zero out diagonal (no self-flows)
#     util = util[util["from_zone"] != util["to_zone"]]

#     # 5) Pivot to adjacency matrix (from × to)
#     zones = sorted(set(util["from_zone"]) | set(util["to_zone"]))
#     mat = (util.pivot(index="from_zone", columns="to_zone", values="util")
#                .reindex(index=zones, columns=zones)
#                .fillna(0.0))

#     # 6) Plot heatmap
#     ax = tools.get_axes()
#     im = ax.imshow(mat.values, vmin=0, vmax=1, aspect="equal", interpolation="nearest")
#     ax.set_xticks(range(len(zones)))
#     ax.set_yticks(range(len(zones)))
#     ax.set_xticklabels(zones, rotation=90)
#     ax.set_yticklabels(zones)
#     ax.set_xlabel("To zone")
#     ax.set_ylabel("From zone")
#     cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#     cbar.set_label("Mean utilization (|flow| / capacity)")

#     # Optional numeric annotations for prominent corridors
#     try:
#         top = (util.assign(val=lambda d: d["util"])
#                     .nlargest(10, "val"))
#         for _, r in top.iterrows():
#             i = zones.index(r["from_zone"])
#             j = zones.index(r["to_zone"])
#             ax.text(j, i, f"{r['util']:.2f}", ha="center", va="center", fontsize=8)
#     except Exception:
#         pass

#     ax.set_title("Transmission utilization matrix (time-averaged)")
#     ax.figure.tight_layout()

# @graph(
#     "transmission_utilization_line",
#     title="Transmission flow over time (all corridors)",
#     supports_multi_scenario=True
# )
# def graph_transmission_utilization_line(tools):
#     flows = tools.get_dataframe("DispatchTx.csv")

#     # Recover corridor + timepoint
#     tcols = [c for c in flows.columns if c.upper().startswith("TRANS_TIMEPOINTS_")]
#     if len(tcols) >= 3:
#         t1, t2, t3 = tcols[:3]
#         flows["corridor"] = flows[[t1, t2]].astype(str).agg(" - ".join, axis=1)
#         flows["timepoint_id"] = flows[t3]
#     else:
#         flows["corridor"] = "all"
#         flows["timepoint_id"] = range(len(flows))

#     # Map to timestamp if available
#     try:
#         tps = tools.get_dataframe("timepoints.csv")
#         tp_key = "timepoint_id" if "timepoint_id" in tps.columns else (tps.columns[0])
#         if tp_key in tps.columns and "timestamp" in tps.columns:
#             ts_map = tps.set_index(tp_key)["timestamp"]
#             flows["timestamp"] = flows["timepoint_id"].map(ts_map)
#         else:
#             flows["timestamp"] = flows["timepoint_id"]
#     except Exception:
#         flows["timestamp"] = flows["timepoint_id"]

#     # If multi-scenario, pick the scenario with the largest total flow to show all its corridors in one plot
#     if tools.num_scenarios > 1 and "scenario_name" in flows.columns:
#         # total DispatchTx per scenario
#         scn_totals = flows.groupby("scenario_name")["DispatchTx"].sum().sort_values(ascending=False)
#         sel_scn = scn_totals.index[0]
#         sub = flows[flows["scenario_name"] == sel_scn].copy()
#         note_extra = f"\nScenario shown: {sel_scn}"
#     else:
#         sub = flows.copy()
#         note_extra = ""

#     # One line per corridor on the same axes
#     # Sum across any duplicate rows (e.g., multiple entries per timestamp/corridor)
#     pivot = (
#         sub.groupby(["timestamp", "corridor"], as_index=False)["DispatchTx"].sum()
#            .pivot(index="timestamp", columns="corridor", values="DispatchTx")
#            .sort_index()
#     )

#     ax = tools.get_axes()
#     if isinstance(ax, tuple):
#         fig, ax = ax
#     else:
#         fig = ax.figure

#     # Plot all corridors together
#     pivot.plot(ax=ax, kind="line", xlabel="Time", ylabel="Flow (MW)")
#     ax.legend(title="Corridor", bbox_to_anchor=(1.02, 1), loc="upper left")

#     f = Figure(fig, ax)
#     # Helpful note
#     f.add_note(("All corridors shown" if pivot.shape[1] > 1 else "Single corridor") + note_extra)
