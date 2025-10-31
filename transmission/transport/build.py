# Copyright (c) 2015-2024 The Switch Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0, which is in the LICENSE file.

"""
Defines transmission build-outs.
"""

import logging
import os

import pandas as pd
from pyomo.environ import *

from switch_model.financials import capital_recovery_factor as crf
from switch_model.utilities import unique_list

# from switch_model.tools.graph.main import graph

dependencies = (
    "switch_model.timescales",
    "switch_model.balancing.load_zones",
    "switch_model.financials",
)


def define_components(mod):
    """
    Defines a transport model for inter-zone transmission. Unless otherwise
    stated, all power capacity is specified in units of MW and all sets and
    parameters are mandatory.


    TRANSMISSION_LINES is the complete set of transmission pathways
    connecting load zones. Each member of this set is a one dimensional
    identifier such as "A-B". This set has no regard for directionality
    of transmission lines and will generate an error if you specify two
    lines that move in opposite directions such as (A to B) and (B to
    A). Another derived set - TRANS_LINES_DIRECTIONAL - stores
    directional information. Transmission may be abbreviated as trans or
    tx in parameter names or indexes.

    trans_lz1[tx] and trans_lz2[tx] specify the load zones at either end
    of a transmission line. The order of 1 and 2 is unimportant, but you
    are encouraged to be consistent to simplify merging information back
    into external databases.

    trans_dbid[tx in TRANSMISSION_LINES] is an external database
    identifier for each transmission line. This is an optional parameter
    than defaults to the identifier of the transmission line.

    trans_length_km[tx in TRANSMISSION_LINES] is the length of each
    transmission line in kilometers.

    trans_efficiency[tx in TRANSMISSION_LINES] is the proportion of
    energy sent down a line that is delivered. If 2 percent of energy
    sent down a line is lost, this value would be set to 0.98.

    trans_new_build_allowed[tx in TRANSMISSION_LINES] is a binary value
    indicating whether new transmission build-outs are allowed along a
    transmission line. This optional parameter defaults to True.

    TRANS_BLD_YRS is the set of transmission lines and future years in
    which they could be built. This set is composed of two
    elements with members: (tx, build_year). In a prior implementation,
    this set also contained existing transmission (with build_year typically
    set to 'Legacy'), but this changed in commit 868ca08 on June 13, 2019.

    existing_trans_cap[tx in TRANSMISSION_LINES] is a parameter that
    describes how many MW of capacity was been installed before the
    start of the study.

    BuildTx[(tx, bld_yr) in TRANS_BLD_YRS] is a decision variable
    that describes the transfer capacity in MW installed on a corridor
    in a given build year. For existing builds, this variable is locked
    to the existing capacity.

    TxCapacityNameplate[(tx, bld_yr) in TRANS_BLD_YRS] is an expression
    that returns the total nameplate transfer capacity of a transmission
    line in a given period. This is the sum of existing and newly-build
    capacity.

    trans_derating_factor[tx in TRANSMISSION_LINES] is an overall
    derating factor for each transmission line that can reflect forced
    outage rates, stability or contingency limitations. This parameter
    is optional and defaults to 1. This parameter should be in the
    range of 0 to 1. A value of 0 will disables the line completely.

    TxCapacityNameplateAvailable[(tx, bld_yr) in TRANS_BLD_YRS] is an
    expression that returns the available transfer capacity of a
    transmission line in a given period, taking into account the
    nameplate capacity and derating factor.

    trans_terrain_multiplier[tx in TRANSMISSION_LINES] is
    a cost adjuster applied to each transmission line that reflects the
    additional costs that may be incurred for traversing that specific
    terrain. Crossing mountains or cities will be more expensive than
    crossing plains. This parameter is optional and defaults to 1. This
    parameter should be in the range of 0.5 to 3.

    trans_capital_cost_per_mw_km describes the generic costs of building
    new transmission in units of $BASE_YEAR per MW transfer capacity per
    km. This is optional and defaults to 1000.

    trans_lifetime_yrs is the number of years in which a capital
    construction loan for a new transmission line is repaid. This
    optional parameter defaults to 20 years based on 2009 WREZ
    transmission model transmission data. At the end of this time,
    we assume transmission lines will be rebuilt at the same cost.

    trans_fixed_om_fraction describes the fixed Operations and
    Maintenance costs as a fraction of capital costs. This optional
    parameter defaults to 0.03 based on 2009 WREZ transmission model
    transmission data costs for existing transmission maintenance.

    trans_cost_hourly[tx TRANSMISSION_LINES] is the cost of building
    transmission lines in units of $BASE_YEAR / MW- transfer-capacity /
    hour. This derived parameter is based on the total annualized
    capital and fixed O&M costs, then divides that by hours per year to
    determine the portion of costs incurred hourly.

    DIRECTIONAL_TX is a derived set of directional paths that
    electricity can flow along transmission lines. Each element of this
    set is a two-dimensional entry that describes the origin and
    destination of the flow: (load_zone_from, load_zone_to). Every
    transmission line will generate two entries in this set. Members of
    this set are abbreviated as trans_d where possible, but may be
    abbreviated as tx in situations where brevity is important and it is
    unlikely to be confused with the overall transmission line.

    trans_d_line[trans_d] is the transmission line associated with this
    directional path.

    --- NOTES ---

    The cost stream over time for transmission lines differs from the
    Switch-WECC model. The Switch-WECC model assumed new transmission
    had a financial lifetime of 20 years, which was the length of the
    loan term. During this time, fixed operations & maintenance costs
    were also incurred annually and these were estimated to be 3 percent
    of the initial capital costs. These fixed O&M costs were obtained
    from the 2009 WREZ transmission model transmission data costs for
    existing transmission maintenance .. most of those lines were old
    and their capital loans had been paid off, so the O&M were the costs
    of keeping them operational. Switch-WECC basically assumed the lines
    could be kept online indefinitely with that O&M budget, with
    components of the lines being replaced as needed. This payment
    schedule and lifetimes was assumed to hold for both existing and new
    lines. This made the annual costs change over time, which could
    create edge effects near the end of the study period. Switch-WECC
    had different cost assumptions for local T&D; capital expenses and
    fixed O&M expenses were rolled in together, and those were assumed
    to continue indefinitely. This basically assumed that local T&D would
    be replaced at the end of its financial lifetime.

    Switch treats all transmission and distribution (long-
    distance or local) the same. Any capacity that is built will be kept
    online indefinitely. At the end of its financial lifetime, existing
    capacity will be retired and rebuilt, so the annual cost of a line
    upgrade will remain constant in every future year.

    """

    mod.TRANSMISSION_LINES = Set(dimen=1)
    mod.trans_lz1 = Param(mod.TRANSMISSION_LINES, within=mod.LOAD_ZONES)
    mod.trans_lz2 = Param(mod.TRANSMISSION_LINES, within=mod.LOAD_ZONES)
    # we don't do a min_data_check for TRANSMISSION_LINES, because it may be empty for model
    # configurations that are sometimes run with interzonal transmission and sometimes not
    # (e.g., island interconnect scenarios). However, presence of this column will still be
    # checked by load_data_aug.
    mod.min_data_check("trans_lz1", "trans_lz2")

    def _check_tx_duplicate_paths(m):
        forward_paths = set(
            [(m.trans_lz1[tx], m.trans_lz2[tx]) for tx in m.TRANSMISSION_LINES]
        )
        reverse_paths = set(
            [(m.trans_lz2[tx], m.trans_lz1[tx]) for tx in m.TRANSMISSION_LINES]
        )
        overlap = forward_paths.intersection(reverse_paths)
        if overlap:
            logging.error(
                "Transmission lines have bi-directional paths specified "
                "in input files. They are expected to specify a single path "
                "per pair of connected load zones. "
                "(Ex: either A->B or B->A, but not both). "
                "Over-specified lines: {}".format(overlap)
            )
            return False
        else:
            return True

    mod.check_tx_duplicate_paths = BuildCheck(rule=_check_tx_duplicate_paths)

    mod.trans_dbid = Param(mod.TRANSMISSION_LINES, default=lambda m, tx: tx, within=Any)
    mod.trans_length_km = Param(mod.TRANSMISSION_LINES, within=NonNegativeReals)
    mod.trans_efficiency = Param(mod.TRANSMISSION_LINES, within=PercentFraction)
    mod.existing_trans_cap = Param(mod.TRANSMISSION_LINES, within=NonNegativeReals)
    mod.min_data_check("trans_length_km", "trans_efficiency", "existing_trans_cap")
    mod.trans_new_build_allowed = Param(
        mod.TRANSMISSION_LINES, within=Boolean, default=True
    )
    mod.TRANS_BLD_YRS = Set(
        dimen=2,
        initialize=mod.TRANSMISSION_LINES * mod.PERIODS,
        filter=lambda m, tx, p: m.trans_new_build_allowed[tx],
    )
    mod.BuildTx = Var(mod.TRANS_BLD_YRS, within=NonNegativeReals)
    mod.TxCapacityNameplate = Expression(
        mod.TRANSMISSION_LINES,
        mod.PERIODS,
        rule=lambda m, tx, period: sum(
            m.BuildTx[tx, bld_yr]
            for bld_yr in m.PERIODS
            if bld_yr <= period and (tx, bld_yr) in m.TRANS_BLD_YRS
        )
        + m.existing_trans_cap[tx],
    )
    mod.trans_derating_factor = Param(
        mod.TRANSMISSION_LINES, within=PercentFraction, default=1
    )
    mod.TxCapacityNameplateAvailable = Expression(
        mod.TRANSMISSION_LINES,
        mod.PERIODS,
        rule=lambda m, tx, period: (
            m.TxCapacityNameplate[tx, period] * m.trans_derating_factor[tx]
        ),
    )
    mod.trans_terrain_multiplier = Param(
        mod.TRANSMISSION_LINES, within=NonNegativeReals, default=1
    )
    mod.trans_capital_cost_per_mw_km = Param(within=NonNegativeReals, default=1000)
    mod.trans_lifetime_yrs = Param(within=NonNegativeReals, default=20)
    mod.trans_fixed_om_fraction = Param(within=NonNegativeReals, default=0.03)
    # Total annual fixed costs for building new transmission lines...
    # Multiply capital costs by capital recover factor to get annual
    # payments. Add annual fixed O&M that are expressed as a fraction of
    # overnight costs.
    mod.trans_cost_annual = Param(
        mod.TRANSMISSION_LINES,
        within=NonNegativeReals,
        initialize=lambda m, tx: (
            m.trans_capital_cost_per_mw_km
            * m.trans_terrain_multiplier[tx]
            * m.trans_length_km[tx]
            * (crf(m.interest_rate, m.trans_lifetime_yrs) + m.trans_fixed_om_fraction)
        ),
    )
    # An expression to summarize annual costs for the objective
    # function. Units should be total annual future costs in $base_year
    # real dollars. The objective function will convert these to
    # base_year Net Present Value in $base_year real dollars.
    mod.TxFixedCosts = Expression(
        mod.PERIODS,
        rule=lambda m, p: sum(
            m.TxCapacityNameplate[tx, p] * m.trans_cost_annual[tx]
            for tx in m.TRANSMISSION_LINES
        ),
    )
    mod.Cost_Components_Per_Period.append("TxFixedCosts")

    def init_DIRECTIONAL_TX(model):
        tx_dir = []
        for tx in model.TRANSMISSION_LINES:
            tx_dir.append((model.trans_lz1[tx], model.trans_lz2[tx]))
            tx_dir.append((model.trans_lz2[tx], model.trans_lz1[tx]))
        return tx_dir

    mod.DIRECTIONAL_TX = Set(dimen=2, initialize=init_DIRECTIONAL_TX)
    mod.TX_CONNECTIONS_TO_ZONE = Set(
        mod.LOAD_ZONES,
        dimen=1,
        initialize=lambda m, lz: [
            z for z in m.LOAD_ZONES if (z, lz) in m.DIRECTIONAL_TX
        ],
    )

    def init_trans_d_line(m, zone_from, zone_to):
        for tx in m.TRANSMISSION_LINES:
            if (m.trans_lz1[tx] == zone_from and m.trans_lz2[tx] == zone_to) or (
                m.trans_lz2[tx] == zone_from and m.trans_lz1[tx] == zone_to
            ):
                return tx

    mod.trans_d_line = Param(
        mod.DIRECTIONAL_TX, within=mod.TRANSMISSION_LINES, initialize=init_trans_d_line
    )


def load_inputs(mod, switch_data, inputs_dir):
    """
    Import data related to transmission builds. The following files are
    expected in the input directory. Optional files & columns are marked with
    a *.

    transmission_lines.csv
        TRANSMISSION_LINE, trans_lz1, trans_lz2, trans_length_km,
        trans_efficiency, existing_trans_cap, trans_dbid*,
        trans_derating_factor*, trans_terrain_multiplier*,
        trans_new_build_allowed*

    Note that in the next file, parameter names are written on the first
    row (as usual), and the single value for each parameter is written in
    the second row.

    trans_params.csv*
        trans_capital_cost_per_mw_km*, trans_lifetime_yrs*,
        trans_fixed_om_fraction*
    """
    # TODO: send issue / pull request to Pyomo to allow .csv files with
    # no rows after header (fix bugs in pyomo.core.plugins.data.text)
    switch_data.load_aug(
        filename=os.path.join(inputs_dir, "transmission_lines.csv"),
        index=mod.TRANSMISSION_LINES,
        optional_params=(
            "trans_dbid",
            "trans_derating_factor",
            "trans_terrain_multiplier",
            "trans_new_build_allowed",
        ),
        param=(
            mod.trans_lz1,
            mod.trans_lz2,
            mod.trans_length_km,
            mod.trans_efficiency,
            mod.existing_trans_cap,
            mod.trans_dbid,
            mod.trans_derating_factor,
            mod.trans_terrain_multiplier,
            mod.trans_new_build_allowed,
        ),
    )
    switch_data.load_aug(
        filename=os.path.join(inputs_dir, "trans_params.csv"),
        optional=True,
        param=(
            mod.trans_capital_cost_per_mw_km,
            mod.trans_lifetime_yrs,
            mod.trans_fixed_om_fraction,
        ),
    )


def post_solve(instance, outdir):
    mod = instance
    normalized_dat = [
        {
            "TRANSMISSION_LINE": tx,
            "PERIOD": p,
            "trans_lz1": mod.trans_lz1[tx],
            "trans_lz2": mod.trans_lz2[tx],
            "trans_dbid": mod.trans_dbid[tx],
            "trans_length_km": mod.trans_length_km[tx],
            "trans_efficiency": mod.trans_efficiency[tx],
            "trans_derating_factor": mod.trans_derating_factor[tx],
            "TxCapacityNameplate": value(mod.TxCapacityNameplate[tx, p]),
            "TxCapacityNameplateAvailable": value(
                mod.TxCapacityNameplateAvailable[tx, p]
            ),
            "TotalAnnualCost": value(
                mod.TxCapacityNameplate[tx, p] * mod.trans_cost_annual[tx]
            ),
        }
        for tx, p in mod.TRANSMISSION_LINES * mod.PERIODS
    ]
    tx_build_df = pd.DataFrame(normalized_dat)
    tx_build_df.set_index(["TRANSMISSION_LINE", "PERIOD"], inplace=True)
    if instance.options.sorted_output:
        tx_build_df.sort_index(inplace=True)
    tx_build_df.to_csv(os.path.join(outdir, "transmission.csv"))


# all plotting is now done in post processing in switch_model/plot_switch_results.py
# @graph(
#     "transmission_builds_network",
#     title="Transmission network: existing vs new builds (total over all periods)",
#     supports_multi_scenario=False  # keep single scenario for clarity
# )
# def graph_transmission_builds_network(tools):
#     import math

#     lines = tools.get_dataframe("transmission_lines.csv",from_inputs=True)  # existing_trans_cap
#     tx = tools.get_dataframe("transmission.csv")  # line-period nameplate
#     bld = tools.get_dataframe("BuildTx.csv")  # TRANS_BLD_YRS_1 (line), TRANS_BLD_YRS_2 (period), BuildTx

#     # Corridor label
#     lines["corridor"] = lines[["trans_lz1", "trans_lz2"]].astype(str).agg(" - ".join, axis=1)

#     # Existing capacity per corridor
#     exist = lines.groupby("corridor", as_index=False)["existing_trans_cap"].sum()

#     # New builds per corridor = sum over all periods in BuildTx
#     bld = bld.rename(columns={"TRANS_BLD_YRS_1": "TRANSMISSION_LINE", "TRANS_BLD_YRS_2": "PERIOD"})
#     if "TRANSMISSION_LINE" in tx.columns:
#         # Map line id -> (lz1,lz2) using transmission (has line-per-period), fallback to transmission_lines
#         map_lines = tx.drop_duplicates("TRANSMISSION_LINE")[["TRANSMISSION_LINE", "trans_lz1", "trans_lz2"]]
#     else:
#         map_lines = lines[["TRANSMISSION_LINE", "trans_lz1", "trans_lz2"]]
#     bld = bld.merge(map_lines, on="TRANSMISSION_LINE", how="left")
#     bld["corridor"] = bld[["trans_lz1", "trans_lz2"]].astype(str).agg(" - ".join, axis=1)
#     newb = bld.groupby("corridor", as_index=False)["BuildTx"].sum()

#     # Normalize linewidths
#     exist = exist.set_index("corridor")["existing_trans_cap"]
#     newb = newb.set_index("corridor")["BuildTx"]
#     # Union of corridors
#     all_corridors = sorted(set(exist.index).union(set(newb.index)))

#     # Build a circular layout for load zones
#     zones = sorted(set(lines["trans_lz1"].astype(str)).union(set(lines["trans_lz2"].astype(str))))
#     n = max(1, len(zones))
#     coords = {z: (math.cos(2*math.pi*i/n), math.sin(2*math.pi*i/n)) for i, z in enumerate(zones)}

#     ax = tools.get_axes()
#     ax.set_xlabel("")
#     ax.set_ylabel("")
#     ax.set_xticks([])
#     ax.set_yticks([])

#     # Draw nodes
#     for z, (x, y) in coords.items():
#         ax.scatter([x], [y], s=40)
#         ax.text(x, y, z, ha="center", va="center", fontsize=8, clip_on=True)

#     # Helper to draw an edge
#     def draw_edge(a, b, lw, alpha):
#         x1, y1 = coords[a]
#         x2, y2 = coords[b]
#         ax.plot([x1, x2], [y1, y2], linewidth=lw, alpha=alpha)

#     # Determine scaling
#     exist_max = float(exist.max()) if len(exist) else 0.0
#     newb_max = float(newb.max()) if len(newb) else 0.0
#     # Avoid division by zero
#     exist_den = exist_max if exist_max > 0 else 1.0
#     newb_den = newb_max if newb_max > 0 else 1.0

#     # Draw each corridor: base = existing (light), overlay = new (darker)
#     for _, row in lines.drop_duplicates(subset=["trans_lz1", "trans_lz2"]).iterrows():
#         a = str(row["trans_lz1"]); b = str(row["trans_lz2"])
#         key = f"{a} - {b}"
#         ecap = float(exist.get(key, 0.0))
#         ncap = float(newb.get(key, 0.0))
#         lw_e = 0.5 + 2.0 * (ecap / exist_den)
#         lw_n = 0.5 + 2.0 * (ncap / newb_den)
#         # existing
#         draw_edge(a, b, lw_e, alpha=0.4)
#         # overlay new
#         if ncap > 0:
#             draw_edge(a, b, lw_n, alpha=0.85)

#     # Legend hint
#     ax.text(0.01, 0.99, "Line width ∝ capacity\n(light = existing, dark = new builds)", transform=ax.transAxes, va="top", fontsize=8)


# @graph(
#     "transmission_builds_line",
#     title="New transmission builds by corridor",
#     supports_multi_scenario=True
# )
# def graph_transmission_builds_line(tools):
#     bld = tools.get_dataframe("BuildTx.csv")
#     lines = tools.get_dataframe("transmission.csv")

#     # Build corridor label from the two zone columns present in BuildTx
#     # Expecting TRANS_BLD_YRS_1 and TRANS_BLD_YRS_2 to identify the corridor; if not, fall back to trans_lz1/trans_lz2
#     corridor_cols = [c for c in bld.columns if c.upper().startswith("TRANS_BLD_YRS_")]
#     if len(corridor_cols) >= 2:
#         a, b = corridor_cols[:2]
#         bld["corridor"] = bld[[a, b]].astype(str).agg(" - ".join, axis=1)
#     else:
#         # Try to map via transmission_lines if a dbid exists
#         if "TRANSMISSION_LINE" in lines.columns:
#             # If bld has a line id, map to zones
#             key = "TRANSMISSION_LINE"
#             if key in bld.columns:
#                 m = lines.set_index(key)[["trans_lz1", "trans_lz2"]]
#                 bld = bld.join(m, on=key)
#                 bld["corridor"] = bld[["trans_lz1", "trans_lz2"]].astype(str).agg(" - ".join, axis=1)
#             else:
#                 bld["corridor"] = "Unknown corridor"
#         else:
#             bld["corridor"] = "Unknown corridor"

#     period_col = "PERIOD" if "PERIOD" in bld.columns else "period"
#     val_col = "BuildTx"

#     groupby = [period_col, "corridor"] if tools.num_scenarios == 1 else [period_col, "scenario_name", "corridor"]
#     bld_agg = bld.groupby(groupby, as_index=False)[val_col].sum()

#     # Cumulative sum across periods within each corridor
#     bld_agg = bld_agg.sort_values(groupby)
#     if tools.num_scenarios == 1:
#         bld_agg["cum"] = bld_agg.groupby("corridor")[val_col].cumsum()
#         idx = [period_col]
#     else:
#         bld_agg["cum"] = bld_agg.groupby(["scenario_name", "corridor"])[val_col].cumsum()
#         idx = [period_col, "scenario_name"]

#     # Pivot to lines per corridor
#     pivot = bld_agg.pivot_table(index=idx, columns="corridor", values="cum", aggfunc="last").fillna(0)

#     ax = tools.get_axes()
#     pivot.plot(ax=ax, kind="line", xlabel="Period", ylabel="Capacity built (MW)")

# @graph(
#     "transmission_builds_network",
#     title="Transmission network (existing + new builds)",
#     supports_multi_scenario=False  # keep single for clarity; multi can be added later
# )
# def graph_transmission_builds_network(tools):
#     lines = tools.get_dataframe("transmission_lines.csv")
#     # Optional: add BuildTx to show new vs existing; if not present, show existing only
#     try:
#         bld = tools.get_dataframe("BuildTx.csv")
#     except Exception:
#         bld = None

#     # Construct node set
#     zones = sorted(set(lines["trans_lz1"].astype(str)).union(set(lines["trans_lz2"].astype(str))))
#     import math
#     n = len(zones)
#     # Simple circle layout
#     coords = {z: (math.cos(2*math.pi*i/n), math.sin(2*math.pi*i/n)) for i, z in enumerate(zones)}

#     # Aggregate existing capacity per corridor
#     lines["corridor"] = lines[["trans_lz1", "trans_lz2"]].astype(str).agg(" - ".join, axis=1)
#     exist = lines.groupby("corridor", as_index=False)["existing_trans_cap"].sum()

#     # Aggregate new builds across all periods
#     if bld is not None and "BuildTx" in bld.columns:
#         # Try to map to lz1-lz2 if available
#         if "TRANSMISSION_LINE" in lines.columns and "TRANSMISSION_LINE" in bld.columns:
#             m = lines.set_index("TRANSMISSION_LINE")[["trans_lz1", "trans_lz2"]]
#             bld = bld.join(m, on="TRANSMISSION_LINE")
#         build_corridor = bld.dropna(subset=["trans_lz1", "trans_lz2"]) if {"trans_lz1", "trans_lz2"} <= set(bld.columns) else None
#         if build_corridor is not None:
#             bld["corridor"] = bld[["trans_lz1", "trans_lz2"]].astype(str).agg(" - ".join, axis=1)
#             newb = bld.groupby("corridor", as_index=False)["BuildTx"].sum()
#         else:
#             newb = exist.copy()
#             newb["BuildTx"] = 0.0
#     else:
#         newb = exist.copy()
#         newb["BuildTx"] = 0.0

#     newb = newb.set_index("corridor")["BuildTx"]
#     exist = exist.set_index("corridor")["existing_trans_cap"]

#     # Drawing
#     ax = tools.get_axes()
#     ax.set_xlabel("")
#     ax.set_ylabel("")
#     ax.set_xticks([])
#     ax.set_yticks([])

#     # Draw nodes
#     for z, (x, y) in coords.items():
#         ax.scatter([x], [y], s=50)
#         ax.text(x, y, z, ha="center", va="center", fontsize=8, clip_on=True)

#     # Draw edges with linewidth scaled by capacity
#     for _, row in lines.iterrows():
#         a = str(row["trans_lz1"])
#         b = str(row["trans_lz2"])
#         x1, y1 = coords[a]
#         x2, y2 = coords[b]
#         corridor = f"{a} - {b}"
#         cap_exist = float(exist.get(corridor, 0.0))
#         cap_new = float(newb.get(corridor, 0.0))
#         lw_exist = 0.5 + 2.0 * (cap_exist / max(exist.max() if exist.max() > 0 else 1.0, 1.0))
#         lw_new = 0.5 + 2.0 * (cap_new / max(newb.max() if newb.max() > 0 else 1.0, 1.0))
#         # Existing line
#         ax.plot([x1, x2], [y1, y2], linewidth=lw_exist, alpha=0.5)
#         # Overlay new build as a thicker line on top
#         if cap_new > 0:
#             ax.plot([x1, x2], [y1, y2], linewidth=lw_new, alpha=0.9)

#     tools.add_note("Line width encodes capacity. Darker overlay indicates new builds.")
