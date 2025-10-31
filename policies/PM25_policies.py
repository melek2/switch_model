# switch_model/policies/PM25_policies.py
from __future__ import division
import os
from pyomo.environ import Set, Param, Expression, Constraint, Suffix, NonNegativeReals
import switch_model.reporting as reporting


def define_components(model):
    """
    Add PM2.5 cost policies to the model, analogous to carbon_policies.
    """

    # 1) parameter for PM2.5 cost per generator ($ per ton)
    # for any project not listed in gen_pm25_costs.csv, default cost = 0
    # NEW: for any project not listed in gen_emission_costs.csv, default cost = 0
    model.pm25_cost_dollar_per_ton = Param(
        model.GENERATION_PROJECTS,
        within=NonNegativeReals,
        default=0.0,
        doc="Social cost of PM2.5 ($/ton) by generator",
    )
    # # 2) define PM2.5 cost expression per period
    # model.PM25Costs = Expression(
    #     model.PERIODS,
    #     rule=lambda m, p: sum(
    #         m.AnnualPM25[g,p] * m.pm25_cost_dollar_per_ton[g]
    #         for g in m.GENERATION_PROJECTS
    #     ),
    #     doc="PM2.5 cost component ($) per period"
    # )

    # 2a) per‐generator annual PM2.5 emissions (tons/year)
    model.AnnualPM25_by_gen = Expression(
        model.GENERATION_PROJECTS,
        model.PERIODS,
        rule=lambda m, g, p: sum(
            m.DispatchPM25[g, t, f] * m.tp_weight_in_year[t]
            for (g2, t, f) in m.GEN_TP_FUELS
            if g2 == g and m.tp_period[t] == p
        ),
        doc="Annual PM2.5 emissions (tons) by generator and period",
    )

    # 2b) total PM2.5 cost in each period
    model.PM25Costs = Expression(
        model.PERIODS,
        rule=lambda m, p: sum(
            m.AnnualPM25_by_gen[g, p] * m.pm25_cost_dollar_per_ton[g]
            for g in m.GENERATION_PROJECTS
        ),
        doc="PM2.5 cost component ($) per period",
    )

    # # 3) append to list of cost components so objective picks it up
    model.Cost_Components_Per_Period.append("PM25Costs")


def load_inputs(model, switch_data, inputs_dir):
    """
    # OLD: Load per-generator PM2.5 cost data (in $/ton) from gen_pm25_costs.csv.
    NEW: Load per-generator PM2.5 cost data (in $/ton) from gen_emission_costs.csv.
    Expected file format (with header):
      GENERATION_PROJECT,pm25_cost_dollar_per_ton
    """
    switch_data.load_aug(
        # filename=os.path.join(inputs_dir, "gen_pm25_costs.csv"),   # OLD
        filename=os.path.join(inputs_dir, "gen_emission_costs.csv"),  # NEW
        index=model.GENERATION_PROJECTS,
        param=(model.pm25_cost_dollar_per_ton,),
        optional=True,
    )


def post_solve(model, outdir):
    """
    Export annual PM2.5 metrics to PM25.csv
    """

    def get_row(m, period):
        return [
            period,
            # total PM2.5 emitted in base units (from AnnualPM25),
            m.AnnualPM25[period],
            # total PM2.5 cost ($) in this period
            m.PM25Costs[period],
        ]

    reporting.write_table(
        model,
        model.PERIODS,
        output_file=os.path.join(outdir, "PM25.csv"),
        headings=(
            "PERIOD",
            "AnnualPM25_base_units",
            "PM25Cost_dollar_per_period",
        ),
        values=get_row,
    )
