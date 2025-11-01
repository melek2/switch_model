# Copyright (c) 2025 The Switch Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0, which is in the LICENSE file.
"""
Add PM2.5 emission cost policies to the model.

This module introduces an optional mechanism for accounting for the health cost
of fine particulate matter (PM2.5) emissions from generation projects. Similar
to the carbon policies implemented in `carbon_policies.py`, this module can add
a PM2.5-related cost component to the objective function based on specified
per-generator cost rates.

Specifying `pm25_cost_dollar_per_ton` for generators will add a term to the
objective function of the form:

    AnnualPM25_by_gen[g, p] * pm25_cost_dollar_per_ton[g]

where:
    - `AnnualPM25_by_gen[g, p]` represents total PM2.5 emissions (tons/year)
      produced by generator g in period p.
    - `pm25_cost_dollar_per_ton[g]` is the social cost of PM2.5 (in $/ton)
      assigned to generator g.

If no value is specified for a generator, the default PM2.5 cost is zero,
meaning that generator’s PM2.5 emissions will not affect the total system cost.
"""

from __future__ import division
import os
from pyomo.environ import Set, Param, Expression, Constraint, Suffix, NonNegativeReals
import switch_model.reporting as reporting


def define_components(model):
    """
    Define PM2.5-related parameters and expressions for the model.

    This function introduces:
        1. A parameter `pm25_cost_dollar_per_ton[g]` that stores the social cost
           of PM2.5 emissions per generator (in $/ton).
        2. An expression `AnnualPM25_by_gen[g, p]` that calculates the annual
           PM2.5 emissions (tons) produced by each generator g in each period p.
        3. An expression `PM25Costs[p]` that aggregates the total PM2.5 cost
           (in $) for each period.
        4. Inclusion of `PM25Costs` in the model’s list of cost components so that
           it is considered in the total objective function.

    Notes
    -----
    - Generators without a specified PM2.5 cost are assigned a default of zero.
    - PM2.5 emissions are assumed to be precomputed at the dispatch level
      (via `DispatchPM25[g, t, f]`), consistent with other emissions tracking.
    """

    # 1) Health cost of PM2.5 emissions ($/ton) by generator
    model.pm25_cost_dollar_per_ton = Param(
        model.GENERATION_PROJECTS,
        within=NonNegativeReals,
        default=0.0,
        doc="Health cost of PM2.5 emissions ($/ton) for each generator project.",
    )

    # 2) Annual PM2.5 emissions by generator and period (tons/year)
    model.AnnualPM25_by_gen = Expression(
        model.GENERATION_PROJECTS,
        model.PERIODS,
        rule=lambda m, g, p: sum(
            m.DispatchPM25[g, t, f] * m.tp_weight_in_year[t]
            for (g2, t, f) in m.GEN_TP_FUELS
            if g2 == g and m.tp_period[t] == p
        ),
        doc="Annual PM2.5 emissions (tons) by generator and period.",
    )

    # 3) Total PM2.5 cost per period ($)
    model.PM25Costs = Expression(
        model.PERIODS,
        rule=lambda m, p: sum(
            m.AnnualPM25_by_gen[g, p] * m.pm25_cost_dollar_per_ton[g]
            for g in m.GENERATION_PROJECTS
        ),
        doc="Total PM2.5 cost ($) across all generators for each period.",
    )

    # 4) Register PM2.5 cost component in the total cost function
    model.Cost_Components_Per_Period.append("PM25Costs")


def load_inputs(model, switch_data, inputs_dir):
    """
    Load PM2.5 cost data for each generator.

    Reads the file `gen_emission_costs.csv` from the inputs directory and
    assigns PM2.5 cost values (in $/ton) to the parameter
    `pm25_cost_dollar_per_ton[g]`.

    Expected file format (CSV)
    --------------------------
        GENERATION_PROJECT,pm25_cost_dollar_per_ton

    Parameters
    ----------
    model : pyomo.environ.ConcreteModel
        The Pyomo model instance to which components are being added.
    switch_data : switch_model.utilities.SwitchData
        Interface for reading input data files.
    inputs_dir : str
        Path to the directory containing model input files.

    Notes
    -----
    - Any generator not listed in the input file will default to 0 cost.
    - This file may also contain other pollutant costs for extensibility.
    """
    switch_data.load_aug(
        filename=os.path.join(inputs_dir, "gen_emission_costs.csv"),
        index=model.GENERATION_PROJECTS,
        param=(model.pm25_cost_dollar_per_ton,),
        optional=True,
    )


def post_solve(model, outdir):
    """
    Export annual PM2.5 emissions and costs after model solution.

    This function generates a CSV file `PM25.csv` containing annual PM2.5
    emissions and associated monetary costs for each period.

    Output file
    ------------
    PM25.csv
        Columns:
            PERIOD,
            AnnualPM25_base_units,
            PM25Cost_dollar_per_period

    Parameters
    ----------
    model : pyomo.environ.ConcreteModel
        The solved model instance.
    outdir : str
        Directory path where the output file will be written.
    """

    def get_row(m, period):
        return [
            period,
            m.AnnualPM25[period],  # total PM2.5 emitted in base units
            m.PM25Costs[period],   # total PM2.5 cost ($)
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
