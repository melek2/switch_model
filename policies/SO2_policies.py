# Copyright (c) 2025 The Switch Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0, which is in the LICENSE file.
"""
Add SO2 emission cost policies to the model.

This module introduces an optional mechanism for accounting for the health cost
of fine Sulfure Dioxide (SO2) emissions from generation projects. Similar
to the carbon policies implemented in `carbon_policies.py`, this module can add
a SO2-related cost component to the objective function based on specified
per-generator cost rates.

Specifying `so2_cost_dollar_per_ton` for generators will add a term to the
objective function of the form:

    AnnualSO2_by_gen[g, p] * so2_cost_dollar_per_ton[g]

where:
    - `AnnualSO2_by_gen[g, p]` represents total SO2 emissions (tons/year)
      produced by generator g in period p.
    - `so2_cost_dollar_per_ton[g]` is the health cost of SO2 (in $/ton)
      assigned to generator g.

If no value is specified for a generator, the default SO2 cost is zero,
meaning that generator’s SO2 emissions will not affect the total system cost.
"""

from __future__ import division
import os
from pyomo.environ import Set, Param, Expression, Constraint, Suffix, NonNegativeReals
import switch_model.reporting as reporting


def define_components(model):
    """
    Define SO2-related parameters and expressions for the model.

    This function introduces:
        1. A parameter `so2_cost_dollar_per_ton[g]` that stores the health cost
           of SO2 emissions per generator (in $/ton).
        2. An expression `AnnualSO2_by_gen[g, p]` that calculates the annual
           SO2 emissions (tons) produced by each generator g in each period p.
        3. An expression `SO2Costs[p]` that aggregates the total SO2 cost
           (in $) for each period.
        4. Inclusion of `SO2Costs` in the model’s list of cost components so that
           it is considered in the total objective function.

    Notes
    -----
    - Generators without a specified SO2 cost are assigned a default of zero.
    - SO2 emissions are assumed to be precomputed at the dispatch level
      (via `DispatchSO2[g, t, f]`), consistent with other emissions tracking.
    """

    # 1) Health cost of SO2 emissions ($/ton) by generator
    model.so2_cost_dollar_per_ton = Param(
        model.GENERATION_PROJECTS,
        within=NonNegativeReals,
        default=0.0,
        doc="Health cost of SO2 emissions ($/ton) for each generator project.",
    )

    # 2) Annual SO2 emissions by generator and period (tons/year)
    model.AnnualSO2_by_gen = Expression(
        model.GENERATION_PROJECTS,
        model.PERIODS,
        rule=lambda m, g, p: sum(
            m.DispatchSO2[g, t, f] * m.tp_weight_in_year[t]
            for (g2, t, f) in m.GEN_TP_FUELS
            if g2 == g and m.tp_period[t] == p
        ),
        doc="Annual SO2 emissions (tons) by generator and period.",
    )

    # 3) Total SO2 cost per period ($)
    model.SO2Costs = Expression(
        model.PERIODS,
        rule=lambda m, p: sum(
            m.AnnualSO2_by_gen[g, p] * m.so2_cost_dollar_per_ton[g]
            for g in m.GENERATION_PROJECTS
        ),
        doc="Total SO2 cost ($) across all generators for each period.",
    )

    # 4) Register SO2 cost component in the total cost function
    model.Cost_Components_Per_Period.append("SO2Costs")


def load_inputs(model, switch_data, inputs_dir):
    """
    Load SO2 cost data for each generator.

    Reads the file `gen_emission_costs.csv` from the inputs directory and
    assigns SO2 cost values (in $/ton) to the parameter
    `so2_cost_dollar_per_ton[g]`.

    Expected file format (CSV)
    --------------------------
        GENERATION_PROJECT,so2_cost_dollar_per_ton

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
        param=(model.so2_cost_dollar_per_ton,),
        optional=True,
    )


def post_solve(model, outdir):
    """
    Export annual SO2 emissions and costs after model solution.

    This function generates a CSV file `SO2.csv` containing annual SO2
    emissions and associated monetary costs for each period.

    Output file
    ------------
    SO2.csv
        Columns:
            PERIOD,
            AnnualSO2_base_units,
            SO2Cost_dollar_per_period

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
            m.AnnualSO2[period],  # total SO2 emitted in base units
            m.SO2Costs[period],   # total SO2 cost ($)
        ]

    reporting.write_table(
        model,
        model.PERIODS,
        output_file=os.path.join(outdir, "SO2.csv"),
        headings=(
            "PERIOD",
            "AnnualSO2_base_units",
            "SO2Cost_dollar_per_period",
        ),
        values=get_row,
    )

    reporting.write_table(
        model, model.GENERATION_PROJECTS * model.PERIODS,
        output_file=os.path.join(outdir, "SO2_by_generator.csv"),
        headings=("GENERATION_PROJECT","PERIOD","AnnualSO2_ton","SO2Cost_$"),
        values=lambda m,g,p: (g, p, m.AnnualSO2_by_gen[g,p], m.AnnualSO2_by_gen[g,p]*m.so2_cost_dollar_per_ton[g]),
    )
def post_solve(model, outdir):
    """
    Export annual SO2 emissions and costs after the model solves.

    This writes two CSV files summarizing SO2 outcomes by period and by generator.

    Assumptions and units
    ---------------------
    GenFuelUseRate is in MMBtu_per_hour.
    gen_so2_intensity and f_so2_intensity are in ton_per_MMBtu.
    tp_weight_in_year is in hours_per_period.
    Therefore AnnualSO2 is in ton_per_period and SO2Cost is in dollars_per_period.

    Outputs
    -------
    1) SO2.csv
       Columns:
            PERIOD
            AnnualSO2_ton
            SO2Cost_dollar_per_period

    2) SO2_by_generator.csv
        Columns:
            GENERATION_PROJECT
            PERIOD
            AnnualSO2_ton
            SO2Cost_dollar_per_period

    Parameters
    ----------
    model : pyomo.environ.ConcreteModel
        Solved model instance containing AnnualSO2, AnnualSO2_by_gen, and SO2Costs.
    outdir : str
        Directory path where output files will be written.
    """

    def get_row(m, period):
        # total SO2 emitted and its cost in the given period
        return [
            period,
            m.AnnualSO2[period],      # ton per period
            m.SO2Costs[period],       # dollar per period
        ]

    reporting.write_table(
        model,
        model.PERIODS,
        output_file=os.path.join(outdir, "SO2.csv"),
        headings=("PERIOD", "AnnualSO2_ton", "SO2Cost_dollar_per_period"),
        values=get_row,
    )

    reporting.write_table(
        model,
        model.GENERATION_PROJECTS * model.PERIODS,
        output_file=os.path.join(outdir, "SO2_by_generator.csv"),
        headings=("GENERATION_PROJECT", "PERIOD", "AnnualSO2_ton", "SO2Cost_dollar_per_period"),
        values=lambda m, g, p: (
            g,
            p,
            m.AnnualSO2_by_gen[g, p],
            m.AnnualSO2_by_gen[g, p] * m.so2_cost_dollar_per_ton[g],
        ),
    )