# Copyright (c) 2025 The Switch Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0, which is in the LICENSE file.
"""
Add NOx emission cost policies to the model.

This module introduces an optional mechanism for accounting for the health cost
of fine Nitrogen Oxide (NOx) emissions from generation projects. Similar
to the carbon policies implemented in `carbon_policies.py`, this module can add
a NOx-related cost component to the objective function based on specified
per-generator cost rates.

Specifying `nox_cost_dollar_per_ton` for generators will add a term to the
objective function of the form:

    AnnualNOx_by_gen[g, p] * nox_cost_dollar_per_ton[g]

where:
    - `AnnualNOx_by_gen[g, p]` represents total NOx emissions (tons/year)
      produced by generator g in period p.
    - `nox_cost_dollar_per_ton[g]` is the health cost of NOx (in $/ton)
      assigned to generator g.

If no value is specified for a generator, the default NOx cost is zero,
meaning that generator’s NOx emissions will not affect the total system cost.
"""

from __future__ import division
import os
from pyomo.environ import Set, Param, Expression, Constraint, Suffix, NonNegativeReals
import switch_model.reporting as reporting


def define_components(model):
    """
    Define NOx-related parameters and expressions for the model.

    This function introduces:
        1. A parameter `nox_cost_dollar_per_ton[g]` that stores the health cost
           of NOx emissions per generator (in $/ton).
        2. An expression `AnnualNOx_by_gen[g, p]` that calculates the annual
           NOx emissions (tons) produced by each generator g in each period p.
        3. An expression `NOxCosts[p]` that aggregates the total NOx cost
           (in $) for each period.
        4. Inclusion of `NOxCosts` in the model’s list of cost components so that
           it is considered in the total objective function.

    Notes
    -----
    - Generators without a specified NOx cost are assigned a default of zero.
    - NOx emissions are assumed to be precomputed at the dispatch level
      (via `DispatchNOx[g, t, f]`), consistent with other emissions tracking.
    """

    # 1) Health cost of NOx emissions ($/ton) by generator
    model.nox_cost_dollar_per_ton = Param(
        model.GENERATION_PROJECTS,
        within=NonNegativeReals,
        default=0.0,
        doc="Health cost of NOx emissions ($/ton) for each generator project.",
    )

    # 2) Annual NOx emissions by generator and period (tons/year)
    model.AnnualNOx_by_gen = Expression(
        model.GENERATION_PROJECTS,
        model.PERIODS,
        rule=lambda m, g, p: sum(
            m.DispatchNOx[g, t, f] * m.tp_weight_in_year[t]
            for (g2, t, f) in m.GEN_TP_FUELS
            if g2 == g and m.tp_period[t] == p
        ),
        doc="Annual NOx emissions (tons) by generator and period.",
    )

    # 3) Total NOx cost per period ($)
    model.NOxCosts = Expression(
        model.PERIODS,
        rule=lambda m, p: sum(
            m.AnnualNOx_by_gen[g, p] * m.nox_cost_dollar_per_ton[g]
            for g in m.GENERATION_PROJECTS
        ),
        doc="Total NOx cost ($) across all generators for each period.",
    )

    # 4) Register NOx cost component in the total cost function
    model.Cost_Components_Per_Period.append("NOxCosts")


def load_inputs(model, switch_data, inputs_dir):
    """
    Load NOx cost data for each generator.

    Reads the file `gen_emission_costs.csv` from the inputs directory and
    assigns NOx cost values (in $/ton) to the parameter
    `nox_cost_dollar_per_ton[g]`.

    Expected file format (CSV)
    --------------------------
        GENERATION_PROJECT,nox_cost_dollar_per_ton

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
        param=(model.nox_cost_dollar_per_ton,),
        optional=True,
    )


def post_solve(model, outdir):
    """
    Export annual NOx emissions and costs after model solution.

    This function generates a CSV file `NOx.csv` containing annual NOx
    emissions and associated monetary costs for each period.

    Output file
    ------------
    NOx.csv
        Columns:
            PERIOD,
            AnnualNOx_base_units,
            NOxCost_dollar_per_period

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
            m.AnnualNOx[period],  # total NOx emitted in base units
            m.NOxCosts[period],   # total NOx cost ($)
        ]

    reporting.write_table(
        model,
        model.PERIODS,
        output_file=os.path.join(outdir, "NOx.csv"),
        headings=(
            "PERIOD",
            "AnnualNOx_base_units",
            "NOxCost_dollar_per_period",
        ),
        values=get_row,
    )

    reporting.write_table(
        model, model.GENERATION_PROJECTS * model.PERIODS,
        output_file=os.path.join(outdir, "NOx_by_generator.csv"),
        headings=("GENERATION_PROJECT","PERIOD","AnnualNOx_ton","NOxCost_$"),
        values=lambda m,g,p: (g, p, m.AnnualNOx_by_gen[g,p], m.AnnualNOx_by_gen[g,p]*m.nox_cost_dollar_per_ton[g]),
    )
def post_solve(model, outdir):
    """
    Export annual NOx emissions and costs after the model solves.

    This writes two CSV files summarizing NOx outcomes by period and by generator.

    Assumptions and units
    ---------------------
    GenFuelUseRate is in MMBtu_per_hour.
    gen_nox_intensity and f_nox_intensity are in ton_per_MMBtu.
    tp_weight_in_year is in hours_per_period.
    Therefore AnnualNOx is in ton_per_period and NOxCost is in dollars_per_period.

    Outputs
    -------
    1) NOx.csv
       Columns:
            PERIOD
            AnnualNOx_ton
            NOxCost_dollar_per_period

    2) NOx_by_generator.csv
        Columns:
            GENERATION_PROJECT
            PERIOD
            AnnualNOx_ton
            NOxCost_dollar_per_period

    Parameters
    ----------
    model : pyomo.environ.ConcreteModel
        Solved model instance containing AnnualNOx, AnnualNOx_by_gen, and NOxCosts.
    outdir : str
        Directory path where output files will be written.
    """

    def get_row(m, period):
        # total NOx emitted and its cost in the given period
        return [
            period,
            m.AnnualNOx[period],      # ton per period
            m.NOxCosts[period],       # dollar per period
        ]

    reporting.write_table(
        model,
        model.PERIODS,
        output_file=os.path.join(outdir, "NOx.csv"),
        headings=("PERIOD", "AnnualNOx_ton", "NOxCost_dollar_per_period"),
        values=get_row,
    )

    reporting.write_table(
        model,
        model.GENERATION_PROJECTS * model.PERIODS,
        output_file=os.path.join(outdir, "NOx_by_generator.csv"),
        headings=("GENERATION_PROJECT", "PERIOD", "AnnualNOx_ton", "NOxCost_dollar_per_period"),
        values=lambda m, g, p: (
            g,
            p,
            m.AnnualNOx_by_gen[g, p],
            m.AnnualNOx_by_gen[g, p] * m.nox_cost_dollar_per_ton[g],
        ),
    )

