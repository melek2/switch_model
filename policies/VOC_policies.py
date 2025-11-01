# Copyright (c) 2025 The Switch Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0, which is in the LICENSE file.
"""
Add VOC emission cost policies to the model.

This module introduces an optional mechanism for accounting for the health cost
of fine Volatile Organic Compound (VOC) emissions from generation projects. Similar
to the carbon policies implemented in `carbon_policies.py`, this module can add
a VOC-related cost component to the objective function based on specified
per-generator cost rates.

Specifying `voc_cost_dollar_per_ton` for generators will add a term to the
objective function of the form:

    AnnualVOC_by_gen[g, p] * voc_cost_dollar_per_ton[g]

where:
    - `AnnualVOC_by_gen[g, p]` represents total VOC emissions (tons/year)
      produced by generator g in period p.
    - `voc_cost_dollar_per_ton[g]` is the health cost of VOC (in $/ton)
      assigned to generator g.

If no value is specified for a generator, the default VOC cost is zero,
meaning that generator’s VOC emissions will not affect the total system cost.
"""

from __future__ import division
import os
from pyomo.environ import Set, Param, Expression, Constraint, Suffix, NonNegativeReals
import switch_model.reporting as reporting


def define_components(model):
    """
    Define VOC-related parameters and expressions for the model.

    This function introduces:
        1. A parameter `voc_cost_dollar_per_ton[g]` that stores the health cost
           of VOC emissions per generator (in $/ton).
        2. An expression `AnnualVOC_by_gen[g, p]` that calculates the annual
           VOC emissions (tons) produced by each generator g in each period p.
        3. An expression `VOCCosts[p]` that aggregates the total VOC cost
           (in $) for each period.
        4. Inclusion of `VOCCosts` in the model’s list of cost components so that
           it is considered in the total objective function.

    Notes
    -----
    - Generators without a specified VOC cost are assigned a default of zero.
    - VOC emissions are assumed to be precomputed at the dispatch level
      (via `DispatchVOC[g, t, f]`), consistent with other emissions tracking.
    """

    # 1) Health cost of VOC emissions ($/ton) by generator
    model.voc_cost_dollar_per_ton = Param(
        model.GENERATION_PROJECTS,
        within=NonNegativeReals,
        default=0.0,
        doc="Health cost of VOC emissions ($/ton) for each generator project.",
    )

    # 2) Annual VOC emissions by generator and period (tons/year)
    model.AnnualVOC_by_gen = Expression(
        model.GENERATION_PROJECTS,
        model.PERIODS,
        rule=lambda m, g, p: sum(
            m.DispatchVOC[g, t, f] * m.tp_weight_in_year[t]
            for (g2, t, f) in m.GEN_TP_FUELS
            if g2 == g and m.tp_period[t] == p
        ),
        doc="Annual VOC emissions (tons) by generator and period.",
    )

    # 3) Total VOC cost per period ($)
    model.VOCCosts = Expression(
        model.PERIODS,
        rule=lambda m, p: sum(
            m.AnnualVOC_by_gen[g, p] * m.voc_cost_dollar_per_ton[g]
            for g in m.GENERATION_PROJECTS
        ),
        doc="Total VOC cost ($) across all generators for each period.",
    )

    # 4) Register VOC cost component in the total cost function
    model.Cost_Components_Per_Period.append("VOCCosts")


def load_inputs(model, switch_data, inputs_dir):
    """
    Load VOC cost data for each generator.

    Reads the file `gen_emission_costs.csv` from the inputs directory and
    assigns VOC cost values (in $/ton) to the parameter
    `voc_cost_dollar_per_ton[g]`.

    Expected file format (CSV)
    --------------------------
        GENERATION_PROJECT,voc_cost_dollar_per_ton

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
        param=(model.voc_cost_dollar_per_ton,),
        optional=True,
    )


def post_solve(model, outdir):
    """
    Export annual VOC emissions and costs after model solution.

    This function generates a CSV file `VOC.csv` containing annual VOC
    emissions and associated monetary costs for each period.

    Output file
    ------------
    VOC.csv
        Columns:
            PERIOD,
            AnnualVOC_base_units,
            VOCCost_dollar_per_period

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
            m.AnnualVOC[period],  # total VOC emitted in base units
            m.VOCCosts[period],   # total VOC cost ($)
        ]

    reporting.write_table(
        model,
        model.PERIODS,
        output_file=os.path.join(outdir, "VOC.csv"),
        headings=(
            "PERIOD",
            "AnnualVOC_base_units",
            "VOCCost_dollar_per_period",
        ),
        values=get_row,
    )

    reporting.write_table(
        model, model.GENERATION_PROJECTS * model.PERIODS,
        output_file=os.path.join(outdir, "VOC_by_generator.csv"),
        headings=("GENERATION_PROJECT","PERIOD","AnnualVOC_ton","VOCCost_$"),
        values=lambda m,g,p: (g, p, m.AnnualVOC_by_gen[g,p], m.AnnualVOC_by_gen[g,p]*m.voc_cost_dollar_per_ton[g]),
    )
def post_solve(model, outdir):
    """
    Export annual VOC emissions and costs after the model solves.

    This writes two CSV files summarizing VOC outcomes by period and by generator.

    Assumptions and units
    ---------------------
    GenFuelUseRate is in MMBtu_per_hour.
    gen_voc_intensity and f_voc_intensity are in ton_per_MMBtu.
    tp_weight_in_year is in hours_per_period.
    Therefore AnnualVOC is in ton_per_period and VOCCost is in dollars_per_period.

    Outputs
    -------
    1) VOC.csv
       Columns:
            PERIOD
            AnnualVOC_ton
            VOCCost_dollar_per_period

    2) VOC_by_generator.csv
        Columns:
            GENERATION_PROJECT
            PERIOD
            AnnualVOC_ton
            VOCCost_dollar_per_period

    Parameters
    ----------
    model : pyomo.environ.ConcreteModel
        Solved model instance containing AnnualVOC, AnnualVOC_by_gen, and VOCCosts.
    outdir : str
        Directory path where output files will be written.
    """

    def get_row(m, period):
        # total VOC emitted and its cost in the given period
        return [
            period,
            m.AnnualVOC[period],      # ton per period
            m.VOCCosts[period],       # dollar per period
        ]

    reporting.write_table(
        model,
        model.PERIODS,
        output_file=os.path.join(outdir, "VOC.csv"),
        headings=("PERIOD", "AnnualVOC_ton", "VOCCost_dollar_per_period"),
        values=get_row,
    )

    reporting.write_table(
        model,
        model.GENERATION_PROJECTS * model.PERIODS,
        output_file=os.path.join(outdir, "VOC_by_generator.csv"),
        headings=("GENERATION_PROJECT", "PERIOD", "AnnualVOC_ton", "VOCCost_dollar_per_period"),
        values=lambda m, g, p: (
            g,
            p,
            m.AnnualVOC_by_gen[g, p],
            m.AnnualVOC_by_gen[g, p] * m.voc_cost_dollar_per_ton[g],
        ),
    )

