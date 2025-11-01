# Copyright (c) 2015-2024 The Switch Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0, which is in the LICENSE file.

"""
Defines model components to describe generation projects build-outs for
the Switch model. This module requires either generators.core.unitcommit or
generators.core.no_commit to constrain project dispatch to either committed or
installed capacity.

"""
from __future__ import division

import logging
import os, collections

import pandas as pd
from pyomo.environ import *

from switch_model.reporting import write_table
from switch_model.utilities import unwrap

# from switch_model.tools.graph.main import graph

dependencies = (
    "switch_model.timescales",
    "switch_model.balancing.load_zones",
    "switch_model.financials",
    "switch_model.energy_sources.properties",
    "switch_model.generators.core.build",
)
optional_dependencies = "switch_model.transmission.local_td"


def define_components(mod):
    """

    Adds components to a Pyomo abstract model object to describe the
    dispatch decisions and constraints of generation and storage
    projects. Unless otherwise stated, all power capacity is specified
    in units of MW and all sets and parameters are mandatory.

    GEN_TPS is a set of projects and timepoints in which
    they can be dispatched. A dispatch decisions is made for each member
    of this set. Members of this set can be abbreviated as (g, t) or
    (g, t).

    TPS_FOR_GEN[g] is a set array showing all timepoints when a
    project is active. These are the timepoints corresponding to
    PERIODS_FOR_GEN. This is the same data as GEN_TPS,
    but split into separate sets for each project.

    TPS_FOR_GEN_IN_PERIOD[g, period] is the same as
    TPS_FOR_GEN, but broken down by period. Periods when
    the project is inactive will yield an empty set.

    GenCapacityInTP[(g, t) in GEN_TPS] is the same as
    GenCapacity but indexed by timepoint rather than period to allow
    more compact statements.

    DispatchGen[(g, t) in GEN_TPS] is the set
    of generation dispatch decisions: how much average power in MW to
    produce in each timepoint. This value can be multiplied by the
    duration of the timepoint in hours to determine the energy produced
    by a project in a timepoint.

    gen_forced_outage_rate[g] and gen_scheduled_outage_rate[g]
    describe the forces and scheduled outage rates for each project.
    These parameters can be specified for individual projects via an
    input file (see load_inputs() documentation), or generically for all
    projects of a given generation technology via
    g_scheduled_outage_rate and g_forced_outage_rate. You will get an
    error if any project is missing values for either of these
    parameters.

    gen_availability[g] describes the fraction of a time a project is
    expected to be available. This is derived from the forced and
    scheduled outage rates of the project. For baseload or flexible
    baseload, this is determined from both forced and scheduled outage
    rates. For all other types of generation technologies, we assume the
    scheduled outages can be performed when the generators were not
    scheduled to produce power, so their availability is only derated
    based on their forced outage rates.

    gen_max_capacity_factor[g, t] is defined for variable renewable
    projects and is the ratio of average power output to nameplate
    capacity in that timepoint. Most renewable capacity factors should
    be in the range of 0 to 1. Some solar capacity factors will be above
    1 because the nameplate capacity is based on solar radiation of 1.0
    kW/m^2 and solar radiation can exceed that value on very clear days
    or on partially cloudy days when light bounces off the bottom of
    clouds onto a solar panel. Some solar thermal capacity factors can
    be less than 0 because of auxillary loads: for example, parts of
    those plants need to be kept warm during winter nights to avoid
    freezing. Those heating loads can be significant during certain
    timepoints.

    gen_variable_om[g] is the variable Operations and Maintenance
    costs (O&M) per MWh of dispatched capacity for a given project.

    gen_full_load_heat_rate[g] is the full load heat rate in units
    of MMBTU/MWh that describes the thermal efficiency of a project when
    running at full load. This optional parameter overrides the generic
    heat rate of a generation technology. In the future, we may expand
    this to be indexed by fuel source as well if we need to support a
    multi-fuel generator whose heat rate depends on fuel source.

    Proj_Var_Costs_Hourly[t in TIMEPOINTS] is the sum of all variable
    costs associated with project dispatch for each timepoint expressed
    in $base_year/hour in the future period (rather than Net Present
    Value).

    FUEL_BASED_GEN_TPS is a subset of GEN_TPS
    showing all times when fuel-consuming projects could be dispatched
    (used to identify timepoints when fuel use must match power production).

    GEN_TP_FUELS is a subset of GEN_TPS * FUELS,
    showing all the valid combinations of project, timepoint and fuel,
    i.e., all the times when each project could consume a fuel that is
    limited, costly or produces emissions.

    GenFuelUseRate[(g, t, f) in GEN_TP_FUELS] is a
    variable that describes fuel consumption rate in MMBTU/h. This
    should be constrained to the fuel consumed by a project in each
    timepoint and can be calculated as Dispatch [MW] *
    effective_heat_rate [MMBTU/MWh] -> [MMBTU/h]. The choice of how to
    constrain it depends on the treatment of unit commitment. Currently
    the project.no_commit module implements a simple treatment that
    ignores unit commitment and assumes a full load heat rate, while the
    project.unitcommit module implements unit commitment decisions with
    startup fuel requirements and a marginal heat rate.

    DispatchEmissions[(g, t, f) in GEN_TP_FUELS] is the
    emissions produced by dispatching a fuel-based project in units of
    metric tonnes CO2 per hour. This is derived from the fuel
    consumption GenFuelUseRate, the fuel's direct carbon intensity, the
    fuel's upstream emissions, as well as Carbon Capture efficiency for
    generators that implement Carbon Capture and Sequestration. This does
    not yet support multi-fuel generators.

    AnnualEmissions[p in PERIODS]:The system's annual emissions, in metric
    tonnes of CO2 per year.

    gen_pm25_intensity[g in GENERATION_PROJECTS] is an optional
    generator-level PM2.5 emission intensity in units of tonnes/MMBtu.
    This parameter allows individual generators to override the
    fuel-level PM2.5 intensity specified in fuels.csv. If the value for
    a given generator is left at its default of 0.0, the model will fall
    back to the corresponding f_pm25_intensity[f] for the fuel consumed
    by that generator. Values are loaded from
    inputs/gen_emission_costs.csv (column: gen_pm25_intensity).
    
    DispatchPM25[(g, t, f) in GEN_TP_FUELS] is the instantaneous
    PM2.5 emission rate from generator g at timepoint t when using fuel f,
    expressed in tonnes per hour. It is calculated as
    DispatchPM25[g, t, f] = GenFuelUseRate[g, t, f] * intensity,
    where GenFuelUseRate[g, t, f] is the fuel consumption rate in
    MMBtu/hour, and intensity is determined per generator as
    gen_pm25_intensity[g] (tonnes/MMBtu) if greater than 0, otherwise
    f_pm25_intensity[f] (tonnes/MMBtu) from fuels.csv. This rule ensures
    that generator-specific emission factors, when available, take
    precedence over fuel-level defaults.
    
    AnnualPM25[p in PERIODS] is the total PM2.5 emissions aggregated
    over all generators, fuels, and timepoints within period p,
    expressed in tonnes per year. It is computed as
    AnnualPM25[p] = Σ_(g,t,f) DispatchPM25[g, t, f] * tp_weight_in_year[t]
    for all (g, t, f) with tp_period[t] = p. This expression scales each
    timepoint’s hourly emissions by its representative hours in a typical
    year to obtain annual totals.

    gen_pm25_intensity[g in GENERATION_PROJECTS] is an optional
    generator-level PM2.5 emission intensity in units of tonnes/MMBtu.
    This parameter allows individual generators to override the
    fuel-level PM2.5 intensity specified in fuels.csv. If the value for
    a given generator is left at its default of 0.0, the model will fall
    back to the corresponding f_pm25_intensity[f] for the fuel consumed
    by that generator. Values are loaded from
    inputs/gen_emission_costs.csv (column: gen_pm25_intensity).
    
    DispatchNOx[(g, t, f) in GEN_TP_FUELS] is the instantaneous
    NOx emission rate from generator g at timepoint t when using fuel f,
    expressed in tonnes per hour. It is calculated as
    DispatchNOx[g, t, f] = GenFuelUseRate[g, t, f] * intensity,
    where GenFuelUseRate[g, t, f] is the fuel consumption rate in
    MMBtu/hour, and intensity is determined per generator as
    gen_pm25_intensity[g] (tonnes/MMBtu) if greater than 0, otherwise
    f_pm25_intensity[f] (tonnes/MMBtu) from fuels.csv. This rule ensures
    that generator-specific emission factors, when available, take
    precedence over fuel-level defaults.

    AnnualNOx[p in PERIODS] is the total NOx emissions aggregated
    over all generators, fuels, and timepoints within period p,
    expressed in tonnes per year. It is computed as
    AnnualNOx[p] = Σ_(g,t,f) DispatchNOx[g, t, f] * tp_weight_in_year[t]
    for all (g, t, f) with tp_period[t] = p. This expression scales each
    timepoint’s hourly emissions by its representative hours in a typical
    year to obtain annual totals.

    Flexible baseload support for plants that can ramp slowly over the
    course of days. These kinds of generators can provide important
    seasonal support in high renewable and low emission futures.

    Parasitic loads that make solar thermal plants consume energy from
    the grid on cold nights to keep their fluids from getting too cold.

    Storage support.

    Hybrid project support (pumped hydro & CAES) will eventually get
    implemented in separate modules.

    """

    def period_active_gen_rule(m, period):
        if not hasattr(m, "period_active_gen_dict"):
            m.period_active_gen_dict = dict()
            for _g, _period in m.GEN_PERIODS:
                m.period_active_gen_dict.setdefault(_period, []).append(_g)
        result = m.period_active_gen_dict.pop(period)
        if len(m.period_active_gen_dict) == 0:
            delattr(m, "period_active_gen_dict")
        return result

    mod.GENS_IN_PERIOD = Set(
        mod.PERIODS,
        dimen=1,
        initialize=period_active_gen_rule,
        doc="The set of projects active in a given period.",
    )

    mod.TPS_FOR_GEN = Set(
        mod.GENERATION_PROJECTS,
        dimen=1,
        within=mod.TIMEPOINTS,
        initialize=lambda m, g: (
            tp for p in m.PERIODS_FOR_GEN[g] for tp in m.TPS_IN_PERIOD[p]
        ),
    )

    def init(m, gen, period):
        try:
            d = m._TPS_FOR_GEN_IN_PERIOD_dict
        except AttributeError:
            d = m._TPS_FOR_GEN_IN_PERIOD_dict = dict()
            for _gen in m.GENERATION_PROJECTS:
                for t in m.TPS_FOR_GEN[_gen]:
                    d.setdefault((_gen, m.tp_period[t]), []).append(t)
        result = d.pop((gen, period), [])
        if not d:  # all gone, delete the attribute
            del m._TPS_FOR_GEN_IN_PERIOD_dict
        return result

    mod.TPS_FOR_GEN_IN_PERIOD = Set(
        mod.GENERATION_PROJECTS,
        mod.PERIODS,
        dimen=1,
        within=mod.TIMEPOINTS,
        initialize=init,
    )

    mod.GEN_TPS = Set(
        dimen=2,
        initialize=lambda m: (
            (g, tp) for g in m.GENERATION_PROJECTS for tp in m.TPS_FOR_GEN[g]
        ),
    )
    mod.VARIABLE_GEN_TPS = Set(
        dimen=2,
        initialize=lambda m: (
            (g, tp) for g in m.VARIABLE_GENS for tp in m.TPS_FOR_GEN[g]
        ),
    )
    mod.FUEL_BASED_GEN_TPS = Set(
        dimen=2,
        initialize=lambda m: (
            (g, tp) for g in m.FUEL_BASED_GENS for tp in m.TPS_FOR_GEN[g]
        ),
    )
    mod.GEN_TP_FUELS = Set(
        dimen=3,
        initialize=lambda m: (
            (g, t, f) for (g, t) in m.FUEL_BASED_GEN_TPS for f in m.FUELS_FOR_GEN[g]
        ),
    )

    mod.GenCapacityInTP = Expression(
        mod.GEN_TPS, rule=lambda m, g, t: m.GenCapacity[g, m.tp_period[t]]
    )
    mod.DispatchGen = Var(mod.GEN_TPS, within=NonNegativeReals)
    mod.ZoneTotalCentralDispatch = Expression(
        mod.LOAD_ZONES,
        mod.TIMEPOINTS,
        rule=lambda m, z, t: sum(
            m.DispatchGen[p, t]
            for p in m.GENS_IN_ZONE[z]
            if (p, t) in m.GEN_TPS and not m.gen_is_distributed[p]
        )
        - sum(
            m.DispatchGen[p, t] * m.gen_ccs_energy_load[p]
            for p in m.GENS_IN_ZONE[z]
            if (p, t) in m.GEN_TPS and p in m.CCS_EQUIPPED_GENS
        ),
        doc="Net power from grid-tied generation projects.",
    )
    mod.Zone_Power_Injections.append("ZoneTotalCentralDispatch")

    # Divide distributed generation into a separate expression so that we can
    # put it in the distributed node's power balance equations if local_td is
    # included.
    mod.ZoneTotalDistributedDispatch = Expression(
        mod.LOAD_ZONES,
        mod.TIMEPOINTS,
        rule=lambda m, z, t: sum(
            m.DispatchGen[g, t]
            for g in m.GENS_IN_ZONE[z]
            if (g, t) in m.GEN_TPS and m.gen_is_distributed[g]
        ),
        doc="Total power from distributed generation projects.",
    )
    try:
        mod.Distributed_Power_Injections.append("ZoneTotalDistributedDispatch")
    except AttributeError:
        mod.Zone_Power_Injections.append("ZoneTotalDistributedDispatch")

    def init_gen_availability(m, g):
        if g in m.BASELOAD_GENS:
            return (1 - m.gen_forced_outage_rate[g]) * (
                1 - m.gen_scheduled_outage_rate[g]
            )
        else:
            return 1 - m.gen_forced_outage_rate[g]

    mod.gen_availability = Param(
        mod.GENERATION_PROJECTS,
        within=NonNegativeReals,
        initialize=init_gen_availability,
    )

    mod.VARIABLE_GEN_TPS_RAW = Set(dimen=2, within=mod.VARIABLE_GENS * mod.TIMEPOINTS)
    mod.gen_max_capacity_factor = Param(
        mod.VARIABLE_GEN_TPS_RAW,
        within=Reals,
        validate=lambda m, val, g, t: -1 < val < 2,
    )
    # Validate that a gen_max_capacity_factor has been defined for every
    # variable gen / timepoint that we need. Extra cap factors (like beyond an
    # existing plant's lifetime) shouldn't cause any problems.
    # This replaces: mod.min_data_check('gen_max_capacity_factor') from when
    # gen_max_capacity_factor was indexed by VARIABLE_GEN_TPS.
    mod.have_minimal_gen_max_capacity_factors = BuildCheck(
        mod.VARIABLE_GEN_TPS, rule=lambda m, g, t: (g, t) in m.VARIABLE_GEN_TPS_RAW
    )

    if mod.logger.isEnabledFor(logging.INFO):
        # Tell user if the input files specify timeseries for renewable plant
        # capacity factors that extend beyond the lifetime of the plant.
        def rule(m):
            extra_indexes = m.VARIABLE_GEN_TPS_RAW - m.VARIABLE_GEN_TPS
            if extra_indexes:
                num_impacted_generators = len(set(g for g, t in extra_indexes))
                extraneous = {g: [] for (g, t) in extra_indexes}
                for g, t in extra_indexes:
                    extraneous[g].append(t)
                pprint = "\n".join(
                    "* {}: {} to {}".format(g, min(tps), max(tps))
                    for g, tps in extraneous.items()
                )
                # basic message for everyone at info level
                msg = unwrap(
                    """
                    {} generation project{} data in
                    variable_capacity_factors.csv for timepoints when they are
                    not operable, either before construction is possible or
                    after retirement.
                """.format(
                        num_impacted_generators,
                        " has" if num_impacted_generators == 1 else "s have",
                    )
                )
                if m.logger.isEnabledFor(logging.DEBUG):
                    # more detailed message
                    msg += unwrap(
                        """
                         You can avoid this message by only placing data in
                        variable_capacity_factors.csv for active periods for
                        each project. If you expect these project[s] to be
                        operable during  all the timepoints currently in
                        variable_capacity_factors.csv, then they need to either
                        come online earlier, have longer lifetimes, or have
                        options to build new capacity when the old capacity
                        reaches its maximum age.
                    """
                    )
                    msg += " Plants with extra timepoints:\n{}".format(pprint)
                else:
                    msg += " Use --log-level debug for more details."
                m.logger.info(msg + "\n")

        mod.notify_on_extra_VARIABLE_GEN_TPS = BuildAction(rule=rule)

    mod.GenFuelUseRate = Var(
        mod.GEN_TP_FUELS,
        within=NonNegativeReals,
        doc=(
            "Other modules constraint this variable based on DispatchGen and "
            "module-specific formulations of unit commitment and heat rates."
        ),
    )

    def DispatchEmissions_rule(m, g, t, f):
        if g not in m.CCS_EQUIPPED_GENS:
            return m.GenFuelUseRate[g, t, f] * (
                m.f_co2_intensity[f] + m.f_upstream_co2_intensity[f]
            )
        else:
            ccs_emission_frac = 1 - m.gen_ccs_capture_efficiency[g]
            return m.GenFuelUseRate[g, t, f] * (
                m.f_co2_intensity[f] * ccs_emission_frac + m.f_upstream_co2_intensity[f]
            )

    mod.DispatchEmissions = Expression(mod.GEN_TP_FUELS, rule=DispatchEmissions_rule)
    mod.AnnualEmissions = Expression(
        mod.PERIODS,
        rule=lambda m, period: sum(
            m.DispatchEmissions[g, t, f] * m.tp_weight_in_year[t]
            for (g, t, f) in m.GEN_TP_FUELS
            if m.tp_period[t] == period
        ),
        doc="The system's annual emissions, in metric tonnes of CO2 per year.",
    )

    # PM2.5 emission rate rule
    # Units:
    #   - GenFuelUseRate[g,t,f]: MMBtu/hour
    #   - gen_pm25_intensity[g], f_pm25_intensity[f]: tonnes/MMBtu
    #   => DispatchPM25[g,t,f]: tonnes/hour
    # Fallback (per generator):
    #   use gen_pm25_intensity[g] if > 0 else f_pm25_intensity[f].
    already_reported_pm25 = set()

    def DispatchPM25_rule(m, g, t, f):
        if m.gen_pm25_intensity[g] > 0:
            intensity = m.gen_pm25_intensity[g]
        else:
            intensity = m.f_pm25_intensity[f]
            # Only print once per generator if the fallback value is nonzero
            if g not in already_reported_pm25 and value(m.f_pm25_intensity[f]) != 0:
                print(
                    f"[INFO - PM2.5] No gen_pm25_intensity for {g} found (or reported value = 0) in gen_emission_costs.csv. "
                    f"Using nonzero fallback f_pm25_intensity[{f}] = {value(m.f_pm25_intensity[f])}."
                )
                already_reported_pm25.add(g)
        return m.GenFuelUseRate[g, t, f] * intensity

    # Generator-level override (optional). If left at default 0.0, the model
    # falls back to f_pm25_intensity[f] for that generator.
    # Loaded via load_inputs() from inputs/gen_emission_costs.csv.
    mod.gen_pm25_intensity = Param(
        mod.GENERATION_PROJECTS,
        within=NonNegativeReals,
        default=0.0,
        doc="Generator-level PM2.5 intensity (tonnes/MMBtu).",
    )

    # Instantaneous PM2.5 emission rate (tonnes/hour) by (g,t,f)
    mod.DispatchPM25 = Expression(
        mod.GEN_TP_FUELS,
        rule=DispatchPM25_rule,
        doc="PM2.5 emission rate (mass per hour) from each generator, fuel, and timepoint.",
    )

    # Annual PM2.5 aggregation by period (tonnes/year)
    def annual_pm25_rule(m, period):
        return sum(
            m.DispatchPM25[g, t, f] * m.tp_weight_in_year[t]
            for (g, t, f) in m.GEN_TP_FUELS
            if m.tp_period[t] == period
        )

    mod.AnnualPM25 = Expression(
        mod.PERIODS,
        rule=annual_pm25_rule,
        doc="Total PM2.5 emissions (in base mass units per year) aggregated over all generators and fuels.",
        )

    # NOx emission rate rule
    # Units:
    #   - GenFuelUseRate[g,t,f]: MMBtu/hour
    #   - gen_nox_intensity[g], f_nox_intensity[f]: tonnes/MMBtu
    #   => DispatchNOx[g,t,f]: tonnes/hour
    # Fallback (per generator):
    #   use gen_NOx_intensity[g] if > 0 else f_nox_intensity[f].
    already_reported_nox = set()
    def DispatchNOx_rule(m, g, t, f):
        if m.gen_nox_intensity[g] > 0:
            intensity = m.gen_nox_intensity[g]
        else:
            intensity = m.f_nox_intensity[f]
            # Only print once per generator if the fallback value is nonzero
            if g not in already_reported_nox and value(m.f_nox_intensity[f]) != 0:
                print(
                    f"[INFO - NOx] No gen_nox_intensity for {g} found (or reported value = 0) in gen_emission_costs.csv. "
                    f"Using nonzero fallback f_nox_intensity[{f}] = {value(m.f_nox_intensity[f])}."
                )
                already_reported_nox.add(g)
        return m.GenFuelUseRate[g, t, f] * intensity

    # Generator-level override (optional). If left at default 0.0, the model
    # falls back to f_nox_intensity[f] for that generator.
    # Loaded via load_inputs() from inputs/gen_emission_costs.csv.
    mod.gen_nox_intensity = Param(
        mod.GENERATION_PROJECTS,
        within=NonNegativeReals,
        default=0.0,
        doc="Generator-level NOx intensity (tonnes/MMBtu).",
    )

    # Instantaneous Nox emission rate (tonnes/hour) by (g,t,f)
    mod.DispatchNOx = Expression(
        mod.GEN_TP_FUELS,
        rule=DispatchNOx_rule,
        doc="NOx emission rate (mass per hour) from each generator, fuel, and timepoint.",
    )

    # Annual NOx aggregation by period (tonnes/year)
    def annual_nox_rule(m, period):
        return sum(
            m.DispatchNOx[g, t, f] * m.tp_weight_in_year[t]
            for (g, t, f) in m.GEN_TP_FUELS
            if m.tp_period[t] == period
        )

    mod.AnnualNOx = Expression(
        mod.PERIODS,
        rule=annual_nox_rule,
        doc="Total NOx emissions (in base mass units per year) aggregated over all generators and fuels.",
    )    

    mod.GenVariableOMCostsInTP = Expression(
        mod.TIMEPOINTS,
        rule=lambda m, t: sum(
            m.DispatchGen[g, t] * m.gen_variable_om[g]
            for g in m.GENS_IN_PERIOD[m.tp_period[t]]
        ),
        doc="Summarize costs for the objective function",
    )
    mod.Cost_Components_Per_TP.append("GenVariableOMCostsInTP")


def load_inputs(mod, switch_data, inputs_dir):
    """

    Import project-specific data from an input directory.

    variable_capacity_factors can be skipped if no variable
    renewable projects are considered in the optimization.

    variable_capacity_factors.csv
        GENERATION_PROJECT, timepoint, gen_max_capacity_factor

    """

    switch_data.load_aug(
        optional=True,
        filename=os.path.join(inputs_dir, "variable_capacity_factors.csv"),
        index=mod.VARIABLE_GEN_TPS_RAW,
        param=(mod.gen_max_capacity_factor,),
    )

    # Loads per-generator PM2.5 and NOx override from inputs/gen_emission_costs.csv.
    # Expect a column named 'gen_pm25_intensity' and 'gen_NOx_intensity_ton_per_MMBtu
    # both in (tonnes/MMBtu) keyed by GENERATION_PROJECT.
    switch_data.load_aug(
        optional=True,
        filename=os.path.join(inputs_dir, "gen_emission_costs.csv"),
        select=("GENERATION_PROJECT", "gen_pm25_intensity_ton_per_MMBtu","gen_NOx_intensity_ton_per_MMBtu"),
        index=mod.GENERATION_PROJECTS,
        param=(mod.gen_pm25_intensity,mod.gen_nox_intensity),
    )


def post_solve(instance, outdir):
    """
    Exported files:

    dispatch_wide.csv - Dispatch results timepoints in "wide" format with
    timepoints as rows, generation projects as columns, and dispatch level
    as values

    dispatch.csv - Dispatch results in normalized form where each row
    describes the dispatch of a generation project in one timepoint.

    dispatch_annual_summary.csv - Similar to dispatch.csv, but summarized
    by generation technology and period.

    dispatch_gen_annual_summary.csv - Similar to dispatch_annual_summary.csv
    but broken out by generation project. (Was called
    gen_project_annual_summary.csv prior to 2.0.9.)

    dispatch_zonal_annual_summary.csv - Similar to dispatch_annual_summary.csv
    but broken out by load zone.

    dispatch_annual_summary.pdf - A figure of annual summary data. Only written
    if the ggplot python library is installed.
    """

    gen_proj = list(instance.GENERATION_PROJECTS)  # native order
    if instance.options.sorted_output:
        gen_proj.sort()

    write_table(
        instance,
        instance.TIMEPOINTS,
        output_file=os.path.join(outdir, "dispatch_wide.csv"),
        headings=("timestamp",) + tuple(gen_proj),
        values=lambda m, t: (m.tp_timestamp[t],)
        + tuple(m.DispatchGen[p, t] if (p, t) in m.GEN_TPS else 0.0 for p in gen_proj),
    )

    dispatch_normalized_dat = []
    for g, t in instance.GEN_TPS:
        p = instance.tp_period[t]
        record = {
            "generation_project": g,
            "gen_dbid": instance.gen_dbid[g],
            "gen_tech": instance.gen_tech[g],
            "gen_load_zone": instance.gen_load_zone[g],
            "gen_energy_source": instance.gen_energy_source[g],
            "timestamp": instance.tp_timestamp[t],
            "tp_weight_in_year_hrs": instance.tp_weight_in_year[t],
            "period": instance.tp_period[t],
            "DispatchGen_MW": value(instance.DispatchGen[g, t]),
            "Curtailment_MW": value(instance.DispatchUpperLimit[g, t])
            - value(instance.DispatchGen[g, t]),
            "Energy_GWh_typical_yr": value(
                instance.DispatchGen[g, t] * instance.tp_weight_in_year[t] / 1000
            ),
            "VariableCost_per_yr": value(
                instance.DispatchGen[g, t]
                * instance.gen_variable_om[g]
                * instance.tp_weight_in_year[t]
            ),
            "DispatchEmissions_tCO2_per_typical_yr": (
                value(
                    sum(
                        instance.DispatchEmissions[g, t, f]
                        * instance.tp_weight_in_year[t]
                        for f in instance.FUELS_FOR_GEN[g]
                    )
                )
                if instance.gen_uses_fuel[g]
                else 0
            ),
            "DispatchPM25_ton_per_hr": (  # values are in tonnes (not grams), consistent with PM2.5 intensity units in tonnes/MMBtu.
                value(
                    sum(
                        instance.DispatchPM25[g, t, f]
                        for f in instance.FUELS_FOR_GEN[g]
                    )
                )
                if instance.gen_uses_fuel[g]
                else 0.0
            ),
            # "DispatchPM25_g_per_typical_yr": (
            "DispatchPM25_ton_per_typical_yr": (  # values are in tonnes (not grams), consistent with PM2.5 intensity units in tonnes/MMBtu.
                value(
                    sum(
                        instance.DispatchPM25[g, t, f] * instance.tp_weight_in_year[t]
                        for f in instance.FUELS_FOR_GEN[g]
                    )
                )
                if instance.gen_uses_fuel[g]
                else 0.0
            ),
            "DispatchNOx_ton_per_hr": (  # values are in tonnes (not grams), consistent with NOx intensity units in tonnes/MMBtu.
                value(
                    sum(
                        instance.DispatchNOx[g, t, f]
                        for f in instance.FUELS_FOR_GEN[g]
                    )
                )
                if instance.gen_uses_fuel[g]
                else 0.0
            ),
            "DispatchNOx_ton_per_typical_yr": (  # values are in tonnes (not grams), consistent with NOx intensity units in tonnes/MMBtu.
                value(
                    sum(
                        instance.DispatchNOx[g, t, f] * instance.tp_weight_in_year[t]
                        for f in instance.FUELS_FOR_GEN[g]
                    )
                )
                if instance.gen_uses_fuel[g]
                else 0.0
            ),
            "GenCapacity_MW": value(instance.GenCapacity[g, p]),
            "GenCapitalCosts": value(instance.GenCapitalCosts[g, p]),
            "GenFixedOMCosts": value(instance.GenFixedOMCosts[g, p]),
        }
        try:
            try:
                record["ChargeStorage_MW"] = -1.0 * value(instance.ChargeStorage[g, t])
                record["Store_GWh_typical_yr"] = value(
                    instance.ChargeStorage[g, t] * instance.tp_weight_in_year[t] / 1000
                )
                record["Discharge_GWh_typical_yr"] = record["Energy_GWh_typical_yr"]
                record["Energy_GWh_typical_yr"] -= record["Store_GWh_typical_yr"]
                record["is_storage"] = True
            except KeyError:
                record["ChargeStorage_MW"] = float("NaN")
                record["Store_GWh_typical_yr"] = float("NaN")
                record["Discharge_GWh_typical_yr"] = float("NaN")
                record["is_storage"] = False
        except AttributeError:
            pass
        dispatch_normalized_dat.append(record)
    dispatch_full_df = pd.DataFrame(dispatch_normalized_dat)
    dispatch_full_df.set_index(["generation_project", "timestamp"], inplace=True)
    if instance.options.sorted_output:
        dispatch_full_df.sort_index(inplace=True)
    dispatch_full_df.to_csv(os.path.join(outdir, "dispatch.csv"))

    summary_columns = [
        "Energy_GWh_typical_yr",
        "VariableCost_per_yr",
        "DispatchEmissions_tCO2_per_typical_yr",
        "DispatchPM25_ton_per_typical_yr",
        "DispatchNOx_ton_per_typical_yr",
        "GenCapacity_MW",
        "GenCapitalCosts",
        "GenFixedOMCosts",
        "LCOE_dollar_per_MWh",
        "capacity_factor",
    ]
    if "ChargeStorage" in dir(instance):
        summary_columns.extend(["Store_GWh_typical_yr", "Discharge_GWh_typical_yr"])

    # Annual summary of each generator
    gen_sum = dispatch_full_df.groupby(
        [
            "generation_project",
            "gen_dbid",
            "gen_tech",
            "gen_load_zone",
            "gen_energy_source",
            "period",
            "GenCapacity_MW",
            "GenCapitalCosts",
            "GenFixedOMCosts",
        ]
    ).agg(
        lambda x: x.sum(min_count=1, skipna=False)
    )  # why these arguments?
    gen_sum.reset_index(inplace=True)
    gen_sum.set_index(
        inplace=True,
        keys=[
            "generation_project",
            "gen_dbid",
            "gen_tech",
            "gen_load_zone",
            "gen_energy_source",
            "period",
        ],
    )
    gen_sum["Energy_out_avg_MW"] = (
        gen_sum["Energy_GWh_typical_yr"] * 1000 / gen_sum["tp_weight_in_year_hrs"]
    )
    hrs_per_yr = gen_sum.iloc[0]["tp_weight_in_year_hrs"]
    try:
        idx = gen_sum["is_storage"].astype(bool)
        gen_sum.loc[idx, "Energy_out_avg_MW"] = (
            gen_sum.loc[idx, "Discharge_GWh_typical_yr"]
            * 1000
            / gen_sum.loc[idx, "tp_weight_in_year_hrs"]
        )
    except KeyError:
        pass

    def add_cap_factor_and_lcoe(df):
        df["capacity_factor"] = df["Energy_out_avg_MW"] / df["GenCapacity_MW"]
        no_cap = df["GenCapacity_MW"] == 0
        df.loc[no_cap, "capacity_factor"] = 0

        df["LCOE_dollar_per_MWh"] = (
            df["GenCapitalCosts"] + df["GenFixedOMCosts"] + df["VariableCost_per_yr"]
        ) / (df["Energy_out_avg_MW"] * hrs_per_yr)
        no_energy = df["Energy_out_avg_MW"] == 0
        df.loc[no_energy, "LCOE_dollar_per_MWh"] = 0

        return df

    gen_sum = add_cap_factor_and_lcoe(gen_sum)
    gen_sum.to_csv(
        os.path.join(outdir, "dispatch_gen_annual_summary.csv"), columns=summary_columns
    )

    zone_sum = gen_sum.groupby(
        ["gen_tech", "gen_load_zone", "gen_energy_source", "period"]
    ).sum()
    zone_sum = add_cap_factor_and_lcoe(zone_sum)
    zone_sum.to_csv(
        os.path.join(outdir, "dispatch_zonal_annual_summary.csv"),
        columns=summary_columns,
    )

    annual_summary = zone_sum.groupby(["gen_tech", "gen_energy_source", "period"]).sum()
    annual_summary = add_cap_factor_and_lcoe(annual_summary)
    annual_summary.to_csv(
        os.path.join(outdir, "dispatch_annual_summary.csv"), columns=summary_columns
    )

    import warnings

    with warnings.catch_warnings():
        # suppress warnings during import and use of plotnine
        warnings.simplefilter("ignore")
        try:
            import plotnine as p9
            import matplotlib
        except ImportError:
            pass
        else:
            # plotnine and matplotlib were imported successfully
            # Tell matplotlib to use a non-interactive backend so it won't add
            # an icon to the taskbar/dock while creating the plots
            # see https://matplotlib.org/stable/users/explain/backends.html
            matplotlib.use("pdf")
            plots = [
                ("gen_energy_source", "dispatch_annual_summary_fuel.pdf"),
                ("gen_tech", "dispatch_annual_summary_tech.pdf"),
            ]
            for y, outfile in plots:
                annual_summary_plot = (
                    p9.ggplot(
                        annual_summary.reset_index(),
                        p9.aes(
                            x="period",
                            weight="Energy_GWh_typical_yr",
                            fill="factor({})".format(y),
                        ),
                    )
                    + p9.geom_bar(position="stack")
                    + p9.scale_y_continuous(name="Energy (GWh/yr)")
                    + p9.theme_bw()
                )
                plot_dir = os.path.join(outdir, "plots")
                os.makedirs(plot_dir, exist_ok=True)
                annual_summary_plot.save(filename=os.path.join(plot_dir, outfile))
                annual_summary_plot.save(
                    filename=os.path.join(outdir, "plots", outfile)
                )


# all plotting is now done in post processing in switch_model/plot_switch_results.py
# # ==========================================================
# # ==========================================================
# #                ****** PLOTTING ******
# # ==========================================================
# # ==========================================================

# # 1) One dispatch matrix for both single & multi-scenario
# @graph(
#     name = "dispatch_matrix",
#     title ="Average daily dispatch",
#     supports_multi_scenario = True,
#     is_long = True
# )
# def dispatch_matrix(tools):
#     df = tools.get_dataframe("dispatch.csv")
#     df["DispatchGen_MW"] = df["DispatchGen_MW"] / 1e3  # MW→GW

#     if tools.num_scenarios > 1:
#         tools.graph_scenario_matrix(
#             df,
#             value_column="DispatchGen_MW",
#             ylabel="Average daily dispatch (GW)"
#         )
#     else:
#         tools.graph_time_matrix(
#             df,
#             value_column="DispatchGen_MW",
#             ylabel="Average daily dispatch (GW)"
#         )


# # 2) One curtailment matrix for both single & multi-scenario
# @graph(
#     name="curtailment_matrix",
#     title="Average daily curtailment",
#     supports_multi_scenario = True,
#     is_long = True
# )
# def curtailment_matrix(tools):
#     df = tools.get_dataframe("dispatch.csv")
#     df = df[df["gen_energy_source"].isin(["Solar", "Wind"])].copy()

#     # Be safe about dtype (sometimes read as str)
#     df["Curtailment_MW"] = tools.pd.to_numeric(df["Curtailment_MW"], errors="coerce").fillna(0.0)

#     # MW→GW and zero-out tiny negatives from numerical noise
#     df["Curtailment_MW"] = (df["Curtailment_MW"] / 1e3).where(lambda s: s >= 0, 0)

#     if tools.num_scenarios > 1:
#         tools.graph_scenario_matrix(df, value_column="Curtailment_MW",
#                                     ylabel="Average daily curtailment (GW)")
#     else:
#         tools.graph_time_matrix(df, value_column="Curtailment_MW",
#                                 ylabel="Average daily curtailment (GW)")

# # 3) Unify generation mix stacked bar
# @graph(
#     name="generation_mix_stacked_bar",
#     title="Generation mix by period",
#     supports_multi_scenario=True
# )
# def generation_mix_stacked_bar(tools, use_gen_type=True, bucket_small=True, bucket_threshold=0.005, to_twh=True):
#     # pick source
#     src = "dispatch_annual_summary.csv" if use_gen_type else "dispatch_gen_annual_summary.csv"
#     df = tools.get_dataframe(src)

#     if use_gen_type:
#         df = tools.transform.gen_type(df)
#         category_col = "gen_type"
#     else:
#         category_col = "gen_tech"

#     value = "Energy_GWh_typical_yr"
#     groupby = ["period", category_col] if tools.num_scenarios == 1 else ["period", "scenario_name", category_col]
#     agg = df.groupby(groupby, as_index=False)[value].sum()

#     idx = ["period"] if tools.num_scenarios == 1 else ["period", "scenario_name"]
#     mat = agg.pivot_table(index=idx, columns=category_col, values=value, aggfunc="sum").fillna(0)

#     if to_twh:
#         mat = mat * 1e-3

#     if bucket_small:
#         cutoff = mat.sum(axis=1) * bucket_threshold
#         below = mat.lt(cutoff, axis=0).all()
#         mat = (mat.T.groupby(lambda c: "Other" if below[c] else c).sum()).T

#     mat = mat.reindex(columns=mat.sum(axis=0).sort_values().index)
#     ax = tools.get_axes()
#     ylabel = "Energy (TWh)" if to_twh else "Energy (GWh)"
#     mat.plot(ax=ax, kind="bar", stacked=True, xlabel="Period", ylabel=ylabel)
#     tools.bar_label()

# @graph(
#     name="curtailment_rate_by_period",
#     title="Percent of total dispatchable capacity curtailed",
#     is_long=True
# )
# def graph_curtailment_per_tech(tools):
#     # Load dispatch.csv
#     df = tools.get_dataframe('dispatch.csv')
#     df = tools.transform.gen_type(df)
#     df["Total"] = df['DispatchGen_MW'] + df["Curtailment_MW"]
#     df['is_renewable'] = df['gen_energy_source'].isin(['Solar', 'Wind'])
#     df = df[df["is_renewable"]]
#     # Make PERIOD a category to ensure x-axis labels don't fill in years between period
#     # TODO we should order this by period here to ensure they're in increasing order
#     df["period"] = df["period"].astype("category")
#     df = df.groupby(["period", "gen_type"], as_index=False, observed=False).sum(numeric_only=True)
#     df["Percent Curtailed"] = df["Curtailment_MW"] / (df['DispatchGen_MW'] + df["Curtailment_MW"])
#     df = df.pivot(index="period", columns="gen_type", values="Percent Curtailed").fillna(0)
#     if len(df) == 0:  # No dispatch from renewable technologies
#         return
#     # Set the name of the legend.
#     df = df.rename_axis("Type", axis='columns')
#     # Get axes to graph on
#     ax = tools.get_axes()
#     # Plot
#     color = tools.get_colors()
#     kwargs = dict() if color is None else dict(color=color)
#     df.plot(ax=ax, kind='line',  xlabel='Period', marker="x", **kwargs)

#     # Set the y-axis to use percent
#     ax.yaxis.set_major_formatter(tools.plt.ticker.PercentFormatter(1.0))
#     # Horizontal line at 100%
#     ax.axhline(y=1, linestyle="--", color='b')

# @graph("dispatch_map", title="Dispatched electricity per load zone")
# def dispatch_map(tools):
#     if not tools.maps.can_make_maps():
#         tools.logger.info("Skipping dispatch_map: mapping dependencies or files not found.")
#         return

#     dispatch = (
#         tools.get_dataframe("dispatch_zonal_annual_summary.csv")
#              .rename({"Energy_GWh_typical_yr": "value"}, axis=1)
#     )
#     dispatch = tools.transform.gen_type(dispatch)
#     dispatch = dispatch.groupby(["gen_type", "gen_load_zone"], as_index=False)["value"].sum()
#     dispatch["value"] *= 1e-3  # GWh → TWh

#     tools.maps.graph_pie_chart(
#         dispatch, bins=(0, 10, 100, 200, float("inf")), title="Yearly Dispatch (TWh)"
#     )


# @graph(
#     "generation_mix_by_period",
#     title="Generation mix by period (GWh)",
#     supports_multi_scenario=True
# )
# def graph_generation_mix_by_period(tools):
#     df = tools.get_dataframe("dispatch_gen_annual_summary.csv")

#     # Clean duplicate column names if present
#     cols = df.columns.tolist()
#     if cols.count("GenCapacity_MW") > 1:
#         # Drop all but the first occurrence
#         first = True
#         new_cols = []
#         for c in cols:
#             if c == "GenCapacity_MW":
#                 if first:
#                     new_cols.append(c)
#                     first = False
#                 else:
#                     new_cols.append("GenCapacity_MW_2")
#             else:
#                 new_cols.append(c)
#         df.columns = new_cols
#         if "GenCapacity_MW_2" in df.columns:
#             df = df.drop(columns=["GenCapacity_MW_2"])

#     # Aggregate to period × tech (and scenario if present)
#     keep_cols = ["period", "gen_tech", "Energy_GWh_typical_yr"]
#     keep_cols = [c for c in keep_cols if c in df.columns]
#     df = df[keep_cols + (["scenario_name"] if tools.num_scenarios > 1 else [])].copy()

#     groupby = ["period", "gen_tech"] if tools.num_scenarios == 1 else ["period", "scenario_name", "gen_tech"]
#     mix = df.groupby(groupby, as_index=False)["Energy_GWh_typical_yr"].sum()

#     # Pivot for stacked bars: index = period or (period, scenario), columns = gen_tech
#     idx = ["period"] if tools.num_scenarios == 1 else ["period", "scenario_name"]
#     mix_p = mix.pivot_table(index=idx, columns="gen_tech", values="Energy_GWh_typical_yr", aggfunc="sum").fillna(0)

#     # Order techs by total contribution
#     mix_p = mix_p.reindex(columns=mix_p.sum(axis=0).sort_values().index)

#     ax = tools.get_axes()
#     mix_p.plot(ax=ax, kind="bar", stacked=True, xlabel="Period", ylabel="Energy (GWh)")

# # @graph(
# #     "dispatch",
# #     title="Average daily dispatch",
# #     is_long=True
# # )
# # def graph_hourly_dispatch(tools):
# #     """
# #     Generates a matrix of hourly dispatch plots for each time region
# #     """
# #     # Read dispatch.csv
# #     df = tools.get_dataframe(filename=('dispatch.csv'))
# #     # Convert to GW
# #     df["DispatchGen_MW"] /= 1e3
# #     # Plot Dispatch
# #     tools.graph_time_matrix(
# #         df,
# #         value_column="DispatchGen_MW",
# #         ylabel="Average daily dispatch (GW)",
# #     )

# # @graph(
# #     "curtailment",
# #     title="Average daily curtailment",
# #     is_long=True
# # )
# # def graph_hourly_curtailment(tools):
# #     # Read dispatch.csv
# #     df = tools.get_dataframe('dispatch.csv')
# #     # Keep only renewable
# #     df['is_renewable'] = df['gen_energy_source'].isin(['Solar', 'Wind'])
# #     df = df[df["is_renewable"]]
# #     df["Curtailment_MW"] /= 1e3 # Convert to GW
# #     # Plot curtailment
# #     tools.graph_time_matrix(
# #         df,
# #         value_column="Curtailment_MW",
# #         ylabel="Average daily curtailment (GW)"
# #     )


# # @graph(
# #     "dispatch_per_scenario",
# #     title="Average daily dispatch",
# #     requires_multi_scenario=True,
# #     is_long=True,
# # )
# # def graph_hourly_dispatch(tools):
# #     """
# #     Generates a matrix of hourly dispatch plots for each time region
# #     """
# #     # Read dispatch.csv
# #     df = tools.get_dataframe('dispatch.csv')
# #     # Convert to GW
# #     df["DispatchGen_MW"] /= 1e3
# #     # Plot Dispatch
# #     tools.graph_scenario_matrix(
# #         df,
# #         value_column="DispatchGen_MW",
# #         ylabel="Average daily dispatch (GW)"
# #     )


# # @graph(
# #     "curtailment_compare_scenarios",
# #     title="Average daily curtailment by scenario",
# #     requires_multi_scenario=True,
# #     is_long=True,
# # )
# # def graph_hourly_curtailment(tools):
# #     # Read dispatch.csv
# #     df = tools.get_dataframe('dispatch.csv')
# #     # Keep only renewable
# #     df['is_renewable'] = df['gen_energy_source'].isin(['Solar', 'Wind'])
# #     df = df[df["is_renewable"]]
# #     df["Curtailment_MW"] /= 1e3  # Convert to GW
# #     tools.graph_scenario_matrix(
# #         df,
# #         value_column="Curtailment_MW",
# #         ylabel="Average daily curtailment (GW)"
# #     )


# # @graph(
# #     "total_dispatch",
# #     title="Total dispatched electricity",
# # )
# # def graph_total_dispatch(tools):
# #     # ---------------------------------- #
# #     # total_dispatch.png                 #
# #     # ---------------------------------- #
# #     # read dispatch_annual_summary.csv
# #     total_dispatch = tools.get_dataframe("dispatch_annual_summary.csv")
# #     # add type column
# #     total_dispatch = tools.transform.gen_type(total_dispatch)
# #     # aggregate and pivot
# #     # total_dispatch = total_dispatch.pivot_table(columns="gen_type", index="period", values="Energy_GWh_typical_yr",
# #     #                                             aggfunc=tools.np.sum)
# #     total_dispatch = total_dispatch.pivot_table(
# #         index="period",
# #         columns="gen_type",
# #         values="Energy_GWh_typical_yr",
# #         aggfunc="sum",
# #         observed=False,
# #     )
# #     # Convert values to TWh
# #     total_dispatch *= 1E-3

# #     # For generation types that make less than 2% in every period, group them under "Other"
# #     # ---------
# #     # sum the generation across the energy_sources for each period, 0.5% of that is the cutoff for that period
# #     cutoff_per_period = total_dispatch.sum(axis=1) * 0.005
# #     # Check for each technology if it's below the cutoff for every period
# #     is_below_cutoff = total_dispatch.lt(cutoff_per_period, axis=0).all()
# #     # groupby if the technology is below the cutoff
# #     # total_dispatch = total_dispatch.groupby(axis=1, by=lambda c: "Other" if is_below_cutoff[c] else c).sum()
# #     total_dispatch = (
# #         total_dispatch
# #         .T.groupby(by=lambda c: "Other" if is_below_cutoff[c] else c)
# #         .sum()
# #         .T
# #     )

# #     # Sort columns by the last period
# #     total_dispatch = total_dispatch.sort_values(by=total_dispatch.index[-1], axis=1)
# #     # Give proper name for legend
# #     total_dispatch = total_dispatch.rename_axis("Type", axis=1)
# #     # Get axis
# #     # Plot
# #     total_dispatch.plot(
# #         kind='bar',
# #         stacked=True,
# #         ax=tools.get_axes(),
# #         color=tools.get_colors(len(total_dispatch)),
# #         xlabel="Period",
# #         ylabel="Total dispatched electricity (TWh)"
# #     )

# #     tools.bar_label()

# # @graph(
# #     "energy_balance",
# #     title="Energy Balance For Every Month",
# #     supports_multi_scenario=True,
# #     is_long=True
# # )
# # def energy_balance(tools):
# #     # Get dispatch dataframe
# #     cols = ["timestamp", "gen_tech", "gen_energy_source", "DispatchGen_MW", "scenario_name", "scenario_index",
# #             "Curtailment_MW"]
# #     df = tools.get_dataframe("dispatch.csv", drop_scenario_info=False)[cols]
# #     df = tools.transform.gen_type(df)

# #     # Rename and add needed columns
# #     df["Dispatch Limit"] = df["DispatchGen_MW"] + df["Curtailment_MW"]
# #     df = df.drop("Curtailment_MW", axis=1)
# #     df = df.rename({"DispatchGen_MW": "Dispatch"}, axis=1)
# #     # Sum dispatch across all the projects of the same type and timepoint
# #     key_columns = ["timestamp", "gen_type", "scenario_name", "scenario_index"]
# #     df = df.groupby(key_columns, as_index=False).sum()
# #     df = df.melt(id_vars=key_columns, value_vars=["Dispatch", "Dispatch Limit"], var_name="Type")
# #     df = df.rename({"gen_type": "Source"}, axis=1)

# #     discharge = df[(df["Source"] == "Storage") & (df["Type"] == "Dispatch")].drop(["Source", "Type"], axis=1).rename(
# #         {"value": "discharge"}, axis=1)

# #     # Get load dataframe
# #     load = tools.get_dataframe("load_balance.csv", drop_scenario_info=False)
# #     # load = load.drop("normalized_energy_balance_duals_dollar_per_mwh", axis=1)

# #     # Sum load across all the load zones
# #     key_columns = ["timestamp", "scenario_name", "scenario_index"]
# #     load = load.groupby(key_columns, as_index=False).sum()

# #     # Subtract storage dispatch from generation and add it to the storage charge to get net flow
# #     load = load.merge(
# #         discharge,
# #         how="left",
# #         on=key_columns,
# #         validate="one_to_one"
# #     )
# #     # load["ZoneTotalCentralDispatch"] -= load["discharge"]
# #     # load["StorageNetCharge"] += load["discharge"]
# #     # load = load.drop("discharge", axis=1)

# #     # Rename and convert from wide to long format
# #     load = load.rename({
# #         "ZoneTotalCentralDispatch": "Total Generation (excl. storage discharge)",
# #         "TXPowerNet": "Transmission Losses",
# #         # "StorageNetCharge": "Storage Net Flow",
# #         "ZoneTotalCentralDispatch": "Demand",
# #     }, axis=1).sort_index(axis=1)
# #     load = load.melt(id_vars=key_columns, var_name="Source")
# #     load["Type"] = "Dispatch"

# #     # Merge dispatch contributions with load contributions
# #     df = pd.concat([load, df])

# #     # Add the timestamp information and make period string to ensure it doesn't mess up the graphing
# #     df = tools.transform.timestamp(df).astype({"period": str})
# #     # Ensure both columns are numeric
# #     df["value"] = pd.to_numeric(df["value"], errors="coerce")
# #     df["tp_duration"] = pd.to_numeric(df["tp_duration"], errors="coerce")
# #     # Convert to TWh (incl. multiply by timepoint duration)
# #     df["value"] *= df["tp_duration"] / 1e6

# #     FREQUENCY = "1W"

# #     def groupby_time(df):
# #         return df.groupby([
# #             "scenario_name",
# #             "period",
# #             "Source",
# #             "Type",
# #             tools.pd.Grouper(key="datetime", freq=FREQUENCY, origin="start")
# #         ])["value"]

# #     df = groupby_time(df).sum().reset_index()

# #     # Get the state of charge data
# #     # soc = tools.get_dataframe("StateOfCharge.csv", dtype={"STORAGE_GEN_TPS_1": str}, drop_scenario_info=False)
# #     # soc = soc.rename({"STORAGE_GEN_TPS_2": "timepoint", "StateOfCharge": "value"}, axis=1)
# #     # # Sum over all the projects that are in the same scenario with the same timepoint
# #     # soc = soc.groupby(["timepoint", "scenario_name"], as_index=False).sum()
# #     # soc["Source"] = "State Of Charge"
# #     # soc["value"] /= 1e6  # Convert to TWh

# #     # # Group by time
# #     # soc = tools.transform.timestamp(soc, use_timepoint=True, key_col="timepoint").astype({"period": str})
# #     # soc["Type"] = "Dispatch"
# #     # soc = groupby_time(soc).mean().reset_index()

# #     # # Add state of charge to dataframe
# #     # df = pd.concat([df, soc])
# #     # Add column for day since that's what we really care about
# #     df["day"] = df["datetime"].dt.dayofyear

# #     # Plot
# #     # Get the colors for the lines
# #     colors = tools.get_colors()
# #     # colors.update({
# #     #     "Transmission Losses": "brown",
# #     #     # "Storage Net Flow": "cadetblue",
# #     #     "Demand": "black",
# #     #     "Total Generation (excl. storage discharge)": "black",
# #     #     "State Of Charge": "green"
# #     # })

# #     # plot
# #     num_periods = df["period"].nunique()
# #     pn = tools.pn
# #     plot = pn.ggplot(df) + \
# #            pn.geom_line(pn.aes(x="day", y="value", color="Source", linetype="Type")) + \
# #            pn.facet_grid("period ~ scenario_name") + \
# #            pn.labs(y="Contribution to Energy Balance (TWh)") + \
# #            pn.scales.scale_x_continuous(
# #                name="Month",
# #                labels=["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"],
# #                breaks=(15, 46, 76, 106, 137, 167, 198, 228, 259, 289, 319, 350),
# #                limits=(0, 366)) + \
# #            pn.scales.scale_linetype_manual(
# #                values={"Dispatch Limit": "dotted", "Dispatch": "solid"}
# #            ) + \
# #            pn.theme(
# #                figure_size=(pn.options.figure_size[0] * tools.num_scenarios, pn.options.figure_size[1] * num_periods))

# #     tools.save_figure(plot.draw())

# #         #    pn.scales.scale_color_manual(values=colors, aesthetics="color", na_value=colors["Other"]) + \
# # @graph(
# #     "curtailment_per_period",
# #     title="Percent of total dispatchable capacity curtailed",
# #     is_long=True
# # )
# # def graph_curtailment_per_tech(tools):
# #     # Load dispatch.csv
# #     df = tools.get_dataframe('dispatch.csv')
# #     df = tools.transform.gen_type(df)
# #     df["Total"] = df['DispatchGen_MW'] + df["Curtailment_MW"]
# #     df['is_renewable'] = df['gen_energy_source'].isin(['Solar', 'Wind'])
# #     df = df[df["is_renewable"]]
# #     # Make PERIOD a category to ensure x-axis labels don't fill in years between period
# #     # TODO we should order this by period here to ensure they're in increasing order
# #     df["period"] = df["period"].astype("category")
# #     df = df.groupby(["period", "gen_type"], as_index=False, observed=False).sum(numeric_only=True)
# #     df["Percent Curtailed"] = df["Curtailment_MW"] / (df['DispatchGen_MW'] + df["Curtailment_MW"])
# #     df = df.pivot(index="period", columns="gen_type", values="Percent Curtailed").fillna(0)
# #     if len(df) == 0:  # No dispatch from renewable technologies
# #         return
# #     # Set the name of the legend.
# #     df = df.rename_axis("Type", axis='columns')
# #     # Get axes to graph on
# #     ax = tools.get_axes()
# #     # Plot
# #     color = tools.get_colors()
# #     kwargs = dict() if color is None else dict(color=color)
# #     df.plot(ax=ax, kind='line',  xlabel='Period', marker="x", **kwargs)

# #     # Set the y-axis to use percent
# #     ax.yaxis.set_major_formatter(tools.plt.ticker.PercentFormatter(1.0))
# #     # Horizontal line at 100%
# #     ax.axhline(y=1, linestyle="--", color='b')


# # # @graph(
# # #     "energy_balance_2",
# # #     title="Balance between demand, generation and storage for last period",
# # #     note="Dashed green and red lines are total generation and total demand (incl. transmission losses),"
# # #          " respectively.\nDotted line is the total state of charge (scaled for readability)."
# # #          "\nWe used a 14-day rolling mean to smoothen out values.",
# # #     supports_multi_scenario=True
# # # )
# # def graph_energy_balance_2(tools):
# #     # Get dispatch dataframe
# #     dispatch = tools.get_dataframe("dispatch.csv", usecols=[
# #         "timestamp", "gen_tech", "gen_energy_source", "DispatchGen_MW", "scenario_name"
# #     ]).rename({"DispatchGen_MW": "value"}, axis=1)
# #     print(dispatch.head())
# #     dispatch = tools.transform.gen_type(dispatch)
# #     print(dispatch.head())
# #     # Sum dispatch across all the projects of the same type and timepoint
# #     dispatch = dispatch.groupby(["timestamp", "gen_type"], as_index=False).sum()
# #     dispatch = dispatch[dispatch["gen_type"] != "Storage"]
# #     print(dispatch.head())
# #     # Get load dataframe
# #     load = tools.get_dataframe("load_balance.csv", usecols=[
# #         "timestamp", "ZoneTotalCentralDispatch", "TXPowerNet", "scenario_name"
# #     ])
# #     print(load.head())
# #     def process_time(df):
# #         df = df.astype({"period": int})
# #         df = df[df["period"] == df["period"].max()].drop(columns="period")
# #         return df.set_index("datetime")

# #     # Sum load across all the load zones
# #     load = load.groupby(["timestamp"], as_index=False).sum()
# #     print(load.head())
# #     # Include Tx Losses in demand and flip sign
# #     load["value"] = (load["ZoneTotalCentralDispatch"] + load["TXPowerNet"]) * -1
# #     print(load.head())
# #     # Rename and convert from wide to long format
# #     load = load[["timestamp", "value"]]

# #     # Add the timestamp information and make period string to ensure it doesn't mess up the graphing
# #     dispatch = process_time(tools.transform.timestamp(dispatch))
# #     print(dispatch.head())
# #     load = process_time(tools.transform.timestamp(load))
# #     print(load.head())
# #     # Convert to TWh (incl. multiply by timepoint duration)
# #     dispatch["value"] *= dispatch["tp_duration"] / 1e6
# #     load["value"] *= load["tp_duration"] / 1e6


# #     def rolling_sum(df):
# #         days = 14
# #         freq = str(days) + "D"
# #         offset = tools.pd.Timedelta(freq) / 2
# #         print('#1',df.head())
# #         df = df.rolling(freq, center=True).value.sum().reset_index()
# #         print('#2',df.head())
# #         df["value"] /= days
# #         print('#2.1',df["value"])
# #         df = df[(df.datetime.min() + offset < df.datetime) & (df.datetime < df.datetime.max() - offset)]
# #         print('#3',df.head())
# #         return df

# #     dispatch = rolling_sum(dispatch.groupby("gen_type", as_index=False))
# #     print(dispatch.head())
# #     load = rolling_sum(load).set_index("datetime")["value"]

# #     # # Get the state of charge data
# #     # soc = tools.get_dataframe("StateOfCharge.csv", dtype={"STORAGE_GEN_TPS_1": str}) \
# #     #     .rename(columns={"STORAGE_GEN_TPS_2": "timepoint", "StateOfCharge": "value"})
# #     # # Sum over all the projects that are in the same scenario with the same timepoint
# #     # soc = soc.groupby(["timepoint"], as_index=False).sum()
# #     # soc["value"] /= 1e6  # Convert to TWh
# #     # max_soc = soc["value"].max()

# #     # # Group by time
# #     # soc = process_time(tools.transform.timestamp(soc, use_timepoint=True, key_col="timepoint"))
# #     # soc = soc.rolling(freq, center=True)["value"].mean().reset_index()
# #     # soc = soc[(soc.datetime.min() + offset < soc.datetime) & (soc.datetime < soc.datetime.max() - offset)]
# #     # soc = soc.set_index("datetime")["value"]


# #     dispatch = dispatch[dispatch["value"] != 0]
# #     print(dispatch.head())
# #     dispatch = dispatch.pivot(columns="gen_type", index="datetime", values="value")
# #     # print(dispatch.head())
# #     dispatch = dispatch[dispatch.std().sort_values().index].rename_axis("Technology", axis=1)
# #     # print(dispatch.head())
# #     total_dispatch = dispatch.sum(axis=1)
# #     # print(total_dispatch.head())
# #     max_val = max(total_dispatch.max(), load.max())

# #     # Scale soc to the graph
# #     # soc *= max_val / max_soc

# #     # Plot
# #     # Get the colors for the lines
# #     # plot
# #     ax = tools.get_axes(ylabel="Average Daily Generation (TWh)")
# #     print(max_val)
# #     ax.set_ylim(0, max_val * 1.05)
# #     dispatch.plot(
# #         ax=ax,
# #         color=tools.get_colors()
# #     )
# #     # soc.plot(ax=ax, color="black", linestyle="dotted")
# #     load.plot(ax=ax, color="red", linestyle="dashed")
# #     total_dispatch.plot(ax=ax, color="green", linestyle="dashed")
# #     ax.fill_between(total_dispatch.index, total_dispatch.values, load.values, alpha=0.2, where=load<total_dispatch, facecolor="green")
# #     ax.fill_between(total_dispatch.index, total_dispatch.values, load.values, alpha=0.2, where=load>total_dispatch, facecolor="red")
