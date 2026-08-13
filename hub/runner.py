"""Pyodide runner module for clintrials simulations in the browser web worker."""

import dataclasses
import json
import sys
from typing import Callable, get_args, get_origin

import pandas as pd

from clintrials.core.schema import (
    CRMSchema,
    EffToxSchema,
    GroupSequentialDesignSchema,
    WagesTaitSchema,
    WATUSchema,
    WinRatioSchema,
)
from clintrials.core.simulation import extract_sim_data
from clintrials.dosefinding import simulate_dose_finding_trial

# Import designs
from clintrials.dosefinding.crm import CRM
from clintrials.dosefinding.efficacytoxicity import simulate_trial
from clintrials.dosefinding.efftox import EffTox, LpNormCurve
from clintrials.dosefinding.wagestait import WagesTait
from clintrials.dosefinding.watu import WATU
from clintrials.phase3.gsd import (
    GroupSequentialDesign,
    spending_function_obrien_fleming,
)
from clintrials.utils import ParameterSpace
from clintrials.visualization.provider import get_default_provider
from clintrials.winratio.main import WinRatioTrial

schema_classes = {
    "CRMSchema": CRMSchema,
    "EffToxSchema": EffToxSchema,
    "GroupSequentialDesignSchema": GroupSequentialDesignSchema,
    "WagesTaitSchema": WagesTaitSchema,
    "WATUSchema": WATUSchema,
    "WinRatioSchema": WinRatioSchema,
}


def validate_fields(schema_name: str, payload: dict) -> dict[str, str]:
    """Validate the provided payload against the specified schema class constraints."""
    schema_class = schema_classes.get(schema_name)
    if not schema_class:
        return {"__all__": f"Unknown schema: {schema_name}"}

    errors = {}

    # 1. Validate each individual field's type/bounds/constraints
    for name, f in schema_class.model_fields.items():
        if name not in payload or payload[name] is None:
            # If it's missing, is it required?
            if f.default is dataclasses.MISSING:
                # check if Optional
                origin = get_origin(f.annotation)
                is_optional = False
                if origin is getattr(sys.modules.get("typing"), "Union", None):
                    if type(None) in get_args(f.annotation):
                        is_optional = True
                if not is_optional:
                    errors[name] = f"Field '{name}' is required."
            continue

        val = payload[name]

        try:
            # Call the validation logic of BaseModel on a dummy instance of schema_class
            dummy = schema_class.__new__(schema_class)
            dummy._validate_value(name, val, f)
        except ValueError as e:
            errors[name] = str(e)

    # 2. If individual fields are valid, try instantiating the schema_class
    # to run cross-field constraints in __post_init__!
    if not errors:
        try:
            # Prepare arguments. Note that Python dataclass will supply defaults
            # for missing fields that have them.
            args = {}
            for name, f in schema_class.model_fields.items():
                if name in payload and payload[name] is not None:
                    args[name] = payload[name]
            schema_class(**args)
        except ValueError as e:
            msg = str(e)
            # Find which field this post-init validation error belongs to based on keywords in the message.
            matched = False
            for name in schema_class.model_fields.keys():
                if name in msg:
                    errors[name] = msg
                    matched = True
                    break
            if not matched:
                # fallback keyword mappings for specific post-init checks
                if "beta_T" in msg or "priors" in msg:
                    errors["theta_priors"] = msg
                elif "skeletons" in msg:
                    errors["skeletons"] = msg
                elif "tox_target" in msg or "tox_limit" in msg:
                    errors["tox_target"] = msg
                elif "stage_one_size" in msg:
                    errors["stage_one_size"] = msg
                else:
                    # if no specific field name, put it on some field or a general key
                    errors["__all__"] = msg

    return errors


def run_simulation_py(
    schema_name: str,
    payload_json: str,
    progress_callback: Callable[[int], None],
    accessibility_mode: bool = False,
) -> str:
    """Run a clinical trial simulation from Pyodide in the browser."""
    payload = json.loads(payload_json)

    # Run the validation guard
    errors = validate_fields(schema_name, payload)
    if errors:
        return json.dumps({"status": "validation_error", "errors": errors})

    summary_df = None
    figures = []

    if schema_name == "CRMSchema":
        prior = payload.get("prior")
        target = payload.get("target")
        first_dose = payload.get("first_dose", 1)
        max_size = payload.get("max_size", 30)
        lowest_dose_too_toxic_hurdle = payload.get("lowest_dose_too_toxic_hurdle", 0.0)
        lowest_dose_too_toxic_certainty = payload.get(
            "lowest_dose_too_toxic_certainty", 0.0
        )
        coherency_threshold = payload.get("coherency_threshold", 0.0)
        bootstrap_samples = payload.get("bootstrap_samples", 200)

        crm = CRM(
            prior=prior,
            target=target,
            first_dose=first_dose,
            max_size=max_size,
            lowest_dose_too_toxic_hurdle=lowest_dose_too_toxic_hurdle,
            lowest_dose_too_toxic_certainty=lowest_dose_too_toxic_certainty,
            coherency_threshold=coherency_threshold,
            bootstrap_samples=bootstrap_samples,
            min_beta=payload.get("min_beta"),
            max_beta=payload.get("max_beta"),
            n_points=payload.get("n_points"),
            sample_size=payload.get("sample_size"),
        )

        # Scenarios: Scenario 1 is the prior, Scenario 2 is scaled
        scenarios = [tuple(prior), tuple(min(1.0, p * 1.5) for p in prior)]

        n_replicates = 20
        total_sims = len(scenarios) * n_replicates
        current_sim_count = 0
        sims = []

        for true_tox in scenarios:
            for rep in range(n_replicates):
                report = simulate_dose_finding_trial(
                    crm, true_toxicities=true_tox, cohort_size=3
                )
                report["true_tox"] = true_tox
                sims.append(report)
                current_sim_count += 1
                progress_callback(int((current_sim_count / total_sims) * 100))

        func_map = CRM.get_summary_functions()
        ps = ParameterSpace()
        ps.add("true_tox", scenarios)
        summary_df = extract_sim_data(sims, ps, func_map, return_type="dataframe")

        fig = get_default_provider().plot_crm_simulation_recommendation(
            summary_df, high_contrast=False
        )
        figures.append(("Dose Recommendation Probability", fig))

    elif schema_name == "EffToxSchema":
        real_doses = payload.get("real_doses")
        num_doses = len(real_doses)
        prior_tox_probs = (
            payload.get("prior_tox_probs") or [0.05, 0.1, 0.2, 0.3, 0.4][:num_doses]
        )
        prior_eff_probs = (
            payload.get("prior_eff_probs") or [0.2, 0.4, 0.6, 0.7, 0.8][:num_doses]
        )

        if len(prior_tox_probs) < num_doses:
            prior_tox_probs += [0.4] * (num_doses - len(prior_tox_probs))
        if len(prior_eff_probs) < num_doses:
            prior_eff_probs += [0.8] * (num_doses - len(prior_eff_probs))

        metric = LpNormCurve(0.2, 0.4, 0.5, 0.2)
        trial = EffTox(
            real_doses=real_doses,
            prior_tox_probs=prior_tox_probs,
            prior_eff_probs=prior_eff_probs,
            tox_cutoff=payload.get("tox_cutoff", 0.4),
            eff_cutoff=payload.get("eff_cutoff", 0.2),
            tox_certainty=payload.get("tox_certainty", 0.8),
            eff_certainty=payload.get("eff_certainty", 0.8),
            metric=metric,
            max_size=payload.get("max_size", 30),
            first_dose=payload.get("first_dose", 1),
        )

        tox_scenarios = [tuple(prior_tox_probs)]
        eff_scenarios = [tuple(prior_eff_probs)]

        n_replicates = 10
        total_sims = len(tox_scenarios) * len(eff_scenarios) * n_replicates
        current_sim_count = 0
        sims = []

        for true_prob_tox in tox_scenarios:
            for true_prob_eff in eff_scenarios:
                for rep in range(n_replicates):
                    report = simulate_trial(
                        trial,
                        true_toxicities=true_prob_tox,
                        true_efficacies=true_prob_eff,
                        cohort_size=3,
                    )
                    report["true_prob_tox"] = true_prob_tox
                    report["true_prob_eff"] = true_prob_eff
                    sims.append(report)
                    current_sim_count += 1
                    progress_callback(int((current_sim_count / total_sims) * 100))

        func_map = EffTox.get_summary_functions()
        ps = ParameterSpace()
        ps.add("true_prob_tox", tox_scenarios)
        ps.add("true_prob_eff", eff_scenarios)
        summary_df = extract_sim_data(sims, ps, func_map, return_type="dataframe")

        fig_rec = get_default_provider().plot_bivariate_simulation_recommendation(
            summary_df, high_contrast=False
        )
        fig_accept = get_default_provider().plot_efftox_simulation_acceptability(
            summary_df, high_contrast=False
        )
        figures.append(("Dose Recommendation Probability", fig_rec))
        figures.append(("Acceptability Probabilities", fig_accept))

    elif schema_name == "WATUSchema":
        skeletons = payload.get("skeletons") or [
            [0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            [0.5, 0.6, 0.5, 0.4, 0.3, 0.2],
            [0.4, 0.5, 0.6, 0.5, 0.4, 0.3],
        ]
        prior_tox_probs = payload.get("prior_tox_probs") or [
            0.01,
            0.08,
            0.15,
            0.22,
            0.29,
            0.36,
        ]
        metric = LpNormCurve(0.2, 0.4, 0.5, 0.2)

        trial = WATU(
            skeletons=skeletons,
            prior_tox_probs=prior_tox_probs,
            tox_target=payload.get("tox_target", 0.3),
            tox_limit=payload.get("tox_limit", 0.33),
            eff_limit=payload.get("eff_limit", 0.05),
            metric=metric,
            max_size=payload.get("max_size", 64),
            stage_one_size=payload.get("stage_one_size", 16),
            tox_certainty=payload.get("tox_certainty", 0.05),
            eff_certainty=payload.get("eff_certainty", 0.05),
            first_dose=payload.get("first_dose", 1),
        )

        tox_scenarios = [tuple(prior_tox_probs)]
        eff_scenarios = [tuple(skeletons[0])]

        n_replicates = 10
        total_sims = len(tox_scenarios) * len(eff_scenarios) * n_replicates
        current_sim_count = 0
        sims = []

        for true_prob_tox in tox_scenarios:
            for true_prob_eff in eff_scenarios:
                for rep in range(n_replicates):
                    report = simulate_trial(
                        trial,
                        true_toxicities=true_prob_tox,
                        true_efficacies=true_prob_eff,
                        cohort_size=3,
                    )
                    report["true_prob_tox"] = true_prob_tox
                    report["true_prob_eff"] = true_prob_eff
                    sims.append(report)
                    current_sim_count += 1
                    progress_callback(int((current_sim_count / total_sims) * 100))

        func_map = WATU.get_summary_functions()
        ps = ParameterSpace()
        ps.add("true_prob_tox", tox_scenarios)
        ps.add("true_prob_eff", eff_scenarios)
        summary_df = extract_sim_data(sims, ps, func_map, return_type="dataframe")

        fig_rec = get_default_provider().plot_bivariate_simulation_recommendation(
            summary_df, high_contrast=False
        )
        figures.append(("Dose Recommendation Probability", fig_rec))

    elif schema_name == "WagesTaitSchema":
        skeletons = payload.get("skeletons") or [
            [0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            [0.5, 0.6, 0.5, 0.4, 0.3, 0.2],
            [0.4, 0.5, 0.6, 0.5, 0.4, 0.3],
        ]
        prior_tox_probs = payload.get("prior_tox_probs") or [
            0.01,
            0.08,
            0.15,
            0.22,
            0.29,
            0.36,
        ]

        trial = WagesTait(
            skeletons=skeletons,
            prior_tox_probs=prior_tox_probs,
            tox_target=payload.get("tox_target", 0.3),
            tox_limit=payload.get("tox_limit", 0.33),
            eff_limit=payload.get("eff_limit", 0.05),
            first_dose=payload.get("first_dose", 1),
            max_size=payload.get("max_size", 64),
            randomisation_stage_size=payload.get("randomisation_stage_size", 16),
        )

        tox_scenarios = [tuple(prior_tox_probs)]
        eff_scenarios = [tuple(skeletons[0])]

        n_replicates = 10
        total_sims = len(tox_scenarios) * len(eff_scenarios) * n_replicates
        current_sim_count = 0
        sims = []

        for true_prob_tox in tox_scenarios:
            for true_prob_eff in eff_scenarios:
                for rep in range(n_replicates):
                    report = simulate_trial(
                        trial,
                        true_toxicities=true_prob_tox,
                        true_efficacies=true_prob_eff,
                        cohort_size=3,
                    )
                    report["true_prob_tox"] = true_prob_tox
                    report["true_prob_eff"] = true_prob_eff
                    sims.append(report)
                    current_sim_count += 1
                    progress_callback(int((current_sim_count / total_sims) * 100))

        func_map = WagesTait.get_summary_functions()
        ps = ParameterSpace()
        ps.add("true_prob_tox", tox_scenarios)
        ps.add("true_prob_eff", eff_scenarios)
        summary_df = extract_sim_data(sims, ps, func_map, return_type="dataframe")

        fig_rec = get_default_provider().plot_bivariate_simulation_recommendation(
            summary_df, high_contrast=False
        )
        figures.append(("Dose Recommendation Probability", fig_rec))

    elif schema_name == "GroupSequentialDesignSchema":
        k = payload.get("k", 3)
        alpha = payload.get("alpha", 0.025)
        timing = payload.get("timing") or [(i + 1) / k for i in range(k)]
        n_sims = payload.get("n_sims") or 1000
        theta = payload.get("theta") or 1.0

        gsd = GroupSequentialDesign(
            k=k, alpha=alpha, timing=timing, sfu=spending_function_obrien_fleming
        )

        sims = []
        batches = 10
        batch_size = max(1, n_sims // batches)
        for b in range(batches):
            batch_sims = gsd.run(n_sims=batch_size, method="bulk", theta=theta)
            sims.extend(batch_sims)
            progress_callback(int(((b + 1) / batches) * 100))

        rejected = [sim.get("Rejected", False) for sim in sims]
        power = sum(rejected) / len(rejected) if rejected else 0.0

        results_dict = {
            "k": k,
            "alpha": alpha,
            "sfu": "O'Brien-Fleming",
            "n_sims": len(sims),
            "theta": theta,
            "power": power,
        }
        summary_df = pd.DataFrame([results_dict])

        from collections import Counter

        stop_stages = [sim.get("Stage", k) for sim in sims]
        stage_counts = Counter(stop_stages)
        stages = list(range(1, k + 1))
        counts = [stage_counts.get(s, 0) for s in stages]
        plot_df = pd.DataFrame(
            {"Stage": stages, "Count": counts, "Outcome": ["Stop" for _ in stages]}
        )
        fig = get_default_provider().create_bar_chart(
            plot_df,
            x="Stage",
            y="Count",
            color="Outcome",
            title="Trial Progression (Stop Stages)",
        )
        figures.append(("Trial Progression (Stop Stages)", fig))

    elif schema_name == "WinRatioSchema":
        trial = WinRatioTrial(**payload)
        num_simulations = payload.get("num_simulations", 1000)

        results_list = []
        batches = 10
        batch_size = max(1, num_simulations // batches)
        for b in range(batches):
            batch_results = trial.run(n_sims=batch_size, method="iterative")
            results_list.extend(list(batch_results))
            progress_callback(int(((b + 1) / batches) * 100))

        successes = sum(1 for r in results_list if r.get("success", False))
        total_sims = len(results_list)

        sum_ci0 = 0.0
        sum_ci1 = 0.0
        ci_count = 0
        for r in results_list:
            if r.get("ci") is not None:
                sum_ci0 += r["ci"][0]
                sum_ci1 += r["ci"][1]
                ci_count += 1

        power = successes / total_sims if total_sims > 0 else 0.0
        average_ci = (
            (sum_ci0 / ci_count, sum_ci1 / ci_count) if ci_count > 0 else (0.0, 0.0)
        )

        results_dict = payload.copy()
        results_dict["power"] = power
        results_dict["ci_lower"] = average_ci[0]
        results_dict["ci_upper"] = average_ci[1]

        summary_df = pd.DataFrame([results_dict])

        fig = get_default_provider().plot_winratio_power_curve(
            summary_df, high_contrast=False
        )
        figures.append(("Win Ratio Simulation Power Curve", fig))

    else:
        raise ValueError(f"Unknown schema name: {schema_name}")

    # Serialize results to HTML and Plotly JSON format
    if accessibility_mode:
        from clintrials.visualization.models import MultiFormatSummaryContainer

        df_for_rendering = summary_df.copy()

        # Reset index to include true_tox / scenario variables as grouping columns
        if df_for_rendering.index.names and any(
            name is not None for name in df_for_rendering.index.names
        ):
            df_for_rendering = df_for_rendering.reset_index()

        # Rename any empty/unnamed index columns
        rename_map = {}
        for c in df_for_rendering.columns:
            if str(c).lower() == "index" or c == "":
                rename_map[c] = "Index"
        df_for_rendering = df_for_rendering.rename(columns=rename_map)

        # Convert unhashable columns (like dicts) to strings
        for col in df_for_rendering.columns:
            try:
                df_for_rendering[col].nunique()
            except TypeError:
                df_for_rendering[col] = df_for_rendering[col].astype(str)

        container = MultiFormatSummaryContainer(title=schema_name, df=df_for_rendering)
        summary_html = container._generate_hierarchical_html()
    else:
        summary_html = summary_df.to_html(classes="table table-striped", index=True)

    # We serialize each plotly figure to JSON
    serialized_figures = []
    for title, fig in figures:
        serialized_figures.append({"title": title, "plotly_json": fig.to_json()})

    return json.dumps({"summary_html": summary_html, "figures": serialized_figures})
