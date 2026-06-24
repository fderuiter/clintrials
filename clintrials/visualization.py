"""Centralized visualization module using Plotly."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_dose_finding_outcomes(trial, chart_title=None):
    """Plots the dose-finding trial outcomes."""
    if not chart_title:
        chart_title = "Each point represents a patient<br>A circle indicates no toxicity, a cross toxicity"

    if trial.size() == 0:
        return go.Figure()

    patient_number = list(range(1, trial.size() + 1))
    doses = trial.doses()
    toxicities = trial.toxicities()

    symbol = ["x" if t else "circle-open" for t in toxicities]
    size = [12] * len(toxicities)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=patient_number,
            y=doses,
            mode="markers",
            marker=dict(
                symbol=symbol,
                size=size,
                color="black",
                line=dict(width=2, color="black"),
            ),
            name="Patients",
        )
    )

    fig.update_layout(
        title=chart_title,
        xaxis_title="Patient number",
        yaxis_title="Dose level",
        yaxis=dict(tickvals=list(trial.dose_levels())),
        showlegend=False,
        width=800,
        height=500,
        template="plotly_white",
    )
    return fig


def plot_crm_toxicity_probabilities(trial, chart_title=None):
    """Plots the CRM dose-toxicity probabilities."""
    if not chart_title:
        chart_title = "Prior (dashed) and posterior (solid) dose-toxicity curves"

    dl = list(trial.dose_levels())
    prior_tox = trial.prior
    post_tox = trial.prob_tox()
    post_tox_lower = trial.get_tox_prob_quantile(0.05)
    post_tox_upper = trial.get_tox_prob_quantile(0.95)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dl,
            y=prior_tox,
            mode="lines+markers",
            line=dict(dash="dash", color="black"),
            marker=dict(symbol="x", size=12, color="black"),
            name="Prior",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dl,
            y=post_tox,
            mode="lines+markers",
            line=dict(color="black"),
            marker=dict(
                symbol="circle-open",
                size=12,
                color="black",
                line=dict(width=2, color="black"),
            ),
            name="Posterior",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dl,
            y=post_tox_lower,
            mode="lines",
            line=dict(dash="dashdot", color="black"),
            name="5% Quantile",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dl,
            y=post_tox_upper,
            mode="lines",
            line=dict(dash="dashdot", color="black"),
            name="95% Quantile",
            fill="tonexty",
            fillcolor="rgba(0,0,0,0.1)",
        )
    )

    fig.add_hline(y=trial.target, line_color="red", annotation_text="Target")

    fig.update_layout(
        title=chart_title,
        xaxis_title="Dose level",
        yaxis_title="Probability of toxicity",
        xaxis=dict(tickvals=dl),
        yaxis=dict(range=[0, 1]),
        template="plotly_white",
        width=800,
        height=500,
    )
    return fig


def plot_efftox_utility_contours(
    metric,
    prob_eff=None,
    prob_tox=None,
    n=100,
    util_lower=-0.8,
    util_upper=0.8,
    util_delta=0.2,
    title="EffTox utility contours",
    custom_points_label="priors",
):
    """Plots the EffTox utility contours."""
    eff_vals = np.linspace(0, 1, n)
    util_vals = np.linspace(
        util_lower, util_upper, int(round((util_upper - util_lower) / util_delta)) + 1
    )

    fig = go.Figure()

    # Plot general contours
    for u in util_vals:
        tox_vals = [metric.get_tox(eff=x, util=u) for x in eff_vals]
        # remove None or out of bounds tox values
        valid_eff = []
        valid_tox = []
        for e, t in zip(eff_vals, tox_vals):
            if t is not None and 0 <= t <= 1:
                valid_eff.append(e)
                valid_tox.append(t)
        if valid_eff:
            fig.add_trace(
                go.Scatter(
                    x=valid_eff,
                    y=valid_tox,
                    mode="lines",
                    line=dict(color="gray", width=1),
                    showlegend=False,
                )
            )

    # Add neutral utility contour
    tox_vals = [metric.get_tox(eff=x, util=0) for x in eff_vals]
    valid_eff = []
    valid_tox = []
    for e, t in zip(eff_vals, tox_vals):
        if t is not None and 0 <= t <= 1:
            valid_eff.append(e)
            valid_tox.append(t)
    if valid_eff:
        fig.add_trace(
            go.Scatter(
                x=valid_eff,
                y=valid_tox,
                mode="lines",
                line=dict(color="black", width=3),
                name="neutral utility",
            )
        )

    # Add hinge points
    hinge_prob_eff, hinge_prob_tox = zip(*metric.hinge_points)
    fig.add_trace(
        go.Scatter(
            x=hinge_prob_eff,
            y=hinge_prob_tox,
            mode="markers",
            marker=dict(color="red", size=10),
            name="hinge points",
        )
    )

    # Add custom points
    if prob_eff is not None and prob_tox is not None:
        fig.add_trace(
            go.Scatter(
                x=prob_eff,
                y=prob_tox,
                mode="markers",
                marker=dict(color="blue", size=8, symbol="x"),
                name=custom_points_label,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Probability of Efficacy",
        yaxis_title="Probability of Toxicity",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        template="plotly_white",
        width=800,
        height=600,
    )
    return fig


def plot_efftox_density(
    data_func, trial, x_name="", plot_title="", include_doses=None, boot_samps=1000
):
    """Plots the EffTox probability densities."""
    if include_doses is None:
        include_doses = range(1, trial.num_doses + 1)

    x_boot = []
    dose_indices = []
    samp = trial.pds._samp
    p = trial.pds._probs
    p /= p.sum()

    for i, x in enumerate(trial.scaled_doses()):
        dose_index = i + 1
        if dose_index in include_doses:
            dist = np.random.multinomial(boot_samps, p)
            samp_boot = []
            for j, count in enumerate(dist):
                if count > 0:
                    samp_boot.extend([samp[j]] * count)
            samp_boot = np.array(samp_boot)

            vals = data_func(x, samp_boot)
            x_boot.extend(vals)
            dose_indices.extend([str(dose_index)] * boot_samps)

    df = pd.DataFrame({x_name: x_boot, "Dose": dose_indices})

    # Use plotly express for kernel density estimation via violin or histogram, or scipy gaussian_kde
    import plotly.figure_factory as ff

    hist_data = [df[df["Dose"] == str(d)][x_name] for d in include_doses]
    group_labels = [f"Dose {d}" for d in include_doses]

    fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False)
    fig.update_layout(
        title=plot_title,
        xaxis_title=x_name,
        yaxis_title="Density",
        template="plotly_white",
        width=800,
        height=500,
    )
    return fig


def plot_crm_simulation_recommendation(summary_df):
    """Plots CRM simulation recommendation probabilities."""
    if summary_df.empty or "recommended_dose_prob" not in summary_df.columns:
        return go.Figure()
    rec_dose_df = summary_df["recommended_dose_prob"].apply(pd.Series).fillna(0)
    rec_dose_df_melted = rec_dose_df.reset_index().melt(
        id_vars=[col for col in rec_dose_df.index.names],
        var_name="Dose Level",
        value_name="Probability",
    )
    fig = px.bar(
        rec_dose_df_melted,
        x="true_tox",
        y="Probability",
        color="Dose Level",
        barmode="group",
        labels={
            "true_tox": "True Toxicity Scenario",
            "Probability": "Recommendation Probability",
        },
        title="Dose Recommendation Probabilities by Scenario",
    )
    return fig


def plot_efftox_simulation_recommendation(summary_df):
    """Plots EffTox simulation recommendation probabilities."""
    if summary_df.empty or "recommended_dose_prob" not in summary_df.columns:
        return go.Figure()
    rec_dose_df = summary_df["recommended_dose_prob"].apply(pd.Series).fillna(0)
    rec_dose_df_melted = rec_dose_df.reset_index().melt(
        id_vars=[col for col in rec_dose_df.index.names],
        var_name="Dose Level",
        value_name="Probability",
    )
    fig = px.bar(
        rec_dose_df_melted,
        x=["true_prob_tox", "true_prob_eff"],
        y="Probability",
        color="Dose Level",
        barmode="group",
        title="Dose Recommendation Probabilities",
    )
    return fig


def plot_efftox_simulation_acceptability(summary_df):
    """Plots EffTox simulation acceptability probabilities."""
    if (
        summary_df.empty
        or "prob_accept_tox" not in summary_df.columns
        or "prob_accept_eff" not in summary_df.columns
    ):
        return go.Figure()
    accept_df = summary_df[["prob_accept_tox", "prob_accept_eff"]].reset_index()
    accept_df_melted = accept_df.melt(
        id_vars=["true_prob_tox", "true_prob_eff"],
        var_name="Probability Type",
        value_name="Probability",
    )
    fig = px.line(
        accept_df_melted,
        x="true_prob_tox",  # simplified axis mapping
        y="Probability",
        color="Probability Type",
        title="Probability of Acceptable Efficacy and Toxicity",
    )
    return fig
