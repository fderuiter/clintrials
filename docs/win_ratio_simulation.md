# Win-Ratio Simulation

This module runs Monte Carlo simulations to estimate the statistical power of a
win-ratio test for a composite endpoint. The win-ratio compares outcomes between
a treatment and a control group when several ordered outcomes are of interest.

## How it Works

1. **Data Generation** – Binary outcomes for treatment and control groups are
   generated with user-specified probabilities.
2. **Pairwise Comparison** – Every treatment subject is compared with every
   control subject hierarchically across outcomes; the first difference decides
the winner.
3. **Win-Ratio Calculation** – Total wins and losses form a win ratio.
4. **Statistical Analysis** – Confidence intervals and p-values are computed on
   the log scale.
5. **Power Calculation** – Repeating the simulation many times yields the
   proportion of significant results, i.e. the power.

## Running the Simulation

```bash
python -m clintrials.winratio.main
```

Command-line options allow the sample sizes, probabilities and number of
simulations to be modified. The script reports the estimated power and the
average 95% confidence interval for the win ratio.
