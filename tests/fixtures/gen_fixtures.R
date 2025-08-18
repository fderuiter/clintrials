# Fixture generation script for clintrials tests
#
# This script generates test fixtures for the Bayesian CRM tests.
# It uses the `bcrm` R package to calculate the expected posterior
# probabilities of dose-limiting toxicity (DLT) and the next
# recommended dose for a set of predefined clinical trial scenarios.
#
# The output of this script is a set of CSV files that are used by
# the `pytest` suite to validate the clintrials Python implementation.
#
# To run this script, you need to have R installed with the `bcrm`
# and `mvtnorm` packages.
#
# To install dependencies (in R):
# > install.packages(c("bcrm", "mvtnorm"))
#
# To run the script (in bash):
# > Rscript tests/fixtures/gen_fixtures.R

library(bcrm)

# Scenario 1: Standard case
# Define prior probabilities of toxicity
p_tox_prior_1 <- c(0.1, 0.2, 0.3, 0.4)
# Define the target toxicity level
target_tox_1 <- 0.3
# Define the observed data: dose levels and DLTs
doses_1 <- c(1, 1, 2, 2, 3, 3)
dlt_1 <- c(0, 0, 0, 1, 1, 1)
data_1 <- data.frame(
  patient = 1:length(doses_1),
  dose = doses_1,
  tox = dlt_1
)

# Run the bcrm analysis
# Using method="exact" to match the Python implementation's approach
crm_1 <- bcrm(
  data = data_1,
  p.tox0 = p_tox_prior_1,
  target.tox = target_tox_1,
  ff = "logit1",
  prior.alpha = list(3, 0, 1.34^2),
  constrain = FALSE,
  sdose.calculate = "mean",
  pointest = "mean",
  method = "exact",
  stop = list(nmax = length(doses_1))
)

# Extract the posterior probabilities of DLT and the next recommended dose
posterior_dlt_probs_1 <- crm_1$ndose[[length(crm_1$ndose)]]$est
next_dose_1 <- crm_1$ndose[[length(crm_1$ndose)]]$ndose

# Scenario 2: No toxicities observed
p_tox_prior_2 <- c(0.05, 0.1, 0.2, 0.35, 0.5)
target_tox_2 <- 0.2
doses_2 <- c(1, 1, 1, 2, 2, 2)
dlt_2 <- c(0, 0, 0, 0, 0, 0)
data_2 <- data.frame(
  patient = 1:length(doses_2),
  dose = doses_2,
  tox = dlt_2
)

crm_2 <- bcrm(
  data = data_2,
  p.tox0 = p_tox_prior_2,
  target.tox = target_tox_2,
  ff = "logit1",
  prior.alpha = list(3, 0, 1.34^2),
  constrain = FALSE,
  sdose.calculate = "mean",
  pointest = "mean",
  method = "exact",
  stop = list(nmax = length(doses_2))
)

posterior_dlt_probs_2 <- crm_2$ndose[[length(crm_2$ndose)]]$est
next_dose_2 <- crm_2$ndose[[length(crm_2$ndose)]]$ndose

# Combine the results into data frames
posterior_dlt_probs_df <- rbind(
  data.frame(scenario = 1, dose = 1:length(posterior_dlt_probs_1), prob = posterior_dlt_probs_1),
  data.frame(scenario = 2, dose = 1:length(posterior_dlt_probs_2), prob = posterior_dlt_probs_2)
)

next_dose_df <- data.frame(
  scenario = c(1, 2),
  next_dose = c(next_dose_1, next_dose_2)
)

# Write the data frames to CSV files
write.csv(posterior_dlt_probs_df, "tests/fixtures/expected_posterior_dlt_probs.csv", row.names = FALSE)
write.csv(next_dose_df, "tests/fixtures/next_dose_recommendations.csv", row.names = FALSE)

print("Fixtures generated successfully.")
