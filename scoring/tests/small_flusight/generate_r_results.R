#!/usr/bin/env Rscript
#' Generate R scoringutils reference results for test data
#' 
#' This script processes the small_flusight test data using R scoringutils
#' and saves results to r_scoringutils_results.csv for comparison with Python.
#' 
#' Usage: Rscript generate_r_results.R

library(scoringutils)
library(data.table)
library(dplyr)

# Set paths
test_dir <- getwd()  # Use current working directory
model_output_dir <- file.path(test_dir, "model_output")
target_data_file <- file.path(test_dir, "target-data/target-hospital-admissions.csv")

cat("🧮 Generating R scoringutils reference results...\n")
cat("Test directory:", test_dir, "\n")

# Load ground truth
ground_truth <- fread(target_data_file)
ground_truth[, date := as.Date(date)]
setnames(ground_truth, "date", "target_end_date")
# Normalize location to consistent string format
ground_truth[, location := as.character(location)]
ground_truth[, location := gsub('^"([^"]*)"$', '\\1', location)]  # Remove quotes
ground_truth[, location := ifelse(nchar(location) == 1, paste0('0', location), location)]  # Pad single digits

# Load all forecast files
forecasts_list <- list()
model_dirs <- list.dirs(model_output_dir, full.names = TRUE, recursive = FALSE)

for (model_dir in model_dirs) {
  model_name <- basename(model_dir)
  csv_files <- list.files(model_dir, pattern = "\\.csv$", full.names = TRUE)
  
  for (csv_file in csv_files) {
    df <- fread(csv_file)
    
    # Extract forecast date from filename
    forecast_date <- gsub("^(\\d{4}-\\d{2}-\\d{2}).*", "\\1", basename(csv_file))
    df[, forecast_date := as.Date(forecast_date)]
    df[, model := model_name]
    
    # Prepare data types
    df[, target_end_date := as.Date(target_end_date)]
    df[, output_type_id := as.numeric(output_type_id)]
    df[, horizon := as.integer(horizon)]
    # Normalize location to consistent string format (remove quotes, pad if needed)
    df[, location := as.character(location)]
    df[, location := gsub('^"([^"]*)"$', '\\1', location)]  # Remove surrounding quotes
    df[, location := ifelse(nchar(location) == 1, paste0('0', location), location)]  # Pad single digits
    
    # Filter to quantile forecasts for target of interest
    df <- df[output_type == "quantile" & target == "wk inc flu hosp"]
    
    if (nrow(df) > 0) {
      forecasts_list[[length(forecasts_list) + 1]] <- df
    }
  }
}

# Combine all forecasts
forecasts <- rbindlist(forecasts_list, fill = TRUE)

cat("Forecast columns:", paste(colnames(forecasts), collapse = ", "), "\n")
cat("Looking for columns: output_type_id, value\n")

# Rename columns for scoringutils
setnames(forecasts, 
         old = c("output_type_id", "value"),
         new = c("quantile", "predicted"))

# Merge with ground truth
combined <- merge(forecasts, 
                 ground_truth[, .(target_end_date, location, value)],
                 by = c("target_end_date", "location"),
                 all.x = TRUE)

# Add observed values
combined[, observed := value]
combined[, value := NULL]

# Remove rows without ground truth
combined <- combined[!is.na(observed)]

cat("Combined data:", nrow(combined), "rows\n")

# Convert to forecast object for scoringutils
combined_forecast <- as_forecast_quantile(
  combined,
  forecast_unit = c("model", "target_end_date", "location", "horizon"),
  observed = "observed",
  predicted = "predicted",
  quantile = "quantile"
)

# Compute scores
scores <- combined_forecast %>% score()

cat("Computed scores:", nrow(scores), "rows\n")
cat("Available metrics:", paste(unique(scores$metric), collapse = ", "), "\n")

# Save results
output_file <- file.path(test_dir, "r_scoringutils_results.csv")
fwrite(scores, output_file)

cat("✅ R results saved to:", output_file, "\n")

# Print summary
if ("wis" %in% colnames(scores)) {
  wis_values <- scores$wis[!is.na(scores$wis)]
  if (length(wis_values) > 0) {
    cat("WIS summary:\n")
    cat("  Mean:", mean(wis_values), "\n")
    cat("  SD:", sd(wis_values), "\n")
    cat("  Min:", min(wis_values), "\n")
    cat("  Max:", max(wis_values), "\n")
  }
}