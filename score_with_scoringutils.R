#!/usr/bin/env Rscript
#' Score combined forecast+truth data using scoringutils R package
#' 
#' Takes a single combined CSV file (forecast+truth already merged in Python)
#' and produces comprehensive scores
#' 
#' Usage: Rscript score_with_scoringutils.R <combined_file> <output_file>

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Usage: Rscript score_with_scoringutils.R <combined_file> <output_file>")
}

combined_file <- args[1]
output_file <- args[2]

library(scoringutils)
library(dplyr)

cat("Starting scoringutils evaluation...\n")
cat("Combined file:", combined_file, "\n")
cat("Output file:", output_file, "\n")

# Load pre-processed data (all merging done in Python)
cat("Loading combined data...\n")
start_time <- Sys.time()

combined <- read.csv(combined_file, stringsAsFactors = FALSE, colClasses = list(location = "character"))

load_time <- Sys.time() - start_time
cat("Data loading completed in", round(as.numeric(load_time, units="secs"), 2), "seconds\n")

cat("Loaded combined data:", nrow(combined), "rows\n")
cat("Unique models:", length(unique(combined$model)), "\n")
cat("Seasons:", paste(unique(combined$season), collapse = ", "), "\n")
cat("Date range:", as.character(min(as.Date(combined$target_end_date))), "to", as.character(max(as.Date(combined$target_end_date))), "\n")

# Prepare data types for scoringutils
cat("Preparing data types...\n")
combined$target_end_date <- as.Date(combined$target_end_date)
combined$forecast_date <- as.Date(combined$forecast_date)

# Check available columns
cat("Available columns:", paste(colnames(combined), collapse = ", "), "\n")

# Convert to scoringutils format
cat("Converting to scoringutils format...\n")
convert_start <- Sys.time()

forecast_object <- as_forecast_quantile(
    combined,
    forecast_unit = c("model", "group", "season", "reference_date", "target_end_date", "location", "horizon"),
    observed = "observed",
    predicted = "predicted", 
    quantile = "quantile"
)

convert_time <- Sys.time() - convert_start
cat("Format conversion completed in", round(as.numeric(convert_time, units="secs"), 2), "seconds\n")

# Score with default metrics
cat("Computing scores...\n")
score_start <- Sys.time()

scores <- forecast_object %>% score()

score_time <- Sys.time() - score_start
cat("Scoring completed in", round(as.numeric(score_time, units="secs"), 2), "seconds\n")

cat("Computed scores:", nrow(scores), "rows\n")
cat("Available columns:", paste(unique(colnames(scores)), collapse = ", "), "\n")

# Check for specific coverage columns
coverage_cols <- colnames(scores)[grepl("coverage", colnames(scores))]
cat("Coverage columns:", paste(coverage_cols, collapse = ", "), "\n")

# Save results
cat("Saving results...\n")
write.csv(scores, output_file, row.names = FALSE)

total_time <- Sys.time() - start_time
cat("TOTAL R PROCESSING TIME:", round(as.numeric(total_time, units="secs"), 2), "seconds\n")
cat("Results saved to:", output_file, "\n")