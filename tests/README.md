# InfluPaint Scoring Module Tests

Minimal pytest test suite for the InfluPaint scoring module.

## Test Data

Uses `scoring/tests/small_flusight/` with:
- **3 models**: CADPH-FluCAT_Ensemble, FluSight-baseline, UNC_IDD-InfluPaint  
- **2 forecast dates**: 2024-12-28, 2025-03-22
- **Ground truth**: Hospital admissions data

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific tests
python -m pytest tests/test_scoring.py -v
python -m pytest tests/test_r_comparison.py -v
```

## Test Files

- `test_scoring.py` - Basic functionality tests (data loading, metric computation)
- `test_r_comparison.py` - Comparison with R scoringutils package

## R Comparison Setup

To enable R comparison tests:

1. **Install R packages:**
   ```r
   install.packages(c("scoringutils", "data.table", "dplyr"))
   ```

2. **Generate R reference results:**
   ```bash
   cd scoring/tests/small_flusight/
   Rscript generate_r_results.R
   ```

3. **Enable R comparison test:**
   ```python
   # In tests/test_r_comparison.py, change:
   @pytest.mark.skipif(True, reason="R results not yet generated")
   # to:
   @pytest.mark.skipif(False, reason="R results available")
   ```

The R script will create `r_scoringutils_results.csv` with WIS scores computed by the R scoringutils package for comparison with Python results.