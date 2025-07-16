# Batch Processing

Parallel training and inpainting for InfluPaint experiments.

## Files

- `config.py` - Model, dataset, transform configurations
- `scenarios.py` - Scenario generation and object creation
- `training.py` - Train diffusion models
- `inpainting.py` - Generate forecasts
- `jobs.py` - Generate SLURM array jobs
- `mlflow_utils.py` - MLflow utilities
- `aggregation.py` - Result aggregation

## Usage

### Training
```bash
python -m influpaint.batch.training -s 5 -e "experiment_name"
sbatch train.run
```

### Inpainting
```bash
python -m influpaint.batch.inpainting -s 5 -r "mlflow_run_id" -e "experiment_name" --forecast_date "2022-11-14" --config_name "celebahq_try1"
sbatch inpaint.run
```

### Array Jobs
```bash
python -m influpaint.batch.jobs -e "experiment_name" --scenarios "0-31" --start_date "2022-10-12" --end_date "2023-05-15"
sbatch inpaint_array_*.run
```

## Configuration

### Add Model
```python
# config.py
AVAILABLE_MODELS = ["MyUnet200", "MyUnet500", "NewModel"]

def model_library():
    unet_spec = {
        "NewModel": ddpm.DDPM(...)
    }
```

### Add Dataset
```python
# config.py
AVAILABLE_DATASETS = ["R1Fv", "R1", "NewDataset"]

def dataset_library():
    dataset_spec = {
        "NewDataset": training_datasets.FluDataset.from_source(...)
    }
```

### Modify Parameters
```python
# config.py
def copaint_config_library(timesteps):
    config_lib = {
        "celebahq_try1": config.Config(default_config_dict={
            "jump_length": 20,  # Change value
            "jump_n_sample": 4,  # Change value
        })
    }
```

## Scenarios

Training scenarios combine:
- Model: MyUnet200, MyUnet500
- Dataset: R1Fv, R1, SURV_ONLY, HYBRID_70S_30M, HYBRID_30S_70M, MOD_ONLY
- Transform: Lins, Sqrt
- Enrichment: No, PoisPadScale, PoisPadScaleSmall, Pois

Inpainting scenarios use:
- Config: celebahq_try1, celebahq_noTT, celebahq_noTT2, celebahq_try3, celebahq

List all scenarios:
```bash
python -m influpaint.batch.scenarios
```

## Workflow

1. Train models: `sbatch train.run`
2. Generate jobs: `python jobs.py`
3. Run inpainting: `sbatch inpaint_array_*.run`
4. Check results in MLflow and output directory

## Output

- Training: Model checkpoints, MLflow artifacts
- Inpainting: Forecast files, plots, MLflow metrics
- Jobs: SLURM scripts, job lists