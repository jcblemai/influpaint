"""
Simple scenario management for InfluPaint research.
Minimal dataclasses for organization - easy to modify.
"""

from dataclasses import dataclass
from typing import List
import itertools
from .config import AVAILABLE_DDPMS,AVAILABLE_UNETS, AVAILABLE_DATASETS, AVAILABLE_TRANSFORMS, AVAILABLE_ENRICHMENTS, AVAILABLE_COPAINT_CONFIGS
from .config import CONFIG_BASELINE


@dataclass(frozen=True)
class TrainingScenario:
    """Simple training scenario specification"""
    scenario_id: int
    ddpm_name: str
    unet_name: str
    dataset_name: str
    transform_name: str
    enrich_name: str
    
    @property
    def scenario_string(self) -> str:
        return f"i{self.scenario_id}::m_{self.ddpm_name}{self.unet_name}::ds_{self.dataset_name}::tr_{self.transform_name}::ri_{self.enrich_name}"


@dataclass(frozen=True)
class InpaintingScenario:
    """Simple inpainting scenario specification"""
    scenario_id: int
    config_name: str
    
    @property
    def scenario_string(self) -> str:
        return f"i{self.scenario_id}::conf_{self.config_name}"


def get_all_training_scenarios() -> List[TrainingScenario]:
    """Generate all training scenarios from available options"""
    scenarios = []
    scn_id = 0
    
    for ddpm_name in AVAILABLE_DDPMS:
        for unet_name in AVAILABLE_UNETS:
            for dataset_name in AVAILABLE_DATASETS:
                for transform_name in AVAILABLE_TRANSFORMS:
                    for enrich_name in AVAILABLE_ENRICHMENTS:
                        scenario = TrainingScenario(
                            scenario_id=scn_id,
                            ddpm_name=ddpm_name,
                            unet_name=unet_name,
                            dataset_name=dataset_name,
                            transform_name=transform_name,
                            enrich_name=enrich_name
                        )
                        scenarios.append(scenario)
                        scn_id += 1
    
    return scenarios

from typing import List, Dict
from .config import (
    AVAILABLE_DDPMS, AVAILABLE_UNETS, AVAILABLE_DATASETS, 
    AVAILABLE_TRANSFORMS, AVAILABLE_ENRICHMENTS
)

# This function assumes the 'TrainingScenario' dataclass is defined in the same file.

def get_essential_scenarios(
    all_scenarios: List['TrainingScenario'], 
    baseline: Dict[str, str]
) -> List['TrainingScenario']:
    """
    Prunes a list of scenarios to the essential set for analysis.
    This includes the baseline and all single-parameter variations from it.
    
    For example, if baseline is {ddpm: "U500c", unet: "Rx124", dataset: "70S30M", ...}
    Returns scenarios that vary only ONE parameter while keeping others at baseline:
    - baseline scenario
    - all ddpm options with other params at baseline  
    - all unet options with other params at baseline
    - etc.

    Args:
        all_scenarios: The full list of TrainingScenario objects to be pruned.
        baseline: A dictionary defining the baseline configuration.
                  Keys must match TrainingScenario attribute names.

    Returns:
        A pruned list of TrainingScenario objects.
    """
    essential_scenarios = []
    seen_ids = set()

    # Map baseline keys to the lists of all available options
    options_map = {
        'ddpm_name': AVAILABLE_DDPMS,
        'unet_name': AVAILABLE_UNETS,
        'dataset_name': AVAILABLE_DATASETS,
        'transform_name': AVAILABLE_TRANSFORMS,
        'enrich_name': AVAILABLE_ENRICHMENTS,
    }

    # Helper to check if a scenario object matches a config dictionary
    def _matches(scenario: 'TrainingScenario', config: Dict[str, str]) -> bool:
        for key, value in config.items():
            if getattr(scenario, key) != value:
                return False
        return True

    # 1. Add the baseline configuration
    target_configs = [baseline]
    
    # 2. For each parameter, create configs that vary ONLY that parameter
    for param_key, all_options in options_map.items():
        for option in all_options:
            if baseline[param_key] != option:  # Only non-baseline values
                # Create config with ONLY this parameter changed
                new_config = baseline.copy()
                new_config[param_key] = option
                target_configs.append(new_config)

    # 3. Find the scenario object for each target configuration
    for config in target_configs:
        for scenario in all_scenarios:
            if scenario.scenario_id not in seen_ids and _matches(scenario, config):
                essential_scenarios.append(scenario)
                seen_ids.add(scenario.scenario_id)
                break # Found the match, move to the next config

    return essential_scenarios


def get_all_inpainting_scenarios() -> List[InpaintingScenario]:
    """Generate all inpainting scenarios from available options"""
    scenarios = []
    scn_id = 0
    
    for config_name in AVAILABLE_COPAINT_CONFIGS:
        scenario = InpaintingScenario(
            scenario_id=scn_id,
            config_name=config_name
        )
        scenarios.append(scenario)
        scn_id += 1
    
    return scenarios


def get_training_scenario(scenario_id: int) -> TrainingScenario:
    """Get specific training scenario by ID"""
    scenarios = get_all_training_scenarios()
    if scenario_id >= len(scenarios):
        raise ValueError(f"Scenario ID {scenario_id} out of range. Max: {len(scenarios)-1}")
    return scenarios[scenario_id]


def get_inpainting_scenario(scenario_id: int) -> InpaintingScenario:
    """Get specific inpainting scenario by ID"""
    scenarios = get_all_inpainting_scenarios()
    if scenario_id >= len(scenarios):
        raise ValueError(f"Scenario ID {scenario_id} out of range. Max: {len(scenarios)-1}")
    return scenarios[scenario_id]


# Simple helper for research use
def create_scenario_objects(scenario_spec: TrainingScenario, season_setup, image_size=64, channels=1, batch_size=512, epochs=800, device="cuda"):
    """Create actual objects from scenario spec - one function does everything"""
    from .config import ddpm_library, unet_library, get_dataset, transform_library
    import numpy as np
    
    # Create objects
    unet_spec = unet_library(image_size, channels)
    unet = unet_spec[scenario_spec.unet_name]
    ddpm_spec = ddpm_library(image_size, channels, epochs, device, batch_size, unet=unet)
    ddpm = ddpm_spec[scenario_spec.unet_name]
    
    dataset = get_dataset(scenario_spec.dataset_name, season_setup, channels)
    
    # Create transforms
    # scaling_per_channel = np.array(max(dataset.max_per_feature, gt1.gt_xarr.max(dim=["date", "place"])))
    scaling_per_channel = np.array(dataset.max_per_feature)
    data_mean = dataset.flu_dyn.mean()
    data_std = dataset.flu_dyn.std()
    
    transforms_spec, transform_enrich = transform_library(scaling_per_channel, 
                                                        data_mean=data_mean, 
                                                        data_std=data_std)
    transform = transforms_spec[scenario_spec.transform_name]
    enrich = transform_enrich[scenario_spec.enrich_name]
    
    # Configure dataset
    dataset.add_transform(
        transform=transform["reg"], 
        transform_inv=transform["inv"], 
        transform_enrich=enrich, 
        bypass_test=False
    )
    
    return ddpm, dataset, transform, enrich, scaling_per_channel, data_mean, data_std


def print_available_scenarios():
    """Print all available scenarios for easy reference"""
    print("=== AVAILABLE TRAINING SCENARIOS ===")
    scenarios = get_all_training_scenarios()
    for i, scenario in enumerate(scenarios):
        print(f"{i:2d}: {scenario.scenario_string}")

    print(f"\nTotal: {len(scenarios)} training scenarios")

    print("\n=== ESSENTIAL TRAINING SCENARIOS ===")
    scenarios = get_essential_scenarios(scenarios, CONFIG_BASELINE)
    for i, scenario in enumerate(scenarios):
        print(f"{i:2d}: {scenario.scenario_string}")
    print(f"\nTotal: {len(scenarios)} essential training scenarios")

    
    print("\n=== AVAILABLE INPAINTING SCENARIOS ===")
    scenarios = get_all_inpainting_scenarios()
    for i, scenario in enumerate(scenarios):
        print(f"{i:2d}: {scenario.scenario_string}")
    
    print(f"\nTotal: {len(scenarios)} inpainting scenarios")


if __name__ == "__main__":
    print_available_scenarios()