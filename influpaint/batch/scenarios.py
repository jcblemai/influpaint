"""
Simple scenario management for InfluPaint research.
Minimal dataclasses for organization - easy to modify.
"""

from dataclasses import dataclass
from typing import List
import itertools
from .config import AVAILABLE_DDPMS,AVAILABLE_UNETS, AVAILABLE_DATASETS, AVAILABLE_TRANSFORMS, AVAILABLE_ENRICHMENTS, AVAILABLE_COPAINT_CONFIGS


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
    
    print("\n=== AVAILABLE INPAINTING SCENARIOS ===")
    scenarios = get_all_inpainting_scenarios()
    for i, scenario in enumerate(scenarios):
        print(f"{i:2d}: {scenario.scenario_string}")
    
    print(f"\nTotal: {len(scenarios)} inpainting scenarios")


if __name__ == "__main__":
    print_available_scenarios()