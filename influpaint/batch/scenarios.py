"""
Scenario definitions for InfluPaint experiments.
Clean scenario specifications without heavy object creation logic.
"""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class TrainingScenarioSpec:
    """Specification for a training scenario - no heavy objects, just config"""
    scenario_id: int
    unet_name: str
    dataset_name: str
    transform_name: str
    enrich_name: str
    
    @property
    def scenario_string(self) -> str:
        return f"i{self.scenario_id}::model_{self.unet_name}::dataset_{self.dataset_name}::trans_{self.transform_name}::enrich_{self.enrich_name}"
    
    @property
    def timesteps(self) -> int:
        return 200 if self.unet_name == "MyUnet200" else 500
    
    @property
    def model_key(self) -> str:
        return f"{self.unet_name}"
    
    @property 
    def dataset_key(self) -> str:
        return f"{self.dataset_name}"


@dataclass(frozen=True)
class InpaintingScenarioSpec:
    """Specification for an inpainting scenario"""
    scenario_id: int
    config_name: str
    
    @property
    def scenario_string(self) -> str:
        return f"i{self.scenario_id}::conf_{self.config_name}"


class ScenarioLibrary:
    """Library that defines available scenarios and creates specifications"""
    
    # Available components
    MODELS = ["MyUnet200", "MyUnet500"]
    DATASETS = ["R1Fv", "R1", "SURV_ONLY", "HYBRID_70S_30M", "HYBRID_30S_70M", "MOD_ONLY"]
    TRANSFORMS = ["Lins", "Sqrt"]
    ENRICHMENTS = ["No", "PoisPadScale", "PoisPadScaleSmall", "Pois"]
    COPAINT_CONFIGS = ["celebahq_try1", "celebahq_noTT", "celebahq_noTT2", "celebahq_try3", "celebahq"]
    
    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of available model names"""
        return ScenarioLibrary.MODELS.copy()
    
    @staticmethod
    def get_available_datasets() -> List[str]:
        """Get list of available dataset names"""
        return ScenarioLibrary.DATASETS.copy()
    
    @staticmethod
    def get_available_transforms() -> List[str]:
        """Get list of available transform names"""
        return ScenarioLibrary.TRANSFORMS.copy()
    
    @staticmethod
    def get_available_enrichments() -> List[str]:
        """Get list of available enrichment names"""
        return ScenarioLibrary.ENRICHMENTS.copy()
    
    @staticmethod
    def get_available_copaint_configs() -> List[str]:
        """Get list of available CoPaint config names"""
        return ScenarioLibrary.COPAINT_CONFIGS.copy()
    
    @staticmethod
    def get_training_scenarios() -> List[TrainingScenarioSpec]:
        """Get all training scenario specifications"""
        scenarios = []
        scn_id = 0
        
        for unet_name in ScenarioLibrary.MODELS:
            for dataset_name in ScenarioLibrary.DATASETS:
                for transform_name in ScenarioLibrary.TRANSFORMS:
                    for enrich_name in ScenarioLibrary.ENRICHMENTS:
                        scenario = TrainingScenarioSpec(
                            scenario_id=scn_id,
                            unet_name=unet_name,
                            dataset_name=dataset_name,
                            transform_name=transform_name,
                            enrich_name=enrich_name
                        )
                        scenarios.append(scenario)
                        scn_id += 1
        
        return scenarios
    
    @staticmethod
    def get_inpainting_scenarios() -> List[InpaintingScenarioSpec]:
        """Get all inpainting scenario specifications"""
        scenarios = []
        scn_id = 0
        
        for config_name in ScenarioLibrary.COPAINT_CONFIGS:
            scenario = InpaintingScenarioSpec(
                scenario_id=scn_id,
                config_name=config_name
            )
            scenarios.append(scenario)
            scn_id += 1
        
        return scenarios
    
    @staticmethod
    def get_training_scenario(scenario_id: int) -> TrainingScenarioSpec:
        """Get specific training scenario by ID"""
        scenarios = ScenarioLibrary.get_training_scenarios()
        if scenario_id >= len(scenarios):
            raise ValueError(f"Scenario ID {scenario_id} out of range. Max: {len(scenarios)-1}")
        return scenarios[scenario_id]
    
    @staticmethod
    def get_inpainting_scenario(scenario_id: int) -> InpaintingScenarioSpec:
        """Get specific inpainting scenario by ID"""
        scenarios = ScenarioLibrary.get_inpainting_scenarios()
        if scenario_id >= len(scenarios):
            raise ValueError(f"Scenario ID {scenario_id} out of range. Max: {len(scenarios)-1}")
        return scenarios[scenario_id]