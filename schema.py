from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

@dataclass
class StructureFeatures:
    lcd_angstrom: Optional[float] = None
    pld_angstrom: Optional[float] = None
    lfpd_angstrom: Optional[float] = None
    density_g_cm3: Optional[float] = None
    asa_m2_g: Optional[float] = None
    asa_m2_cm3: Optional[float] = None
    nasa_m2_g: Optional[float] = None
    nasa_m2_cm3: Optional[float] = None
    pore_volume_cm3_g: Optional[float] = None
    porosity: Optional[float] = None
    void_fraction: Optional[float] = None

    leaf_area_cm2: Optional[float] = None
    leaf_thickness_mm: Optional[float] = None
    stomatal_density_per_mm2: Optional[float] = None
    root_depth_cm: Optional[float] = None
    internal_air_space_fraction: Optional[float] = None

@dataclass
class ChemistryFeatures:
    metal_nodes: Optional[str] = None
    organic_linkers: Optional[str] = None
    functional_groups: List[str] = field(default_factory=list)
    mofid_v1: Optional[str] = None
    mofid_v2: Optional[str] = None
    refcode: Optional[str] = None

    c3_c4_cam: Optional[str] = None
    chlorophyll_content: Optional[float] = None
    nitrogen_content_percent: Optional[float] = None

@dataclass
class StabilityFeatures:
    thermal_stability_C: Optional[float] = None
    solvent_stability: Optional[float] = None
    water_stability: Optional[float] = None
    mechanical_stability: Optional[float] = None
    ph_stability_range: Optional[str] = None
    uv_stability: Optional[float] = None

@dataclass
class PerformanceMetrics:
    h2_uptake_wt_percent: Optional[float] = None
    h2_volumetric_capacity_g_L: Optional[float] = None

    co2_uptake_mmol_g: Optional[float] = None
    co2_uptake_mmol_cm3: Optional[float] = None
    co2_selectivity: Optional[float] = None
    kh_class: Optional[str] = None
    kh_value: Optional[float] = None

    h2o_uptake_mmol_g: Optional[float] = None
    sorption_isotherm_type: Optional[str] = None
    regeneration_energy_kJ_mol: Optional[float] = None

    co2_assimilation_umol_m2_s: Optional[float] = None
    transpiration_rate_mmol_m2_s: Optional[float] = None
    water_use_efficiency: Optional[float] = None

@dataclass
class Material:
    id: str
    name: Optional[str]
    material_type: str  # "MOF", "Plant", "Hybrid"
    source: Optional[str] = None

    structure: StructureFeatures = field(default_factory=StructureFeatures)
    chemistry: ChemistryFeatures = field(default_factory=ChemistryFeatures)
    stability: StabilityFeatures = field(default_factory=StabilityFeatures)
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)

    text_description: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
