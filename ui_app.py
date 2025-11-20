import io
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from llm_agent import (
    run_llm_agent_once,
    run_llm_hybrid_agent,
    parse_hybrid_json,
    link_hybrid_ideas_to_data,
)
from closed_loop_agent import run_closed_loop_optimization
from viz_3d import cif_to_3d_view, detect_metal_elements

from enzyme_map import enzymes_for_plant_row
from chem_tools import enzyme_pdb_to_html

import plotly.express as px

from chem_tools import (
    smiles_is_available,
    validate_hybrid_idea_chemistry,
    make_linker_image,
    make_linker_3d_html,
)

CIF_DIR = Path("./cifs")

# ==========================================
#  TRANSLATIONS
# ==========================================

TRANSLATIONS = {
    "en": {
        "app_title": "🧪 Material Discovery Sidekick",

        "lang_toggle_help": "Switch between Arabic and English UI.",

        # Main sections
        "prompt_builder_title": "Prompt Template Builder 🧱",
        "target_gas_label": "Target gas:",
        "application_scenario_label": "Application scenario (required)",
        "application_scenario_help": "Where will this CO₂-capturing material operate?",

        "application_scenario_options": [
            "",
            "Post-combustion flue gas capture",
            "Direct air capture",
            "CO₂ capture in industrial exhaust streams",
            "CO₂ capture in enclosed / controlled environments",
        ],

        "operating_conditions_title": "Operating conditions",
        "temp_slider_help": "Approximate temperature window the material should tolerate.",
        "pressure_slider_help": "Approximate operating pressure window (e.g., flue gas, DAC, or packed-bed operation).",

        "humidity_label": "Humidity / moisture environment",
        "humidity_help": (
            "Approximate relative humidity (RH) expected during CO₂ adsorption:\n"
            "- Dry: 0–30% RH (desert, industrial flue gas after drying)\n"
            "- Moderate: 30–60% RH (most ambient conditions)\n"
            "- High: 60–90% RH (tropical, greenhouse, moist air)\n"
            "Humidity strongly affects MOF stability and water competition at adsorption sites."
        ),
        "humidity_options": [
            "Dry (0–30% RH)",
            "Moderate humidity (30–60% RH)",
            "High humidity (60–90% RH)",
            "Variable / unknown",
        ],

        "design_priorities_title": "Design priorities (0 = not important, 1 = critical)",

        "affinity_label": "Affinity / selectivity (KH class, CO₂ binding strength)",
        "affinity_help": (
            "Higher = prioritize MOFs with stronger Henry's constant (KH) for CO₂.\n"
            "Useful for low-pressure capture (DAC) or removing dilute CO₂."
        ),

        "capacity_label": "Surface area / pore volume",
        "capacity_help": (
            "Higher = emphasize large geometric capacity: high BET surface area and pore volume.\n"
            "Important for high-uptake applications and large-volume gas streams."
        ),

        "porosity_label": "Porosity / void fraction",
        "porosity_help": (
            "Higher = prioritize materials with open, accessible pores.\n"
            "Affects gas diffusion, uptake kinetics, and hybrid design based on plant stomatal traits."
        ),

        "density_label": "Low framework density",
        "density_help": (
            "Higher = prefer low-density frameworks.\n"
            "Useful for weight-sensitive applications (portable devices, aerospace) "
            "and for maximizing CO₂ uptake per gram."
        ),

        "stability_label": "Stability / robustness",
        "stability_help": (
            "Higher = prefer water-stable, thermally stable MOFs that maintain structural integrity.\n"
            "Critical in humid, high-temperature, or long-term cycling conditions."
        ),

        "extra_notes_label": "Additional design notes (optional)",
        "extra_notes_placeholder": (
            "E.g., avoid very hydrophilic open metal sites in humid flue gas; "
            "prefer Zr-based MOFs; emulate C₄ plant traits at high temperature; "
            "prioritize materials with known scalability and stability."
        ),

        "compiled_goal_title": "Compiled goal that will be sent to the agent",

        "must_select_scenario": (
            "Please select an **Application scenario**. "
            "This is required before running the agent."
        ),
        "cannot_run_agent": "Cannot run agent: please specify an application scenario first.",
        "cannot_run_closed_loop": (
            "Cannot run closed-loop optimization: please specify an application scenario first."
        ),

        "agent_config_title": "Agent Configuration",
        "agent_mode_label": "Agent mode:",
        "agent_mode_hybrid": "MOF–Plant hybrid (hybrid ideas)",
        "agent_mode_mof_only": "MOF-only (ranking & explanation)",
        "top_k_mofs_label": "Top K MOFs",
        "top_k_plants_label": "Top K Plants",
        "n_hybrids_label": "Number of hybrid ideas to generate",
        "n_hybrids_help": "How many distinct MOF–plant hybrid concepts the agent should design.",
        "closed_loop_iters_label": "Closed-loop iterations",
        "closed_loop_iters_help": "Number of iterations to optimize specific MOF objectives.",

        "run_agent_btn": "🚀 Run agent",
        "run_closed_loop_btn": "🧭 Run closed-loop optimization",

        "results_title": "Results",

        # Goal text pieces (English kept for backend consistency)
        "goal_header_mof_only": (
            "\nFocus on ranking and explaining MOFs for this CO₂ capture scenario, without hybridization."
        ),
        "goal_header_hybrid": (
            "\nHybrid requirement: combine MOF candidates with plant species whose "
            "photosynthetic traits and CO₂ assimilation profiles match the above scenario. "
            "Design MOF–plant-inspired hybrid materials for ultra-efficient CO₂ capture and "
            "oxygen balance."
        ),

        # Sections – MOFs
        "top_mof_candidates": "Top MOF candidates (from fused model)",
        "mof_feature_space_title": "3D visualization of MOF feature space",
        "full_mof_df_expander": "Full MOF dataframe",

        # Sections – Plants
        "top_plant_candidates": "Top Plant candidates",
        "plant_feature_space_title": "3D visualization of plant trait space",

        # LLM prompt/response
        "llm_prompt_title": "LLM Prompt (preview)",
        "llm_raw_response_title": "LLM Raw Response",
        "llm_prompt_expander": "LLM Prompt",
        "llm_raw_response_expander": "LLM Raw Response + JSON script",

        "hybrid_json_parse_fail": (
            "Could not parse hybrid JSON from the LLM response. "
            "This may happen if the model didn't follow the schema exactly."
        ),

        # Hybrid MOFs explanation
        "hybrid_section_title": "Generated Plant-Inspired Hybrid MOFs",
        "key_features_title": "Key features:",
        "release_section_title": "Suggested CO₂ release / regeneration method:",
        "release_method_label": "Method",
        "release_rationale_label": "Rationale",
        "inspiration_expander_title": "Inspiration for",

        "mof_subsection_title": "MOFs",
        "plant_subsection_title": "Plants",
        "plant_enzymes_title": "Enzyme structures (by plant pathway/subtype)",
        "plant_justifications_title": "Plant Justifications:",
        "mof_justifications_title": "MOF Justifications:",

        # Cheminformatics
        "cheminfo_title": "Cheminformatics validation",
        "cheminfo_no_rdkit": (
            "RDKit is not available in this environment. "
            "Cheminformatics validation is disabled."
        ),
        "cheminfo_no_linkers": "No `linker_smiles` were provided in this hybrid idea.",
        "cheminfo_invalid_smiles": "Invalid SMILES according to RDKit.",
        "cheminfo_errors": "Errors:",
        "cheminfo_warnings": "Warnings:",
        "cheminfo_3d_unavailable": "_3D depiction not available for this linker (RDKit or py3Dmol issue)._",

        # 3D MOF block
        "mof_3d_title": "3D structure for MOF",
        "style_label": "Style",
        "show_surface_label": "Show surface",
        "legend_expander": "Legend & metal information",
        "legend_title": "Atom color legend (typical 3Dmol defaults):",
        "legend_metals_label": "Detected metal center(s):",

        # 3D feature-space warnings
        "mof_3d_no_data": "No MOF data available for 3D plotting.",
        "mof_3d_not_enough_numeric": (
            "Cannot render MOF 3D feature-space plot: not enough numeric columns "
            "available for fallback visualization."
        ),
        "mof_3d_all_nan": (
            "Cannot render MOF 3D feature-space plot: numeric columns are present "
            "but all rows have NaNs."
        ),

        "plant_3d_missing_cols": "Cannot render plant 3D feature-space plot. Missing columns: ",

        # Closed-loop
        "closed_loop_title": "🌀 Closed-loop optimization (MOFs only)",
        "closed_loop_desc": (
            "Use the same compiled goal text, but let the LLM iteratively adjust the multi-objective "
            "weights (affinity, ASA, void fraction, density penalty) and re-rank MOFs over "
            "several iterations."
        ),
        "closed_loop_final_weights": "Final multi-objective weights:",
        "closed_loop_weight_evolution": "Weight evolution over iterations:",
        "closed_loop_top_mofs": "Top MOFs after final iteration:",
        "closed_loop_full_table_expander": "Full top-MOF table (final iteration)",
        "closed_loop_raw_llm_expander": "Raw LLM responses per iteration",

        # Generic
        "no_mofs_for_ids": "_No matching MOFs found for these IDs._",
        "no_plants_for_ids": "_No matching plants found for these IDs._",
        "no_enzyme_mapping": "_No enzyme mapping available for this pathway/subtype._",
        "no_physio_data": "No physiological data available.",
    },
    "ar": {
        "app_title": "🧪 المساعد الذكي لاكتشاف المواد",

        "lang_toggle_help": "بدّل بين الواجهة العربية والإنجليزية.",

        # Main sections
        "prompt_builder_title": "منشئ نموذج الطلب 🧱",
        "target_gas_label": "غاز الهدف:",
        "application_scenario_label": "سيناريو التطبيق (إلزامي)",
        "application_scenario_help": "في أي بيئة سيعمل نظام التقاط CO₂؟",

        "application_scenario_options": [
            "",
            "التقاط غاز CO₂ بعد الاحتراق من مداخن المصانع",
            "الالتقاط المباشر لثاني أكسيد الكربون من الهواء (DAC)",
            "التقاط CO₂ من غازات عادم صناعية مركّزة",
            "التقاط CO₂ من بيئات مغلقة أو محكومة (مثل البيوت المحمية أو المحطات الفضائية)",
        ],

        "operating_conditions_title": "ظروف التشغيل",
        "temp_slider_help": "نطاق درجات الحرارة التقريبي الذي يجب أن يتحمله النظام.",
        "pressure_slider_help": (
            "نطاق الضغط التقريبي أثناء التشغيل (مثل غاز المداخن، الالتقاط المباشر من الهواء، أو أعمدة التعبئة)."
        ),

        "humidity_label": "الرطوبة / البيئة المائية",
        "humidity_help": (
            "النطاق التقريبي للرطوبة النسبية (RH) أثناء التقاط CO₂:\n"
            "- جافة: 0–30٪ RH (مناطق صحراوية أو غاز مداخن بعد التجفيف)\n"
            "- متوسطة: 30–60٪ RH (الظروف الجوية الشائعة)\n"
            "- عالية: 60–90٪ RH (مناخات رطبة أو بيوت محمية)\n"
            "الرطوبة تؤثر بقوة على ثبات الـ MOF وعلى منافسة الماء مع CO₂ على مواقع الامتزاز."
        ),
        "humidity_options": [
            "جافة (0–30٪ RH)",
            "رطوبة متوسطة (30–60٪ RH)",
            "رطوبة عالية (60–90٪ RH)",
            "متغيرة / غير معروفة",
        ],

        "design_priorities_title": "أولويات التصميم (0 = غير مهم، 1 = حرج)",

        "affinity_label": "الألفة / الانتقائية (قيمة KH وقوة ارتباط CO₂)",
        "affinity_help": (
            "قيمة أعلى = تفضيل MOFs ذات ثابت هنري (KH) أقوى لـ CO₂.\n"
            "مفيد عند التقاط CO₂ بضغط منخفض أو تركيزات ضعيفة."
        ),

        "capacity_label": "المساحة السطحية / حجم المسام",
        "capacity_help": (
            "قيمة أعلى = التركيز على السعة الهندسية الكبيرة: مساحة سطحية عالية وحجم مسام كبير.\n"
            "مهم للتقاط كميات كبيرة من الغاز في التدفقات الضخمة."
        ),

        "porosity_label": "المسامية / جزء الفراغ",
        "porosity_help": (
            "قيمة أعلى = تفضيل مواد ذات مسامات مفتوحة وسهلة الوصول.\n"
            "يؤثر على انتشار الغاز، وسرعة الامتزاز، وتصميم الهجين المستوحى من ثغور النبات."
        ),

        "density_label": "انخفاض كثافة الهيكل",
        "density_help": (
            "قيمة أعلى = تفضيل هياكل منخفضة الكثافة.\n"
            "مفيد للتطبيقات الحساسة للوزن (أجهزة محمولة، طيران) ولتعظيم CO₂ لكل غرام من المادة."
        ),

        "stability_label": "الثبات / المتانة",
        "stability_help": (
            "قيمة أعلى = تفضيل MOFs مستقرة مائياً وحرارياً وتحافظ على بنيتها.\n"
            "حرج في البيئات الرطبة أو الحارة أو عند إعادة الاستخدام لعدد كبير من الدورات."
        ),

        "extra_notes_label": "ملاحظات تصميم إضافية (اختياري)",
        "extra_notes_placeholder": (
            "مثال: تجنّب المواقع المفتوحة شديدة الألفة للماء في غاز المداخن الرطب؛ "
            "تفضيل MOFs مبنية على الزركونيوم؛ محاكاة خصائص نباتات C₄ عند درجات حرارة مرتفعة؛ "
            "تفضيل المواد ذات القابلية العالية للتصنيع والثبات طويل الأمد."
        ),

        "compiled_goal_title": "الهدف المجمّع الذي سيتم إرساله للوكيل",

        "must_select_scenario": (
            "الرجاء اختيار **سيناريو تطبيق** أولاً. "
            "هذا الحقل إلزامي قبل تشغيل الوكيل."
        ),
        "cannot_run_agent": "لا يمكن تشغيل الوكيل: الرجاء تحديد سيناريو التطبيق أولاً.",
        "cannot_run_closed_loop": (
            "لا يمكن تشغيل حلقة التحسين المغلقة: الرجاء تحديد سيناريو التطبيق أولاً."
        ),

        "agent_config_title": "إعدادات الوكيل",
        "agent_mode_label": "وضع الوكيل:",
        "agent_mode_hybrid": "هجين MOF–نبات (أفكار هجينة)",
        "agent_mode_mof_only": "MOF فقط (ترتيب + تفسير)",
        "top_k_mofs_label": "عدد أفضل MOFs (K)",
        "top_k_plants_label": "عدد أفضل النباتات (K)",
        "n_hybrids_label": "عدد المواد الهجينة المقترحة",
        "n_hybrids_help": "كم فكرة مختلفة لمواد هجينة MOF–نبات يجب على الوكيل اقتراحها.",
        "closed_loop_iters_label": "عدد دورات حلقة التحسين المغلقة",
        "closed_loop_iters_help": "عدد الدورات لتحسين أهداف MOF المحددة.",

        "run_agent_btn": "🚀 تشغيل الوكيل",
        "run_closed_loop_btn": "🧭 تشغيل حلقة التحسين المغلقة",

        "results_title": "النتائج",

        # Goal text pieces (keep basic English keywords for backend, Arabic explanation appended)
        "goal_header_mof_only": (
            "\nالتركيز على ترتيب وشرح مواد MOF لهذا السيناريو لالتقاط CO₂، بدون دمج النباتات."
        ),
        "goal_header_hybrid": (
            "\nمتطلب هجين: دمج مرشّحات MOF مع أنواع نباتية "
            "تُظهِر خصائص ضوئية وتمثيل كربوني متوافقة مع السيناريو أعلاه. "
            "صمِّم مواد هجينة مستوحاة من النبات و-MOF لتحقيق التقاط فائق الكفاءة لثاني أكسيد الكربون "
            "مع الحفاظ على توازن الأكسجين."
        ),

        # Sections – MOFs
        "top_mof_candidates": "أفضل مرشّحي MOF (من نموذج الدمج)",
        "mof_feature_space_title": "تمثيل ثلاثي الأبعاد لفضاء خصائص الـ MOF",
        "full_mof_df_expander": "عرض جدول MOF الكامل",

        # Sections – Plants
        "top_plant_candidates": "أفضل المرشّحين من النباتات",
        "plant_feature_space_title": "تمثيل ثلاثي الأبعاد لفضاء خصائص النباتات",

        # LLM prompt/response
        "llm_prompt_title": "نص الطلب المرسَل إلى نموذج اللغة",
        "llm_raw_response_title": "استجابة نموذج اللغة",
        "llm_prompt_expander": "نص الطلب (Prompt)",
        "llm_raw_response_expander": "استجابة نموذج اللغة + كود JSON",

        "hybrid_json_parse_fail": (
            "تعذّر تفسير كائن JSON من استجابة نموذج اللغة. "
            "قد يحدث ذلك إذا لم يتبع النموذج المخطط المطلوب بدقة."
        ),

        # Hybrid MOFs explanation
        "hybrid_section_title": "مواد هجينة مستوحاة من النباتات و-MOF",
        "key_features_title": "السمات الرئيسية:",
        "release_section_title": "طريقة مقترحة لتحرير / تجديد CO₂:",
        "release_method_label": "الطريقة",
        "release_rationale_label": "المبرر",
        "inspiration_expander_title": "مصادر الإلهام لـ",

        "mof_subsection_title": "مواد MOF",
        "plant_subsection_title": "النباتات",
        "plant_enzymes_title": "بُنى الإنزيمات المرتبطة بمسار البناء الضوئي",
        "plant_justifications_title": "مبررات اختيار النباتات:",
        "mof_justifications_title": "مبررات اختيار مواد MOF:",

        # Cheminformatics
        "cheminfo_title": "التحقق الكيميائي-المعلوماتي",
        "cheminfo_no_rdkit": (
            "مكتبة RDKit غير متاحة في هذه البيئة. "
            "تم تعطيل التحقق الكيميائي-المعلوماتي."
        ),
        "cheminfo_no_linkers": "لا توجد أي قيم `linker_smiles` في هذه الفكرة الهجينة.",
        "cheminfo_invalid_smiles": "SMILES غير صالح وفقاً لـ RDKit.",
        "cheminfo_errors": "أخطاء:",
        "cheminfo_warnings": "تحذيرات:",
        "cheminfo_3d_unavailable": "_لا تتوفر تمثيلات ثلاثية الأبعاد لهذا الرابط (مشكلة في RDKit أو py3Dmol)._",

        # 3D MOF block
        "mof_3d_title": "البنية ثلاثية الأبعاد لمادة MOF",
        "style_label": "نمط العرض",
        "show_surface_label": "إظهار السطح",
        "legend_expander": "وسيلة الإيضاح ومعلومات الفلزات",
        "legend_title": "وسيلة إيضاح لألوان الذرات (إعدادات 3Dmol الافتراضية):",
        "legend_metals_label": "مراكز الفلزات المكتشفة:",

        # 3D feature-space warnings
        "mof_3d_no_data": "لا توجد بيانات كافية لـ MOF لعرض مخطط ثلاثي الأبعاد.",
        "mof_3d_not_enough_numeric": (
            "تعذّر عرض مخطط ثلاثي الأبعاد لفضاء خصائص MOF: "
            "لا توجد أعمدة رقمية كافية."
        ),
        "mof_3d_all_nan": (
            "تعذّر عرض مخطط ثلاثي الأبعاد: الأعمدة الرقمية موجودة "
            "ولكن جميع القيم تحتوي على NaN."
        ),

        "plant_3d_missing_cols": "تعذّر عرض مخطط ثلاثي الأبعاد لخصائص النباتات. الأعمدة المفقودة: ",

        # Closed-loop
        "closed_loop_title": "🌀 حلقة تحسين مغلقة (لمواد MOF فقط)",
        "closed_loop_desc": (
            "استخدم نفس هدف التصميم المجمّع، لكن اسمح لنموذج اللغة أن يضبط أوزان "
            "الأهداف المتعددة (الألفة، المساحة السطحية، المسامية، عقوبة الكثافة) "
            "ويعيد ترتيب مواد MOF عبر عدة دورات."
        ),
        "closed_loop_final_weights": "الأوزان النهائية للأهداف المتعددة:",
        "closed_loop_weight_evolution": "تطور الأوزان عبر الدورات:",
        "closed_loop_top_mofs": "أفضل مواد MOF بعد الدورة الأخيرة:",
        "closed_loop_full_table_expander": "جدول مواد MOF الكامل في الدورة الأخيرة",
        "closed_loop_raw_llm_expander": "استجابات نموذج اللغة عبر الدورات",

        # Generic
        "no_mofs_for_ids": "_لا توجد مواد MOF مطابقة لهذه المعرفات._",
        "no_plants_for_ids": "_لا توجد نباتات مطابقة لهذه المعرفات._",
        "no_enzyme_mapping": "_لا يوجد ربط إنزيمي متاح لمسار البناء الضوئي / النمط الفرعي._",
        "no_physio_data": "لا توجد بيانات فيزيولوجية متاحة.",
    },
}


def t(key: str) -> str:
    """Simple translation helper."""
    lang = st.session_state.get("lang", "en")
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, TRANSLATIONS["en"].get(key, key))


# ==========================================
#  TABLE RENDERING HELPER
# ==========================================

def show_table(df: pd.DataFrame, cols=None):
    """Render a table with nice numeric formatting and full width."""
    if cols is not None:
        df = df[cols]

    df_local = df.copy()
    num_cols = df_local.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        fmt_map = {col: "{:.3g}" for col in num_cols}
        styler = df_local.style.format(fmt_map)
        st.dataframe(styler, use_container_width=True)
    else:
        st.dataframe(df_local, use_container_width=True)


# ==========================================
#  JUSTIFICATION HELPERS
# ==========================================

def summarize_mof_row(row: dict) -> str:
    kh = row.get("kh_class_name", "unknown")
    asa = row.get("asa_m2_g", None)
    vf = row.get("void_fraction", None)
    dens = row.get("density_g_cm3", None)

    bits = [f"KH class: **{kh}**"]

    if asa is not None:
        try:
            asa_val = float(asa)
            if asa_val > 3000:
                bits.append(f"very high surface area (~{asa_val:.0f} m²/g)")
            elif asa_val > 1500:
                bits.append(f"high surface area (~{asa_val:.0f} m²/g)")
            else:
                bits.append(f"moderate surface area (~{asa_val:.0f} m²/g)")
        except Exception:
            pass

    if vf is not None:
        try:
            vf_val = float(vf)
            if vf_val > 0.8:
                bits.append(f"very high void fraction ({vf_val:.2f})")
            elif vf_val > 0.5:
                bits.append(f"high void fraction ({vf_val:.2f})")
            else:
                bits.append(f"more compact pores ({vf_val:.2f})")
        except Exception:
            pass

    if dens is not None:
        try:
            d_val = float(dens)
            if d_val > 1.5:
                bits.append(f"relatively dense framework ({d_val:.2f} g/cm³)")
            else:
                bits.append(f"moderate density ({d_val:.2f} g/cm³)")
        except Exception:
            pass

    return "; ".join(bits)


def summarize_plant_row(row: dict) -> str:
    A = row.get("predicted_A", None)
    Ci = row.get("Ci", None)
    temp = row.get("Temp", None)
    qobs = row.get("Qobs", None)

    bits = []

    if A is not None:
        try:
            A_val = float(A)
            if A_val > 20:
                bits.append(f"very high CO₂ assimilation (A ≈ {A_val:.1f})")
            elif A_val > 10:
                bits.append(f"high CO₂ assimilation (A ≈ {A_val:.1f})")
            else:
                bits.append(f"moderate assimilation (A ≈ {A_val:.1f})")
        except Exception:
            pass

    if Ci is not None:
        try:
            bits.append(f"internal CO₂ Ci ≈ {float(Ci):.0f} ppm")
        except Exception:
            pass

    if temp is not None:
        try:
            bits.append(f"measured at ~{float(temp):.1f} °C")
        except Exception:
            pass

    if qobs is not None:
        try:
            bits.append(f"under light intensity Qobs ≈ {float(qobs):.0f}")
        except Exception:
            pass

    return "; ".join(bits) if bits else t("no_physio_data")


# ==========================================
#  CHEMINFORMATICS HELPER
# ==========================================

def render_hybrid_cheminfo(idea: dict):
    """Show cheminformatics validation for a hybrid idea."""
    st.markdown(f"#### {t('cheminfo_title')}")

    if not smiles_is_available():
        st.info(t("cheminfo_no_rdkit"))
        return

    chem_res = validate_hybrid_idea_chemistry(idea)

    linkers = chem_res["linkers"]
    metals = chem_res["metals"]

    if metals:
        st.markdown(f"**Target metal centers:** {', '.join(metals)}")
    else:
        st.markdown("**Target metal centers:** not specified")

    if not linkers:
        st.write(t("cheminfo_no_linkers"))
        return

    for idx, link in enumerate(linkers, start=1):
        smi = link["smiles"]
        val = link["validation"]
        st.markdown(f"**Linker {idx}: `{smi}`**")

        if not val["is_valid"]:
            st.error(t("cheminfo_invalid_smiles"))
            if val["errors"]:
                st.write(t("cheminfo_errors"))
                for e in val["errors"]:
                    st.write(f"- {e}")
            continue

        formula_raw = val.get("formula")
        if formula_raw:
            st.write(
                f"- Formula: `{formula_raw}`; MW ≈ {val['mol_weight']:.2f} g/mol; "
                f"atoms: {val['n_atoms']}"
            )
        else:
            st.write(
                f"- MW ≈ {val['mol_weight']:.2f} g/mol; atoms: {val['n_atoms']}"
            )

        if val["warnings"]:
            st.write(t("cheminfo_warnings"))
            for w in val["warnings"]:
                st.write(f"- {w}")

        img = make_linker_image(smi, size=200)
        if img is not None:
            st.image(img, caption=f"Linker {idx} – 2D", use_container_width=False)

        html3d = make_linker_3d_html(smi, style="stick")
        if html3d is not None:
            st.components.v1.html(html3d, height=320)
        else:
            st.write(t("cheminfo_3d_unavailable"))


# ==========================================
#  3D VISUALIZATION – MOFs
# ==========================================

def show_mof_3d_block(mof_id: str, key_suffix: str):
    """3D visualization block for a single MOF."""
    st.markdown(f"**{t('mof_3d_title')} `{mof_id}`**")

    cif_path = CIF_DIR / f"{mof_id}.cif"
    if not cif_path.exists():
        st.warning(f"No CIF file found for MOF `{mof_id}` at {cif_path}")
        return

    style = st.radio(
        t("style_label"),
        options=["stick", "sphere", "line"],
        index=0,
        horizontal=True,
        key=f"{key_suffix}_style",
    )

    show_surface = st.checkbox(
        t("show_surface_label"),
        value=False,
        key=f"{key_suffix}_surface",
    )

    try:
        view = cif_to_3d_view(str(cif_path), style=style, surface=show_surface)
        html = view._make_html()
        st.components.v1.html(html, height=420)
    except Exception as e:
        st.error(f"Could not render 3D structure for {mof_id}: {e}")
        return

    metals = detect_metal_elements(str(cif_path))
    metal_text = ", ".join(metals) if metals else "Not detected / organic-only"

    with st.expander(t("legend_expander")):
        st.markdown(f"**{t('legend_title')}**")
        st.markdown(
            "- **C**: gray\n"
            "- **H**: white\n"
            "- **O**: red\n"
            "- **N**: blue\n"
            "- **Cl/F**: green\n"
            "- **S**: yellow\n"
            "- **Metals**: usually blue / cyan / purple depending on element\n"
        )
        st.markdown(f"**{t('legend_metals_label')}** `{metal_text}`")


# ==========================================
#  3D VISUALIZATION – PLANTS (TRAIT SPACE)
# ==========================================

def show_plant_3d_feature_space(df_plants: pd.DataFrame, key: str):
    """
    3D scatter of plant traits.

    Preferred axes:
      x = Ci
      y = Temp
      z = predicted_A

    If those are not all available, fall back to three numeric columns
    (trying to keep predicted_A if possible).

    Color:
      - Pathway if present (C3 / C4 / CAM)
      - else Subtype if present
    """
    if df_plants is None or df_plants.empty:
        st.warning("No plant data available for 3D plotting.")
        return

    df_plot = df_plants.copy()

    # Try preferred columns
    x_col, y_col, z_col = None, None, None

    if "Ci" in df_plot.columns:
        x_col = "Ci"
    if "Temp" in df_plot.columns:
        y_col = "Temp"
    if "predicted_A" in df_plot.columns:
        z_col = "predicted_A"

    # If any of the preferred triplet is missing, fall back to numeric cols
    if not (x_col and y_col and z_col):
        numeric_cols = df_plot.select_dtypes(include=[np.number]).columns.tolist()

        # Try to keep predicted_A if present
        if "predicted_A" in numeric_cols:
            z_col = "predicted_A"
            remaining = [c for c in numeric_cols if c != "predicted_A"]
            if len(remaining) >= 2:
                x_col, y_col = remaining[:2]
            elif len(remaining) == 1:
                x_col, y_col = remaining[0], remaining[0]
            else:
                # Only predicted_A is numeric -> cannot make 3D
                st.warning(
                    "Not enough numeric columns to build a 3D plant feature plot."
                )
                return
        else:
            # No predicted_A, just take the first three numeric columns
            if len(numeric_cols) < 3:
                st.warning(
                    "Not enough numeric columns to build a 3D plant feature plot."
                )
                return
            x_col, y_col, z_col = numeric_cols[:3]

    # Ensure numeric and drop NaNs
    for c in [x_col, y_col, z_col]:
        df_plot[c] = pd.to_numeric(df_plot[c], errors="coerce")

    df_sub = df_plot.dropna(subset=[x_col, y_col, z_col]).copy()
    if df_sub.empty:
        st.warning(
            "No plant rows with complete numeric data for the chosen axes "
            f"({x_col}, {y_col}, {z_col})."
        )
        return

    # Choose color column
    color_col = None
    if "Pathway" in df_sub.columns:
        color_col = "Pathway"
    elif "Subtype" in df_sub.columns:
        color_col = "Subtype"

    hover_cols = []
    for c in ["plant_id", "name", "Pathway", "Subtype", "dataset_origin"]:
        if c in df_sub.columns:
            hover_cols.append(c)

    fig = px.scatter_3d(
        df_sub,
        x=x_col,
        y=y_col,
        z=z_col,
        color=color_col if color_col else None,
        hover_data=hover_cols,
    )
    fig.update_layout(
        title="3D trait-space of top plant candidates",
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col,
        ),
        height=600,
    )
    st.plotly_chart(fig, use_container_width=True, key=key)



def show_mof_3d_feature_space(df_mofs: pd.DataFrame, key: str):
    if df_mofs is None or df_mofs.empty:
        st.warning(t("mof_3d_no_data"))
        return

    df_plot = df_mofs.copy()

    asa_col = None
    for c in df_plot.columns:
        cl = c.lower()
        if "asa" in cl or "surface" in cl:
            asa_col = c
            break

    void_col = None
    for c in df_plot.columns:
        cl = c.lower()
        if "void" in cl or "pore" in cl or "porosity" in cl:
            void_col = c
            break

    score_col = "score" if "score" in df_plot.columns else None
    color_col = "kh_class_name" if "kh_class_name" in df_plot.columns else None

    for c in [asa_col, void_col, score_col]:
        if c is not None and c in df_plot.columns:
            df_plot[c] = pd.to_numeric(df_plot[c], errors="coerce")

    if asa_col and void_col and score_col:
        df_sub = df_plot.dropna(subset=[asa_col, void_col, score_col]).copy()
        if df_sub.empty:
            st.warning(
                "No MOF rows with complete surface-area, void-fraction, and score "
                "for 3D plotting."
            )
            return

        hover_cols = ["mof_id"] if "mof_id" in df_sub.columns else []
        for extra in ["chemical_formula", "doi"]:
            if extra in df_sub.columns:
                hover_cols.append(extra)

        fig = px.scatter_3d(
            df_sub,
            x=asa_col,
            y=void_col,
            z=score_col,
            color=color_col if color_col in df_sub.columns else None,
            hover_data=hover_cols,
        )
        fig.update_layout(
            title=t("mof_feature_space_title"),
            scene=dict(
                xaxis_title=f"{asa_col}",
                yaxis_title=f"{void_col}",
                zaxis_title=f"{score_col}",
            ),
            height=600,
        )
        st.plotly_chart(fig, use_container_width=True, key=key)
        return

    numeric_cols = df_plot.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 3:
        st.warning(t("mof_3d_not_enough_numeric"))
        return

    if "score" in numeric_cols:
        x_col = "score"
        remaining = [c for c in numeric_cols if c != "score"]
        if len(remaining) >= 2:
            y_col, z_col = remaining[:2]
        else:
            y_col, z_col = remaining[0], remaining[0]
    else:
        x_col, y_col, z_col = numeric_cols[:3]

    df_sub = df_plot.dropna(subset=[x_col, y_col, z_col]).copy()
    if df_sub.empty:
        st.warning(t("mof_3d_all_nan"))
        return

    hover_cols = ["mof_id"] if "mof_id" in df_sub.columns else []

    fig = px.scatter_3d(
        df_sub,
        x=x_col,
        y=y_col,
        z=z_col,
        color=color_col if color_col in df_sub.columns else None,
        hover_data=hover_cols,
    )
    fig.update_layout(
        title=t("mof_feature_space_title"),
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col,
        ),
        height=600,
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


# ==========================================
#  MOF-only agent results UI
# ==========================================

def render_mof_agent_results(result):
    st.subheader(t("top_mof_candidates"))
    df = result["candidates_df"].copy()

    cols = [
        c
        for c in [
            "mof_id",
            "kh_class_name",
            "score",
            "density_g_cm3",
            "asa_m2_g",
            "void_fraction",
            "chemical_formula",
            "doi",
        ]
        if c in df.columns
    ]
    show_table(df, cols)

    st.subheader(t("mof_feature_space_title"))
    show_mof_3d_feature_space(df, key="mof_feature_space_mof_only")

    with st.expander(t("full_mof_df_expander")):
        show_table(df)

    st.subheader(t("llm_prompt_title"))
    st.text_area(t("llm_prompt_title"), result["prompt"], height=250)

    st.subheader(t("llm_raw_response_title"))
    st.write(result["llm_response"])


# ==========================================
#  Hybrid (MOF + Plant) agent UI
# ==========================================

def render_hybrid_agent_results(result):
    df_mofs = result["mofs_df"].copy()
    df_plants = result["plants_df"]

    st.subheader(t("top_mof_candidates"))
    cols_m = [
        c
        for c in [
            "mof_id",
            "kh_class_name",
            "score",
            "density_g_cm3",
            "asa_m2_g",
            "void_fraction",
            "chemical_formula",
            "doi",
        ]
        if c in df_mofs.columns
    ]
    show_table(df_mofs, cols_m)

    with st.expander(t("mof_feature_space_title")):
        show_mof_3d_feature_space(df_mofs, key="mof_feature_space_hybrid")

    with st.expander(t("full_mof_df_expander")):
        show_table(df_mofs)

    st.subheader(t("top_plant_candidates"))
    cols_p = [
        c
        for c in [
            "plant_id",
            "name",
            "Pathway",
            "Subtype",
            "predicted_A",
            "plant_utility",
            "Ci",
            "Temp",
            "Qobs",
            "dataset_origin",
        ]
        if c in df_plants.columns
    ]
    show_table(df_plants, cols_p)

    with st.expander(t("plant_feature_space_title")):
        show_plant_3d_feature_space(df_plants, key="plant_feature_space_hybrid")

    st.markdown("---")

    hybrid_json = parse_hybrid_json(result["llm_response"])
    if not hybrid_json:
        st.info(t("hybrid_json_parse_fail"))
        return

    enriched = link_hybrid_ideas_to_data(
        hybrid_json,
        df_mofs=df_mofs,
        df_plants=df_plants,
    )

    st.subheader(t("hybrid_section_title"))
    for idea_idx, idea in enumerate(enriched):
        st.markdown(f"### {idea['hybrid_id']}")
        st.write(idea["concept"])

        st.write(f"**{t('key_features_title')}**")
        for kf in idea["key_features"]:
            st.write(f"- {kf}")

        release_method = idea.get("release_method", "") or ""
        release_rationale = idea.get("release_rationale", "") or ""

        if release_method or release_rationale:
            st.markdown(f"**{t('release_section_title')}**")
            if release_method:
                st.markdown(f"- {t('release_method_label')}: **{release_method}**")
            if release_rationale:
                st.markdown(f"- {t('release_rationale_label')}: {release_rationale}")

        render_hybrid_cheminfo(idea)

        with st.expander(f"{t('inspiration_expander_title')} {idea['hybrid_id']}"):
            # MOFs
            st.markdown(f"#### {t('mof_subsection_title')}")
            if not idea["mofs"].empty:
                show_table(idea["mofs"], cols_m)

                for mof_idx, (_, row) in enumerate(idea["mofs"].iterrows()):
                    mof_id = str(row["mof_id"])
                    key_suffix = f"{idea['hybrid_id']}_mof_{mof_idx}"
                    show_mof_3d_block(mof_id, key_suffix=key_suffix)

                st.write(f"**{t('mof_justifications_title')}**")
                for _, row in idea["mofs"].iterrows():
                    rdict = row.to_dict()
                    mof_id = str(rdict.get("mof_id"))

                    formula_raw = (
                        rdict.get("chemical_formula")
                        or rdict.get("formula")
                        or rdict.get("ChemicalFormula")
                        or rdict.get("chem_formula")
                    )
                    formula_str = formula_raw if isinstance(formula_raw, str) and formula_raw.strip() else None

                    summary = summarize_mof_row(rdict)
                    doi_val = rdict.get("doi") or rdict.get("DOI")

                    if formula_str:
                        line = f"- **{mof_id}** (`{formula_str}`): {summary}"
                    else:
                        line = f"- **{mof_id}**: {summary}"

                    st.markdown(line)

                    if isinstance(doi_val, str) and doi_val.strip():
                        st.markdown(
                            f"  - Literature: DOI [{doi_val}](https://doi.org/{doi_val})"
                        )
            else:
                st.write(t("no_mofs_for_ids"))

            # Plants + enzymes
            st.markdown(f"#### {t('plant_subsection_title')}")
            if not idea["plants"].empty:
                show_table(idea["plants"], cols_p)

                for _, row in idea["plants"].iterrows():
                    pid = str(row.get("plant_id"))
                    name = str(row.get("name", "unknown"))
                    st.markdown(f"**Plant {pid} – {name}**")

                    enzymes = enzymes_for_plant_row(row)
                    if not enzymes:
                        st.write(t("no_enzyme_mapping"))
                    else:
                        st.markdown(f"**{t('plant_enzymes_title')}**")
                        for e_idx, enzyme in enumerate(enzymes, start=1):
                            st.markdown(f"- **{enzyme.name}** – {enzyme.role}")

                            if enzyme.structure_type == "protein" and enzyme.pdb_path:
                                try:
                                    html = enzyme_pdb_to_html(enzyme.pdb_path, style="cartoon")
                                    st.components.v1.html(html, height=420)
                                except Exception as ex:
                                    st.write(f"_Could not render 3D structure for {enzyme.name}: {ex}_")
                            elif enzyme.structure_type == "ligand" and enzyme.smiles:
                                img = make_linker_image(enzyme.smiles, size=200)
                                if img is not None:
                                    st.image(img, caption=f"{enzyme.name} – 2D", use_container_width=False)
                                html3d = make_linker_3d_html(enzyme.smiles, style="stick")
                                if html3d is not None:
                                    st.components.v1.html(html3d, height=320)

                st.write(f"**{t('plant_justifications_title')}**")
                for _, row in idea["plants"].iterrows():
                    rdict = row.to_dict()
                    pid = str(rdict.get("plant_id"))
                    name = str(rdict.get("name", "unknown"))
                    summary = summarize_plant_row(rdict)
                    st.markdown(f"- **{pid} – {name}**: {summary}")
            else:
                st.write(t("no_plants_for_ids"))

        with st.expander(t("llm_raw_response_expander")):
            st.write(result["llm_response"])

        st.markdown("---")

def render_backend_analysis_plots(result):
    """
    Backend analysis of the hybrid agent output:
      - MOF score histogram
      - MOF property vs score
      - Plant predicted_A histogram
      - Plant predicted_A by pathway/subtype
      - Hybrid-level MOF score summary

    Everything is rendered as 2D matplotlib plots inside Streamlit,
    with download buttons for each figure.
    """
    st.subheader("📊 Backend analysis plots")

    df_mofs = result.get("mofs_df")
    df_plants = result.get("plants_df")
    llm_response = result.get("llm_response", "")

    # -----------------------
    # 1) MOF score histogram
    # -----------------------
    if df_mofs is not None and not df_mofs.empty and "score" in df_mofs.columns:
        st.markdown("### MOF score distribution")

        scores = pd.to_numeric(df_mofs["score"], errors="coerce").dropna()
        if len(scores) > 0:
            fig, ax = plt.subplots()
            ax.hist(scores, bins=10, edgecolor="black")
            ax.set_xlabel("Model score")
            ax.set_ylabel("Count")
            ax.set_title("Histogram of MOF model scores")
            fig.tight_layout()
            st.pyplot(fig)

            # Download button
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
            buf.seek(0)
            st.download_button(
                label="⬇️ Download MOF score histogram (PNG)",
                data=buf,
                file_name="mof_score_histogram.png",
                mime="image/png",
                key="dl_mof_score_hist",
            )
        else:
            st.info("No valid numeric MOF scores to plot.")

    # -----------------------
    # 2) MOF property vs score (ASA vs score if available)
    # -----------------------
    asa_col = None
    if df_mofs is not None and not df_mofs.empty:
        for c in df_mofs.columns:
            if "asa" in c.lower() or "surface" in c.lower():
                asa_col = c
                break

    if (
        df_mofs is not None
        and not df_mofs.empty
        and asa_col is not None
        and "score" in df_mofs.columns
    ):
        st.markdown(f"### MOF {asa_col} vs model score")

        df_tmp = df_mofs[[asa_col, "score"]].copy()
        df_tmp[asa_col] = pd.to_numeric(df_tmp[asa_col], errors="coerce")
        df_tmp["score"] = pd.to_numeric(df_tmp["score"], errors="coerce")
        df_tmp = df_tmp.dropna()

        if not df_tmp.empty:
            fig, ax = plt.subplots()
            ax.scatter(df_tmp[asa_col], df_tmp["score"], alpha=0.7)
            ax.set_xlabel(asa_col)
            ax.set_ylabel("Model score")
            ax.set_title(f"{asa_col} vs model score")
            fig.tight_layout()
            st.pyplot(fig)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
            buf.seek(0)
            st.download_button(
                label=f"⬇️ Download {asa_col} vs score plot (PNG)",
                data=buf,
                file_name=f"mof_{asa_col}_vs_score.png",
                mime="image/png",
                key="dl_mof_asa_vs_score",
            )
        else:
            st.info(f"No valid numeric data for {asa_col} vs score.")

    # -----------------------
    # 3) Plant predicted_A histogram
    # -----------------------
    if df_plants is not None and not df_plants.empty and "predicted_A" in df_plants.columns:
        st.markdown("### Plant predicted CO₂ assimilation (A)")

        A_vals = pd.to_numeric(df_plants["predicted_A"], errors="coerce").dropna()
        if len(A_vals) > 0:
            fig, ax = plt.subplots()
            ax.hist(A_vals, bins=10, edgecolor="black")
            ax.set_xlabel("predicted A (µmol m⁻² s⁻¹)")
            ax.set_ylabel("Count")
            ax.set_title("Histogram of predicted CO₂ assimilation")
            fig.tight_layout()
            st.pyplot(fig)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
            buf.seek(0)
            st.download_button(
                label="⬇️ Download plant predicted A histogram (PNG)",
                data=buf,
                file_name="plant_predictedA_histogram.png",
                mime="image/png",
                key="dl_plant_predictedA_hist",
            )
        else:
            st.info("No valid predicted_A values to plot for plants.")

    # -----------------------
    # 4) Plant subtype / pathway vs predicted_A
    # -----------------------
    group_col = None
    if df_plants is not None and not df_plants.empty:
        if "Pathway" in df_plants.columns:
            group_col = "Pathway"
        elif "Subtype" in df_plants.columns:
            group_col = "Subtype"

    if (
        df_plants is not None
        and not df_plants.empty
        and "predicted_A" in df_plants.columns
        and group_col is not None
    ):
        st.markdown(f"### Predicted A by plant {group_col}")

        df_tmp = df_plants[[group_col, "predicted_A"]].copy()
        df_tmp["predicted_A"] = pd.to_numeric(df_tmp["predicted_A"], errors="coerce")
        df_tmp = df_tmp.dropna()

        if not df_tmp.empty:
            groups = df_tmp[group_col].unique().tolist()
            fig, ax = plt.subplots()
            data = [df_tmp[df_tmp[group_col] == g]["predicted_A"].values for g in groups]
            ax.boxplot(data, labels=groups, showmeans=True)
            ax.set_xlabel(group_col)
            ax.set_ylabel("predicted A (µmol m⁻² s⁻¹)")
            ax.set_title(f"Distribution of predicted A by {group_col}")
            fig.tight_layout()
            st.pyplot(fig)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
            buf.seek(0)
            st.download_button(
                label=f"⬇️ Download predicted A by {group_col} plot (PNG)",
                data=buf,
                file_name=f"plant_predictedA_by_{group_col}.png",
                mime="image/png",
                key="dl_plant_A_by_group",
            )
        else:
            st.info(f"No valid predicted_A values by {group_col} to plot.")

    # -----------------------
    # 5) Hybrid-level MOF score summary
    # -----------------------
    st.markdown("### Hybrid MOF score summary")

    hybrid_json = parse_hybrid_json(llm_response) if llm_response else None
    if not hybrid_json:
        st.info(
            "Could not parse hybrid JSON from LLM response for hybrid-level plots."
        )
        return

    enriched = link_hybrid_ideas_to_data(
        hybrid_json,
        df_mofs=df_mofs if df_mofs is not None else pd.DataFrame(),
        df_plants=df_plants if df_plants is not None else pd.DataFrame(),
    )

    rows = []
    for idea in enriched:
        df_m = idea.get("mofs", pd.DataFrame())
        if df_m is not None and not df_m.empty and "score" in df_m.columns:
            scores = pd.to_numeric(df_m["score"], errors="coerce").dropna()
            if len(scores) > 0:
                rows.append(
                    {
                        "hybrid_id": idea.get("hybrid_id", "UnknownHybrid"),
                        "avg_mof_score": scores.mean(),
                        "max_mof_score": scores.max(),
                    }
                )

    if rows:
        df_h = pd.DataFrame(rows)
        fig, ax = plt.subplots()
        ax.bar(df_h["hybrid_id"], df_h["avg_mof_score"], edgecolor="black")
        ax.set_xlabel("Hybrid ID")
        ax.set_ylabel("Average MOF score")
        ax.set_title("Average MOF score per hybrid concept")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        st.pyplot(fig)

        # Plot download
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)
        st.download_button(
            label="⬇️ Download hybrid MOF score plot (PNG)",
            data=buf,
            file_name="hybrid_mof_score_summary.png",
            mime="image/png",
            key="dl_hybrid_mof_score_plot",
        )

        # Table + CSV download
        with st.expander("Hybrid score table"):
            st.dataframe(df_h)

            csv_buf = io.StringIO()
            df_h.to_csv(csv_buf, index=False)
            st.download_button(
                label="⬇️ Download hybrid score table (CSV)",
                data=csv_buf.getvalue(),
                file_name="hybrid_mof_scores.csv",
                mime="text/csv",
                key="dl_hybrid_score_csv",
            )
    else:
        st.info(
            "No MOF scores available in the hybrid ideas for plotting hybrid-level metrics."
        )


# ==========================================
#  Closed-loop optimization UI
# ==========================================

def run_closed_loop_ui(goal: str, top_k_mofs: int, max_iters: int = 3):
    st.subheader(t("closed_loop_title"))

    result = run_closed_loop_optimization(
        goal=goal,
        max_iters=max_iters,
        top_k=top_k_mofs,
    )

    st.markdown(f"**{t('closed_loop_final_weights')}**")
    st.json(result["final_weights"])

    rows = []
    for it in result["iterations"]:
        w = it["weights"]
        rows.append(
            {
                "iteration": it["iteration"],
                "affinity": w.get("affinity", None),
                "asa": w.get("asa", None),
                "void_fraction": w.get("void_fraction", None),
                "density": w.get("density", None),
            }
        )

    st.markdown(f"**{t('closed_loop_weight_evolution')}**")
    show_table(pd.DataFrame(rows))

    last_iter = result["iterations"][-1]
    df_top = last_iter["top_mofs"]
    cols = [
        c
        for c in [
            "mof_id",
            "kh_class_name",
            "utility",
            "score",
            "density_g_cm3",
            "asa_m2_g",
            "void_fraction",
            "doi",
        ]
        if c in df_top.columns
    ]

    st.markdown(f"**{t('closed_loop_top_mofs')}**")
    show_table(df_top, cols)

    with st.expander(t("closed_loop_full_table_expander")):
        show_table(df_top)

    with st.expander(t("closed_loop_raw_llm_expander")):
        for it in result["iterations"]:
            st.markdown(f"### Iteration {it['iteration']}")
            st.write(it["llm_response"])
            st.markdown("---")


# ==========================================
#  MAIN STREAMLIT APP
# ==========================================

def main():
    # -----------------------------
    #  Language toggle (top-right)
    # -----------------------------
    if "lang" not in st.session_state:
        st.session_state["lang"] = "en"

    prev_lang = st.session_state["lang"]
    title_col, toggle_col = st.columns([4, 1])

    with toggle_col:
        is_ar = st.toggle(
            "العربية",
            value=(prev_lang == "ar"),
            help=t("lang_toggle_help"),
        )
    st.session_state["lang"] = "ar" if is_ar else "en"
    lang = st.session_state["lang"]

    # Direction & alignment
    if lang == "ar":
        st.markdown(
            """
            <style>
            html, body, [data-testid="stAppViewContainer"] * {
                direction: rtl;
                text-align: right;
            }
            [data-testid="stSidebar"] * {
                direction: rtl;
                text-align: right;
            }
            label, .stTextInput, .stSelectbox, .stSlider, .stTextArea {
                text-align: right;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            html, body, [data-testid="stAppViewContainer"] * {
                direction: ltr;
                text-align: left;
            }
            [data-testid="stSidebar"] * {
                direction: ltr;
                text-align: left;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    with title_col:
        st.title(t("app_title"))

    # Session state for results
    if "agent_result" not in st.session_state:
        st.session_state.agent_result = None
    if "agent_type" not in st.session_state:
        st.session_state.agent_type = None
    if "goal_text" not in st.session_state:
        st.session_state.goal_text = ""

    # Sidebar configuration
    st.sidebar.header(t("agent_config_title"))

    agent_type = st.sidebar.radio(
        t("agent_mode_label"),
        [
            t("agent_mode_hybrid"),
            t("agent_mode_mof_only"),
        ],
    )

    top_k_mofs = st.sidebar.slider(
        t("top_k_mofs_label"),
        min_value=5,
        max_value=50,
        value=15,
        step=5,
    )

    if agent_type == t("agent_mode_hybrid"):
        top_k_plants = st.sidebar.slider(
            t("top_k_plants_label"),
            min_value=5,
            max_value=50,
            value=10,
            step=5,
        )

        n_hybrids = st.sidebar.slider(
            t("n_hybrids_label"),
            min_value=1,
            max_value=6,
            value=2,
            step=1,
            help=t("n_hybrids_help"),
        )
    else:
        top_k_plants = None
        n_hybrids = None

    max_iters_closed = st.sidebar.slider(
        t("closed_loop_iters_label"),
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help=t("closed_loop_iters_help"),
    )

    # Prompt builder
    st.subheader(t("prompt_builder_title"))

    target_gas = "CO₂"
    st.markdown(f"**{t('target_gas_label')} {target_gas}**")

    # Application scenario options based on language
    app_options = TRANSLATIONS[lang]["application_scenario_options"]
    application_scenario = st.selectbox(
        t("application_scenario_label"),
        app_options,
        index=0,
        help=t("application_scenario_help"),
    )

    st.markdown(f"**{t('operating_conditions_title')}**")

    temp_min, temp_max = st.slider(
        "Temperature range (°C)" if lang == "en" else "نطاق درجة الحرارة (°م)",
        min_value=-20,
        max_value=200,
        value=(20, 60),
        step=5,
        help=t("temp_slider_help"),
    )

    pressure_min, pressure_max = st.slider(
        "Pressure range (bar)" if lang == "en" else "نطاق الضغط (بار)",
        min_value=0.01,
        max_value=100.0,
        value=(0.1, 1.0),
        step=0.05,
        help=t("pressure_slider_help"),
    )

    humidity_options = TRANSLATIONS[lang]["humidity_options"]
    humidity = st.selectbox(
        t("humidity_label"),
        humidity_options,
        index=1,
        help=t("humidity_help"),
    )

    st.markdown(f"**{t('design_priorities_title')}**")

    affinity_weight = st.slider(
        t("affinity_label"),
        min_value=0.0,
        max_value=1.0,
        value=0.8,
        step=0.1,
        help=t("affinity_help"),
    )

    capacity_weight = st.slider(
        t("capacity_label"),
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help=t("capacity_help"),
    )

    porosity_weight = st.slider(
        t("porosity_label"),
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.1,
        help=t("porosity_help"),
    )

    density_weight = st.slider(
        t("density_label"),
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help=t("density_help"),
    )

    stability_weight = st.slider(
        t("stability_label"),
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help=t("stability_help"),
    )

    extra_notes = st.text_area(
        t("extra_notes_label"),
        "",
        placeholder=t("extra_notes_placeholder"),
        height=120,
    )

    mandatory_ok = bool(application_scenario)

    if not mandatory_ok:
        st.warning(t("must_select_scenario"))

    goal_lines = [
        f"Target gas: {target_gas}",
        f"Application scenario: {application_scenario}",
        f"Operating temperature range: {temp_min}–{temp_max} °C",
        f"Operating pressure range: {pressure_min:.2f}–{pressure_max:.2f} bar",
        f"Humidity environment: {humidity}",
        "",
        "Design priorities (0–1 scale):",
        f"- Affinity / selectivity (KH class, model score): {affinity_weight:.2f}",
        f"- Capacity (surface area / pore volume): {capacity_weight:.2f}",
        f"- Porosity / void fraction: {porosity_weight:.2f}",
        f"- Low density preference: {density_weight:.2f}",
        f"- Stability / robustness: {stability_weight:.2f}",
    ]

    if agent_type == t("agent_mode_hybrid"):
        goal_lines.append(t("goal_header_hybrid"))
    else:
        goal_lines.append(t("goal_header_mof_only"))

    if extra_notes.strip():
        goal_lines.append("\nAdditional user design notes:")
        goal_lines.append(extra_notes.strip())

    goal = "\n".join(goal_lines)

    with st.expander(f"**{t('compiled_goal_title')}**"):
        st.code(goal, language="markdown")

    # Run agent
    if st.button(t("run_agent_btn")):
        if not mandatory_ok:
            st.error(t("cannot_run_agent"))
        else:
            with st.spinner("Running agent pipeline..."):
                try:
                    if agent_type == t("agent_mode_mof_only"):
                        result = run_llm_agent_once(goal, top_k=top_k_mofs)
                    else:
                        result = run_llm_hybrid_agent(
                            goal=goal,
                            top_k_mofs=top_k_mofs,
                            top_k_plants=top_k_plants,
                            n_hybrids=n_hybrids,
                        )
                    st.session_state.agent_result = result
                    st.session_state.agent_type = agent_type
                    st.session_state.goal_text = goal
                except Exception as e:
                    st.error(f"Error during agent execution: {e}")

    # Render results
    if st.session_state.agent_result is not None:
        st.markdown(f"## {t('results_title')}")
        if st.session_state.agent_type == t("agent_mode_mof_only"):
            render_mof_agent_results(st.session_state.agent_result)
        else:
            render_hybrid_agent_results(st.session_state.agent_result)

        # backend analysis section
        with st.expander("📈 Show backend analysis plots (MOFs, plants & hybrids)"):
            render_backend_analysis_plots(st.session_state.agent_result)


    # Closed-loop optimization
    st.subheader(t("closed_loop_title"))
    st.markdown(t("closed_loop_desc"))

    if st.button(t("run_closed_loop_btn")):
        if not mandatory_ok:
            st.error(t("cannot_run_closed_loop"))
        else:
            with st.spinner("Running closed-loop optimization..."):
                try:
                    run_closed_loop_ui(
                        goal=goal,
                        top_k_mofs=top_k_mofs,
                        max_iters=max_iters_closed,
                    )
                except Exception as e:
                    st.error(f"Error during closed-loop optimization: {e}")


if __name__ == "__main__":
    main()
