#pragma once

#include <array>
#include <string_view>

namespace treesem
{
namespace model
{

inline constexpr std::array<std::string_view, 49> kPphFeatureNames{
    "Gestational_Age",
    "First_Stage_Labor_Min",
    "Second_Stage_Labor_Min",
    "Third_Stage_Labor",
    "Total_Labor_Time_Min",
    "Amniotic_Fluid_Volume",
    "Age",
    "Gravida",
    "Para",
    "Onset_Mode",
    "Membrane_Rupture_Method",
    "Initial_Amniotic_Fluid_Properties",
    "Initial_Amniotic_Fluid_Volume",
    "Fetal_Delivery_Method",
    "Placenta_Delivery_Method",
    "Placenta_Integrity",
    "Placenta_Weight",
    "Placenta_Length",
    "Placenta_Width",
    "Placenta_Thickness",
    "Membrane_Integrity",
    "Umbilical_Cord_Length",
    "Umbilical_Cord_Condition",
    "Umbilical_Cord_Torsion",
    "Placenta_Attachment",
    "Cervical_Condition",
    "Cervical_Outer_Sutures",
    "Cervical_Inner_Sutures",
    "Perineum",
    "Vaginal_Wall_Laceration",
    "Oxytocin",
    "Motherwort",
    "Sinmufai",
    "Carboprost",
    "Newborn_Sex",
    "Birth_Weight",
    "Head_Circumference",
    "Length",
    "Chest_Circumference",
    "Birth_Condition",
    "Breathing",
    "Outcome",
    "Height",
    "Weight",
    "BMI",
    "Education_Level",
    "Compound_Sodium_Chloride",
    "Vaginal_Exam_Count",
    "Intrapartum_Bleeding",
};

} // namespace model
} // namespace treesem
