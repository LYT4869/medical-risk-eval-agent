#pragma once

#include <string>

#include "model/ModelTypes.h"

namespace test
{

inline treesem::model::ModelResult validModelResult()
{
    treesem::model::ModelResult result;
    result.modelName = "treeSem";
    result.modelVersion = "pph-seed42-1a299a474ce5";
    result.servingBackend = "python_reference";
    result.dataset = "pph";
    result.inputSource = "pph_test_split";
    result.sampleIndex = 0;
    result.prediction = {0, 0.1, 0.9, 1, 0.08, 8};
    treesem::model::ImportantFeature important;
    important.index = 48;
    important.name = "Intrapartum_Bleeding";
    important.standardizedValue = -0.12;
    important.treeImportance = 0.94;
    important.displayName = "Intrapartum Bleeding";
    important.originalValue = 200.0;
    result.importantFeatures.push_back(important);

    treesem::model::DecisionPathStep branch;
    branch.nodeId = 0;
    branch.featureIndex = 48;
    branch.featureName = "Intrapartum_Bleeding";
    branch.comparisonOperator = "<=";
    branch.thresholdStandardized = 1.74;
    branch.valueStandardized = -0.12;
    branch.featureDisplayName = "Intrapartum Bleeding";
    branch.thresholdOriginal = 315.0;
    branch.valueOriginal = 200.0;
    result.decisionPath.push_back(branch);

    treesem::model::DecisionPathStep leaf;
    leaf.leafId = 8;
    result.decisionPath.push_back(leaf);
    return result;
}

inline std::string validAdapterResponse()
{
    return R"({
        "model":"treeSem",
        "model_version":"pph-seed42-1a299a474ce5",
        "serving_backend":"python_reference",
        "dataset":"pph",
        "input_source":"pph_test_split",
        "sample_index":0,
        "prediction":{
            "label":0,
            "positive_probability":0.1,
            "confidence":0.9,
            "cluster_id":1,
            "tree_probability":0.08,
            "tree_leaf_id":8
        },
        "important_features":[{
            "index":48,
            "name":"Intrapartum_Bleeding",
            "display_name":"Intrapartum Bleeding",
            "standardized_value":-0.12,
            "original_value":200.0,
            "unit":null,
            "tree_importance":0.94
        }],
        "decision_path":[{
            "node_id":0,
            "feature_index":48,
            "feature_name":"Intrapartum_Bleeding",
            "feature_display_name":"Intrapartum Bleeding",
            "operator":"<=",
            "threshold_standardized":1.74,
            "value_standardized":-0.12,
            "threshold_original":315.0,
            "value_original":200.0,
            "unit":null
        },{"leaf_id":8}]
    })";
}

} // namespace test
