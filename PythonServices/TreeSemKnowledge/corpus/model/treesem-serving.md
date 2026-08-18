# treeSem Serving Model

## Purpose and output

treeSem is an auxiliary binary prediction and explanation model used by this project for a postpartum-haemorrhage demonstration. A prediction contains a label, positive-class probability, confidence, cluster identifier, decision-tree score, tree leaf identifier, important features, model version, and decision path. These values are model output rather than a clinical diagnosis.

## Serving path

The default serving path runs the deterministic neural subgraph with ONNX Runtime in C++. Preprocessing and the decision tree are implemented from the versioned Serving Bundle. The Python Bundle engine is the golden reference and fallback. The decoder, random sampling, dropout, and training-only state are excluded from serving.

## Input and preprocessing

The model input has exactly 49 numeric fields in the order recorded by `feature_schema.json`. Raw values are standardized as `(original - mean) / scale`; explanation values are converted back as `standardized * scale + mean`. Until an authoritative data dictionary is supplied, units and categorical encodings are metadata unavailable and must not be guessed.

## Explanation boundary

Important-feature ranking and a decision-tree path describe how this model processed a supplied input. They do not prove that a feature caused an outcome. A changed probability or path between two predictions is not by itself evidence that a patient's clinical condition improved or worsened.

## Versioning and integrity

The model version includes the trusted training artifact checksum. Bundle files, preprocessing metadata, tree topology, and ONNX input/output contracts are checksum-validated at startup. A corrupt or incompatible bundle fails before the server accepts traffic.

## Current quality boundary

Serving parity means the Python and C++ implementations reproduce the same existing model outputs. It does not establish clinical validity. Model-quality review, external validation, calibration, subgroup analysis, and clinical approval are separate workstreams.
