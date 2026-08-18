# Doctor prediction comparison

1. Resolve both records from session history or explicit IDs.
2. Use the comparison service for label, probability, confidence, cluster, leaf, path and feature deltas.
3. Retrieve explanations only when changed path/features need more context.
4. State model-version and serving-backend differences before interpreting deltas.
5. Do not infer clinical progression or causality from model changes alone.
6. Preserve exact deterministic values and both grounding prediction IDs.
