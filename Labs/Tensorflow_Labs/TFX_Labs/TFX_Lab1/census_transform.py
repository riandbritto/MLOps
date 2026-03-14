import tensorflow as tf
import tensorflow_transform as tft

import census_constants

# Unpack the contents of the constants module
_NUMERIC_FEATURE_KEYS = census_constants.NUMERIC_FEATURE_KEYS
_CATEGORICAL_FEATURE_KEYS = census_constants.CATEGORICAL_FEATURE_KEYS
_BUCKET_FEATURE_KEYS = census_constants.BUCKET_FEATURE_KEYS
_FEATURE_BUCKET_COUNT = census_constants.FEATURE_BUCKET_COUNT
_LABEL_KEY = census_constants.LABEL_KEY
_transformed_name = census_constants.transformed_name


# Define the transformations
def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.
    Args:
        inputs: map from feature keys to raw not-yet-transformed features.
    Returns:
        Map from string feature key to transformed feature operations.
    """
    outputs = {}

    # Scale all numeric features to the range [0,1]
    # Skip bucket features here since they are handled separately below
    for key in _NUMERIC_FEATURE_KEYS:
        if key not in _BUCKET_FEATURE_KEYS:
            outputs[_transformed_name(key)] = tft.scale_to_0_1(inputs[key])

    # Bucketize age and bmi
    for key in _BUCKET_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.bucketize(
            inputs[key], _FEATURE_BUCKET_COUNT[key])

    # No categorical features in diabetes dataset, so this loop is empty
    for key in _CATEGORICAL_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.compute_and_apply_vocabulary(inputs[key])

    # Target is continuous (regression) — scale it to [0,1]
    outputs[_transformed_name(_LABEL_KEY)] = tft.scale_to_0_1(inputs[_LABEL_KEY])

    return outputs
