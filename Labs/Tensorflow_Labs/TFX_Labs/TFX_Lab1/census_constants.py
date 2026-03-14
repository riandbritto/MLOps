# Features with string data types that will be converted to indices
# Diabetes dataset has NO categorical features - all are float
CATEGORICAL_FEATURE_KEYS = []

# Numerical features that are marked as continuous
NUMERIC_FEATURE_KEYS = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

# Feature that can be grouped into buckets
BUCKET_FEATURE_KEYS = ['age', 'bmi']

# Number of buckets used by tf.transform for encoding each bucket feature.
FEATURE_BUCKET_COUNT = {'age': 4, 'bmi': 4}

# Feature that the model will predict
LABEL_KEY = 'target'

# Utility function for renaming the feature
def transformed_name(key):
    return key + '_xf'