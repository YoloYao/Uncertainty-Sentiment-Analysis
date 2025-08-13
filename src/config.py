import torch
import os
import json
import numpy as np

# The root path of the project
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_PATH, "data")
PROCESSED_DATA_PATH = os.path.join(DATA_PATH, "processed")
SRC_PATH = os.path.join(BASE_PATH, "src")
UQ_PARAMS_PATH = os.path.join(SRC_PATH, 'uq_params.json')
SAVED_MODELS_PATH = os.path.join(BASE_PATH, "saved_models")
SAVED_MODELS_NAME = 'best_model_state.bin'
SAVED_SNGP_MODELS_NAME = 'sngp_best_model_state.bin'
HUGGINGFACE_DATASET_NAME = 'stanfordnlp/sentiment140'
DATASET_FILE_NAME = 'sentiment140_50k.csv'
TRAIN_FILE_NAME = 'train.csv'
VALIDATION_FILE_NAME = 'validation.csv'
TEST_FILE_NAME = 'test.csv'

# Ensure that the directory where the model is saved exists
os.makedirs(SAVED_MODELS_PATH, exist_ok=True)

# --- Model and Tokenizer ---
MODEL_NAME = 'distilbert-base-uncased'
TOKENIZER = None  # Will be initialized in the training script

# --- Training Parameters ---
# Training Device
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 2e-5

# --- Dataset label information ---
# Label mapping, consistent with preprocessing results
# 0: negative, 1: neutral, 2: positive
CLASS_NAMES = ['negative', 'neutral', 'positive']
N_CLASSES = len(CLASS_NAMES)

# --- Quantitative method parameters ---
_UQ_PARAMS = {}


def _load_uq_params():
    """Load the contents of the uq-params.json file into the UQ-PAMS variable。"""
    global _UQ_PARAMS
    try:
        with open(UQ_PARAMS_PATH, 'r') as f:
            _UQ_PARAMS = json.load(f)
    except FileNotFoundError:
        print(
            f"Warning: UQ parameter file not found: {UQ_PARAMS_PATH}. Some functions may not be available.")
        _UQ_PARAMS = {}


def get_uq_param(param_name):
    """
    Get the value of the specified parameter

    :param param_name: Parameter Name。
    :return: Return None when value cannot be found
    """
    return _UQ_PARAMS.get(param_name)


def get_all_uq_params():
    """
    Get all parameter values of UQ methods

    :param param_name: Parameter Name。
    :return: Return None when value cannot be found
    """
    return _UQ_PARAMS


def update_uq_params(**kwargs):
    """
    Update parameter values and save UQ parameter file
    """
    global _UQ_PARAMS
    params_path = UQ_PARAMS_PATH

    # 1. Read existing parameters
    try:
        with open(params_path, 'r') as f:
            existing_params = json.load(f)
    except FileNotFoundError:
        existing_params = {}

    # 2. Update dictionary with newly passed parameters
    existing_params.update(kwargs)

    # 3. Convert Numpy numeric types to Python numeric types
    params_to_save = {}
    for key, value in existing_params.items():
        if isinstance(value, (np.float32, np.float64)):
            params_to_save[key] = float(value)
        elif isinstance(value, (np.int32, np.int64)):
            params_to_save[key] = int(value)
        else:
            params_to_save[key] = value

    # 4. Write the updated dictionary to a file
    with open(params_path, 'w') as f:
        json.dump(params_to_save, f, indent=4)

    print(f"UQ parameters have been successfully updated.")


# Automatically call the read parameter method when referencing the current file
_load_uq_params()


# Feature Calculation Dictionary
POSITIVE_WORDS = {'love', 'great', 'amazing',
                  'fantastic', 'good', 'best', 'happy', 'delicious'}
NEGATIVE_WORDS = {'bad', 'worst', 'terrible',
                  'boring', 'slow', 'hate', 'mess', 'disease'}
TRANSITIONAL_WORDS = {'but', 'however', 'although', 'though', 'despite', 'not'}


def _count_sentiment_words(text, sentiment_words):
    """
    Calculate the number of emotional words included
    """
    count = 0
    for word in str(text).split():
        if word in sentiment_words:
            count += 1
    return count


def _contains_transitional_words(text):
    """
    Does it contain transitional words
    """
    for word in str(text).split():
        if word in TRANSITIONAL_WORDS:
            return 1
    return 0


def _get_text_length(text):
    """
    Return text length
    """
    return len(str(text).split())


def analyze_features_by_group(df, uncertainty_col, group_name=""):
    """
    Receive a DataFrame, analyze and print features

    :param df: Must contain 'text', 'predicted_label', and the specified uncertainty_col column
    :param uncertainty_col: (str) The name of the uncertainty score column in the DataFrame
    :param group_name: (str) The analysis group name displayed in the report title
    """
    print("\n" + "="*20 +
          f" {group_name} Comparative analysis of features " + "="*20)

    # 1. Define thresholds and labels
    uncertainty_median = df[uncertainty_col].median()
    POSITIVE_LABEL = CLASS_NAMES.index('positive')
    NEGATIVE_LABEL = CLASS_NAMES.index('negative')

    # 2. Filter four subgroups
    certain_mask = df[uncertainty_col] <= uncertainty_median
    uncertain_mask = df[uncertainty_col] > uncertainty_median

    positive_mask = df['predicted_label'] == POSITIVE_LABEL
    negative_mask = df['predicted_label'] == NEGATIVE_LABEL

    certain_correct_positive = df[certain_mask & positive_mask]
    certain_correct_negative = df[certain_mask & negative_mask]
    uncertain_correct_positive = df[uncertain_mask & positive_mask]
    uncertain_correct_negative = df[uncertain_mask & negative_mask]

    all_groups = {
        f"Certain Positive ({group_name})": certain_correct_positive,
        f"Certain Negative ({group_name})": certain_correct_negative,
        f"Uncertain Positive ({group_name})": uncertain_correct_positive,
        f"Uncertain Negative ({group_name})": uncertain_correct_negative
    }

    # 3. Calculate features and print
    for name, group_df in all_groups.items():
        if not group_df.empty:
            pos_words = group_df['text'].apply(
                lambda x: _count_sentiment_words(x, POSITIVE_WORDS)).mean()
            neg_words = group_df['text'].apply(
                lambda x: _count_sentiment_words(x, NEGATIVE_WORDS)).mean()
            transitional_ratio = group_df['text'].apply(
                _contains_transitional_words).mean()
            text_len = group_df['text'].apply(_get_text_length).mean()

            print(
                f"\n--- Analysis Group: {name} (contains {len(group_df)} samples in total) ---")
            print(f"  - Average number of [Positive] words: {pos_words:.2f}")
            print(f"  - Average number of [Negative] words: {neg_words:.2f}")
            print(
                f"  - Proportion of transitional words included: {transitional_ratio:.2%}")
            print(f"  - Average text length: {text_len:.2f} words")
        else:
            print(
                f"\n--- Analysis Group: {name} (contains 0 samples in total) ---")

    print("="*(64 + len(group_name)))
