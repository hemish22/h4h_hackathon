"""Central config: paths, label maps, feature lists, mapping-conflict definition.

Single source of truth. Every other module imports from here so the mapping
tables, feature columns, and stress-class ordering are defined exactly once.
"""
from __future__ import annotations
import os

# ------------------------------------------------------------------ paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(ROOT, "dataset")
AUDIO_DIR = os.path.join(DATASET_DIR, "Audios")            # Actor_01 .. Actor_24
IMAGE_DIR = os.path.join(DATASET_DIR, "Extracted_images")  # 7 emotion folders
CSV_PATH = os.path.join(DATASET_DIR, "mental_health_multimodal.csv")

SPLITS_DIR = os.path.join(ROOT, "splits")
DATA_DIR = os.path.join(ROOT, "data")
REPORTS_DIR = os.path.join(ROOT, "reports")
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")
ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")

for _d in (SPLITS_DIR, DATA_DIR, REPORTS_DIR, RESULTS_DIR, FIGURES_DIR, ARTIFACTS_DIR):
    os.makedirs(_d, exist_ok=True)

# ------------------------------------------------------------------ classes
# Ordered scale — index IS the ordinal level. Never reorder.
STRESS_CLASSES = ["Healthy", "Mild_Stress", "Moderate_Stress", "Severe_Stress"]
CLASS_TO_ORD = {c: i for i, c in enumerate(STRESS_CLASSES)}
N_CLASSES = 4

# ------------------------------------------------------------------ audio filename codes (RAVDESS)
AUDIO_EMO_CODE = {1: "neutral", 2: "calm", 3: "happy", 4: "sad",
                  5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"}

# committee mapping table: AUDIO emotion -> stress class
AUDIO_EMO_TO_STRESS = {
    "neutral": "Healthy", "calm": "Healthy", "happy": "Healthy",
    "sad": "Mild_Stress", "surprised": "Mild_Stress",
    "fearful": "Moderate_Stress", "angry": "Moderate_Stress",
    "disgust": "Severe_Stress",
}

# ------------------------------------------------------------------ image folder codes (FER2013)
IMAGE_EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# committee mapping table: IMAGE emotion -> stress class
IMAGE_EMO_TO_STRESS = {
    "Happy": "Healthy", "Neutral": "Healthy",
    "Sad": "Mild_Stress", "Surprise": "Mild_Stress",
    "Fear": "Moderate_Stress", "Disgust": "Moderate_Stress",
    "Angry": "Severe_Stress",
}

# ------------------------------------------------------------------ THE cross-modal mapping conflict (plan §1.3)
# angry & disgust map to opposite stress levels across the two tables.
CONFLICT_EMOTIONS = {"angry", "disgust"}  # lowercase, checked against both modalities

# ------------------------------------------------------------------ tabular features
AUDIO_MATCH_COLS = ["MFCC_Mean", "MFCC_Variance", "Pitch_Mean", "Speech_Rate"]
IMAGE_MATCH_COLS = ["Facial_Emotion_Variance"]

FEATURE_COLS = [
    "Sleep_Quality", "Social_Engagement", "Daily_App_Usage_Min", "Typing_Speed_WPM",
    "Session_Frequency", "Idle_Time_Min", "Facial_Emotion_Variance", "Eye_Blink_Rate",
    "Smile_Intensity", "Head_Motion_Index", "MFCC_Mean", "MFCC_Variance",
    "Pitch_Mean", "Speech_Rate", "Heart_Rate_BPM", "HRV_Index",
    "Skin_Temperature", "GSR_Level",
]
assert len(FEATURE_COLS) == 18

TARGET_CAT = "Mental_Health_Status"
TARGET_REG = ["Depression_Score", "Anxiety_Score", "Stress_Score"]
REG_MAXIMA = [34, 24, 39]  # for [0,1] normalisation, invert before reporting

# physiologically-plausible clip ranges (plan §1.4)
CLIP_RANGES = {
    "Heart_Rate_BPM": (30, 220),
    "Skin_Temperature": (30, 40),
    "Depression_Score": (0, 34),
    "Anxiety_Score": (0, 24),
    "Stress_Score": (0, 39),
}

SEED = 42
