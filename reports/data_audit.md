# Data Audit

## 1.1 Counts

- Audio wavs (Actor_01..24): **1440** (expected 1440)
- Images total: **28709** (expected 28709)
- CSV shape: **(4000, 22)** (expected (4000, 22))

Per-emotion image counts:

|          |   actual |   expected |
|:---------|---------:|-----------:|
| Angry    |     3995 |       3995 |
| Disgust  |      436 |        436 |
| Fear     |     4097 |       4097 |
| Happy    |     7215 |       7215 |
| Neutral  |     4965 |       4965 |
| Sad      |     4830 |       4830 |
| Surprise |     3171 |       3171 |


Audio: 24 actors, 60–60 clips/actor (uniform 60).

## 1.2 Post-mapping stress-class distribution

|                 |   Audio |   Images |   CSV |
|:----------------|--------:|---------:|------:|
| Healthy         |     480 |    12180 |  1629 |
| Mild_Stress     |     384 |     8001 |  1237 |
| Moderate_Stress |     384 |     4533 |  1006 |
| Severe_Stress   |     192 |     3995 |   128 |

**Audio Severe = 192 clips** — binding constraint on the build.

## 1.3 Cross-modal mapping conflict

| emotion   | audio->         | image->         | status     |
|:----------|:----------------|:----------------|:-----------|
| neutral   | Healthy         | Healthy         | agree      |
| calm      | Healthy         | —               | audio-only |
| happy     | Healthy         | Healthy         | agree      |
| sad       | Mild_Stress     | Mild_Stress     | agree      |
| surprised | Mild_Stress     | Mild_Stress     | agree      |
| fearful   | Moderate_Stress | Moderate_Stress | agree      |
| angry     | Moderate_Stress | Severe_Stress   | CONFLICT   |
| disgust   | Severe_Stress   | Moderate_Stress | CONFLICT   |

Conflict confined to angry & disgust (the two highest classes), exact inversion:
- Severe: disgust voice + angry face
- Moderate audio pool 50% conflict-sourced; image pool ~9.6%.

## 1.4 CSV sanity

Class balance (normalised):

| Mental_Health_Status   |   frac |
|:-----------------------|-------:|
| Healthy                |  0.407 |
| Mild_Stress            |  0.309 |
| Moderate_Stress        |  0.252 |
| Severe_Stress          |  0.032 |

Total nulls: 0

Out-of-range values: none

Mean scores per class (ordinal justification — should rise monotonically):

| Mental_Health_Status   |   Depression_Score |   Anxiety_Score |   Stress_Score |
|:-----------------------|-------------------:|----------------:|---------------:|
| Healthy                |              9.95  |           9.596 |         12.799 |
| Mild_Stress            |             18.717 |          12.642 |         20.216 |
| Moderate_Stress        |             25.614 |          14.958 |         26.763 |
| Severe_Stress          |             30.664 |          19.734 |         33.812 |

Stress_Score monotonic across Healthy->Severe: **True** → ordinal treatment justified

Top feature↔Stress_Score correlations:

|                         |   corr |
|:------------------------|-------:|
| Head_Motion_Index       | -0.046 |
| Daily_App_Usage_Min     | -0.023 |
| Facial_Emotion_Variance | -0.018 |
| Typing_Speed_WPM        | -0.016 |
| Speech_Rate             |  0.015 |
| Skin_Temperature        |  0.011 |
| Eye_Blink_Rate          |  0.01  |
| Idle_Time_Min           | -0.01  |
