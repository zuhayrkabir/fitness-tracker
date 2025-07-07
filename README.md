# 🏋️ AI Fitness Tracker: Exercise Classification & Repetition Counter

## 🌟 **Project Overview** 🌟
This advanced fitness tracking system uses machine learning to classify gym exercises and count repetitions with professional-grade accuracy. Processing raw accelerometer and gyroscope data from wearable devices (MetaMotion sensors), the system implements a complete pipeline from signal processing to real-time predictions.

Core Capabilities:
- Identifies 5 fundamental exercises: Bench Press, Squats, Deadlifts, Rows, and Overhead Press
- Counts repetitions with medical-grade precision
- Adapts to individual users' movement patterns
- Works with commercial wearable devices (tested with MetaMotion sensors)

## 🔥 Key Features
- **99.6% exercise recognition** (Random Forest on 5 exercise types)
- **Achieved a mean squared error (MSE) of 0.88** in predicting exercise repetition counts using accelerometer and gyroscope data
- **48 engineered features**:
  - Frequency-domain analysis (FFT with 1.429Hz resolution)
  - PCA-reduced dimensions (preserves 92% variance)
  - Temporal statistics (rolling means/stdevs)
- **Chauvenet's outlier detection** (7.2% data cleaned)

## 🛠️ Tech Stack
- **ML**: Scikit-learn (Random Forest, XGBoost, PCA)
- **Signal Processing**: FFT, Butterworth filters (0.4Hz cutoff)
- **Data**: Pandas/Numpy for time-series handling
- **Visualization**: Matplotlib/Seaborn
- **Development**: Python 3.8+

## ⚙️ **Processing Pipeline**
- Data Acquisition: 12.5kHz accelerometer + 25kHz gyroscope data
- Resampling: Downsampled to 200ms intervals (5Hz)
- Filtering: 4th order Butterworth low-pass (cutoff 1.2Hz)
- Feature Extraction: 3-stage feature engineering
- Model Serving: Real-time classification


## 🧑‍💻 Usage Examples
```
from src.models.train_model import FitnessTracker

# Initialize with pre-trained model
classifier = FitnessClassifier.load('models/rf_classifier.pkl')

# Process sensor data file
results = classifier.predict_from_csv('data/raw/session_1.csv')
print(results.summary())
```


## 🚀 Installation
```bash
git clone https://github.com/zuhayrkabir/fitness-tracker.git
cd fitness-tracker
pip install -r requirements.txt
