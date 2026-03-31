 🌿 Crop Disease Detector
A machine learning project that detects diseases in crops using a Random Forest Classifier trained on leaf and environmental features. Built as a capstone project for the Fundamentals of AI and ML course.
---
📌 Problem Statement
Crop diseases cause major agricultural losses every year, especially in rural India where early diagnosis is often unavailable. This project builds a lightweight ML model that can classify a crop as healthy or diseased based on observable leaf features and environmental conditions — no internet or image processing required.
---
🎯 Diseases Detected
Crop	Disease Classes
Tomato	Healthy, Early Blight, Late Blight
Corn	Healthy, Gray Leaf Spot, Northern Corn Leaf Blight
Rice	Healthy, Rice Blast, Brown Spot
Wheat	Healthy, Wheat Rust, Powdery Mildew
Potato	Healthy, Early Blight, Late Blight
---
📁 Project Structure
```
crop-disease-detector/
│
├── crop_disease_detector.py      # Main source code
├── crop_disease_dataset.csv      # Dataset (120 samples, 11 features)
├── README.md                     # This file
├── Project_Report.docx           # Full project report
│
└── outputs/                      # Auto-generated on first run
    ├── confusion_matrix.png
    ├── feature_importance.png
    ├── class_distribution.png
    └── accuracy_vs_estimators.png
```
---
⚙️ Setup & Installation
Prerequisites
Python 3.8 or above
pip
Step 1 — Clone the repository
```bash
git clone https://github.com/<your-username>/crop-disease-detector.git
cd crop-disease-detector
```
Step 2 — Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
Step 3 — Run the project
```bash
python crop_disease_detector.py
```
---
🚀 Usage
Full pipeline (train + evaluate + visualise)
```bash
python crop_disease_detector.py
```
This will:
Load and explore the dataset
Preprocess and encode features
Train a Random Forest model with 5-fold cross-validation
Print a detailed classification report
Save 4 plots (confusion matrix, feature importance, distribution, accuracy curve)
Run a demo prediction on a sample input
Custom prediction
Edit the `sample` dictionary at the bottom of `crop_disease_detector.py`:
```python
sample = {
    "crop_type": "rice",
    "leaf_color": "yellow",
    "spot_size": "medium",
    "spot_color": "brown",
    "yellowing": "yes",
    "wilting": "no",
    "lesion_count": 12,
    "humidity": 92,
    "temperature": 31
}
```
Valid values for each field:
Feature	Valid Values
`crop_type`	tomato, corn, rice, wheat, potato
`leaf_color`	green, pale_green, yellow, pale_yellow, brown
`spot_size`	none, small, medium, large
`spot_color`	none, brown, black, dark_brown, gray, tan, orange, white
`yellowing`	yes, no
`wilting`	yes, no
`lesion_count`	integer (e.g., 0–25)
`humidity`	integer percentage (e.g., 50–97)
`temperature`	integer °C (e.g., 20–40)
---
📊 Dataset
File: `crop_disease_dataset.csv`
Samples: 120
Features: 10 (9 input + 1 target)
Target column: `disease_label`
Classes: 11 (Healthy + 10 disease types across 5 crops)
The dataset was synthetically generated to reflect real-world patterns documented in agricultural disease research.
---
🧠 Model Details
Property	Value
Algorithm	Random Forest Classifier
Trees	100 estimators
Class balancing	`class_weight="balanced"`
Train/Test split	75% / 25%, stratified
Validation	5-fold cross-validation
Accuracy	~95%+ (varies by run)
---
📈 Output Plots
Plot	Filename
Confusion Matrix	`confusion_matrix.png`
Feature Importance	`feature_importance.png`
Class Distribution	`class_distribution.png`
Accuracy vs No. of Trees	`accuracy_vs_estimators.png`
---
🛠 Dependencies
```
pandas
numpy
matplotlib
seaborn
scikit-learn
```
---
📚 Course
Course: Fundamentals of AI and ML
Semester: Winter 2025–26
Submission Type: BYOP Capstone Project
---
📄 License
This project is submitted for academic evaluation. Free to use for educational purposes.
