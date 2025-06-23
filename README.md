# AI Music Recommender 🎵

This project uses a Decision Tree Classifier to predict a music genre based on user preferences such as age, mood, activity, and more.

## 🔧 Requirements

- Python 3.8+
- pandas
- scikit-learn
- joblib

Install dependencies using:

```bash
pip install -r requirements.txt
```

File	Description
AI-music-recommender.joblib	Trained Decision Tree model
label_encoders.joblib	LabelEncoders used for transforming input data
predict.py	Script to predict genre from user input
music.csv (optional)	Dataset used for training (for reference)

🚀 How to Use
Clone/download this repo and navigate to the folder.

Make sure AI-music-recommender.joblib, label_encoders.joblib, and predict.py are in the same directory.

Open and modify predict.py if you want to test with different input values.

Run the script:
python predict.py

You will see the predicted music genre printed in the console.

📥 Example Input
In predict.py, you can modify the following section with your own values:

Edit
new_input = {
    'age': 22,
    'gender': 'Male',
    'mood': 'Calm',
    'time_of_day': 'Afternoon',
    'activity': 'Studying',
    'country': 'India',
    'hobby': 'Photography'
}
Make sure the values match the original dataset’s formatting (e.g., 'Male' not 'male').

📈 Model Info
Algorithm: Decision Tree Classifier

Accuracy: ~X.XX (on test split)

Features used: age, gender, mood, time_of_day, activity, country, hobby

Target: genre

