import pandas as pd
import joblib

# Load your saved model and label encoders
model = joblib.load('AI-music-recommender.joblib')
label_encoders = joblib.load('label_encoders.joblib')

# Features expected by the model (exclude target 'genre')
features = ['age', 'gender', 'mood', 'time_of_day', 'activity', 'country', 'hobby']

def predict_genre(new_input_dict):
    # Create DataFrame for the input
    input_df = pd.DataFrame([new_input_dict])
    
    # Encode categorical columns using saved label encoders
    for col, le in label_encoders.items():
        if col in input_df.columns:
            # For numeric columns like 'age', no encoding needed
            # But if it is categorical, encode it
            input_df[col] = le.transform(input_df[col])
    
    # Predict encoded genre
    pred_encoded = model.predict(input_df)[0]
    
    # Decode genre back to original label
    genre_label = label_encoders['genre'].inverse_transform([pred_encoded])[0]
    return genre_label

if __name__ == '__main__':
    # Your input values here (all except genre)
    new_input = {
        'age': 22,                     
        'gender': 'Female',            
        'mood': 'Sad',              
        'time_of_day': 'Night',   
        'activity': 'Relaxing',       
        'country': 'UK',            
        'hobby': 'Photography'         
    }
    
    prediction = predict_genre(new_input)
    print("Predicted genre:", prediction)
