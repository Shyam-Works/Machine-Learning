
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import joblib

music = pd.read_csv('music.csv')

# Encode categorical columns
label_encoders = {}
for column in music.columns:
    if music[column].dtype == object:
        le = LabelEncoder()
        music[column] = le.fit_transform(music[column])
        label_encoders[column] = le
joblib.dump(label_encoders, 'label_encoders.joblib')
x = music.drop(columns=['genre'])
y = music['genre']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# train
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy_score = accuracy_score(y_test, y_pred)
print("Model Accuracy on Test Set:", accuracy_score)

joblib.dump(model, 'AI-music-recommender.joblib')

tree.export_graphviz(
    model,
    out_file='AI-music-recommender.dot',
    feature_names=x.columns,
    class_names=label_encoders['genre'].classes_,
    label='all',
    rounded=True,
    filled=True
)