from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_recall_curve

digits=load_digits()
x=digits.data
y=digits.target

model=RandomForestClassifier()

model.fit(x,y)

y_pred=model.predict(x)
accuracy=accuracy_score(y,y_pred)

print(f"Accuracy:{accuracy}")


