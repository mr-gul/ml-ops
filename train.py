from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import joblib
import os
import json


def main():
    iris=load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    #save the model
    os.makedirs('artifacts', exist_ok=True)
    model_path = os.path.join('artifacts', 'model.pkl')
    joblib.dump(model, model_path)
    #evaluate the model
    accuracy= model.score(X_test, y_test)
    metrics = {'accuracy': float(accuracy)}
    with open(os.path.join('artifacts', 'metrics.json'), 'w') as f:
        json.dump(metrics, f)
    print(f'Model saved to {model_path}')
    print(f'Model accuracy: {accuracy:}')
if __name__ == '__main__':
    main()


