import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from src.preprocess import load_raw, build_pipeline

def train(path):
    df = load_raw(path)
    pl = build_pipeline()
    X = pl.fit_transform(df)
    y = (df['detection_types'] == 'waf_rule').astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    import sklearn.metrics as m
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1]
    acc = m.accuracy_score(y_test, preds)
    auc = m.roc_auc_score(y_test, probs)

    with mlflow.start_run():
        mlflow.log_metric('accuracy', acc)
        mlflow.log_metric('auc', auc)
        mlflow.sklearn.log_model(model, 'model')
        mlflow.sklearn.log_model(pl, 'preprocessor')  # or save separately
    print("done", acc, auc)

if __name__=='__main__':
    train('data/cybersecurity_data.csv')
