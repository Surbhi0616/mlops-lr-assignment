from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def test_data_loading():
    X, y = fetch_california_housing(return_X_y=True)
    assert X.shape[0] == y.shape[0]

def test_model_instance():
    model = LinearRegression()
    assert isinstance(model, LinearRegression)

def test_model_training():
    X, y = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LinearRegression()
    model.fit(X_train, y_train)
    assert hasattr(model, "coef_")

def test_r2_score_threshold():
    X, y = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    assert r2 > 0.5
