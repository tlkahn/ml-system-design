from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Example input data
X = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
y = [3, 5, 8]

# Create PolynomialFeatures object with degree 2
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

# Generate cross features
X_cross = poly.fit_transform(X)

# Train a linear regression model
reg = LinearRegression()
reg.fit(X_cross, y)  # y represents the target variable
reg.coef_
