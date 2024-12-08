# %%
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer  # For handling NaN values
import os
import math
import numpy as np
import pandas as pd
# pip install rfit
import rfit
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC, LinearSVC
# from sklearn.tree import DecisionTreeClassifier
#
import warnings
warnings.filterwarnings("ignore")
print("\nReady to continue.")

# %%[markdown]
# 1. Use this Housing Price dataset.

# %%
df = pd.read_csv(
    "/Users/apple/Desktop/final-project/data/HousePricesAdv/train.csv", header=0)
df_test = pd.read_csv(
    "/Users/apple/Desktop/final-project/data/HousePricesAdv/test.csv", header=0)
df

# %%
# Use SalePrice as target for K-NN regression.
# Drop the target and other non-predictive columns
X = df.drop(columns=['SalePrice', 'Id'])
y = df['SalePrice']
X.shape, y.shape

# %%
X_test = df_test.drop(columns=['Id'])
X_test.shape, df_test.shape

# %%
# Indentify features as diffferent types

# Categorical features
categorical_features = [
    'MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',
    'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',
    'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
    'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating',
    'CentralAir', 'Electrical', 'Functional', 'GarageType', 'GarageFinish',
    'PavedDrive', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition'
]

# Ordinal features
ordinal_features = [
    'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtQual',
    'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual',
    'GarageCond', 'PoolQC', 'Fence'
]

# Numerical features
numerical_features = [
    'LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
    '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
    'FullBath', 'HalfBath', 'Bedroom', 'Kitchen', 'TotRmsAbvGrd',
    'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
    'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
    'MiscVal', 'MoSold', 'YrSold', 'SalePrice'
]

# Temporal features
temporal_features = ['YearBuilt', 'YearRemodAdd',
                     'GarageYrBlt', 'MoSold', 'YrSold']

# Binary features
binary_features = ['Street', 'Alley', 'CentralAir', 'PavedDrive']

# Target feature
target_feature = ['SalePrice']

# %%
# For features that are *ORDINAL*, recode them as 0,1,2,...
# Ordinal mappings
ordinal_mappings = {
    "OverallQual": {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9},
    "OverallCond": {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9},
    "ExterQual": {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
    "ExterCond": {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
    "BsmtQual": {"NA": -1, "Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
    "BsmtCond": {"NA": -1, "Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
    "BsmtExposure": {"NA": -1, "No": 0, "Mn": 1, "Av": 2, "Gd": 3},
    "BsmtFinType1": {"NA": -1, "Unf": 0, "LwQ": 1, "Rec": 2, "BLQ": 3, "ALQ": 4, "GLQ": 5},
    "BsmtFinType2": {"NA": -1, "Unf": 0, "LwQ": 1, "Rec": 2, "BLQ": 3, "ALQ": 4, "GLQ": 5},
    "HeatingQC": {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
    "KitchenQual": {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
    "FireplaceQu": {"NA": -1, "Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
    "GarageQual": {"NA": -1, "Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
    "GarageCond": {"NA": -1, "Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
    "PoolQC": {"NA": -1, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
    "Fence": {"NA": -1, "MnWw": 0, "GdWo": 1, "MnPrv": 2, "GdPrv": 3}
}

# Recode ordinal features 
for feature, mapping in ordinal_mappings.items():
    if feature in X.columns:
        X[feature] = X[feature].replace(mapping)

X[ordinal_features]

# %%
# Recode ordinal features in test set
for feature, mapping in ordinal_mappings.items():
    if feature in X_test.columns:
        X_test[feature] = X_test[feature].replace(mapping)

X_test[ordinal_features]

# %%
# Drop features that are purely categorical
X_original = X.copy()
X = X_original.drop(columns=categorical_features)

# Check the shapes of X_original and X
print(f"Original X shape (X_original): {X_original.shape}")
print(f"Modified X shape (X): {X.shape}")

# %%
# Drop features that are purely categorical in test set
X_test_original = X_test.copy()
X_test = X_test_original.drop(columns=categorical_features)

# Check the shapes of X_original and X
print(f"Original X_test shape (X_test_original): {X_test_original.shape}")
print(f"Modified X shape (X): {X.shape}")
print(f"Modified X_test shape (X_test): {X_test.shape}")

# %%[markdown]
# 2. Modify the sknn class to perform K-NN regression.

# %%


class knn:
    """
    k-NN Regression Model with Feature Scaling
    Supports gradient-based optimization of feature scaling for k-NN regression.
    """

    import os
    import math
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    def __init__(self,
                 data_x,
                 data_y,
                 k=7,
                 kmax=33,
                 zscale=True,
                 ttsplit=0.5,
                 max_iter=100,
                 seed=1,
                 scoredigits=6,
                 learning_rate_init=0.1,
                 atol=1e-8):
        """
        Initialize k-NN regression model with feature scaling.

        Args:
            data_x (numpy ndarray or pandas DataFrame): Features.
            data_y (numpy ndarray or pandas Series): Target variable.
            k (int): Number of neighbors. Defaults to 7.
            kmax (int): Maximum k-value for benchmarking. Defaults to 33.
            zscale (bool): Whether to standardize features using z-scores. Defaults to True.
            ttsplit (float): Train-test split ratio. Defaults to 0.5.
            max_iter (int): Maximum iterations for scaling optimization. Defaults to 100.
            seed (int): Random seed. Defaults to 1.
            scoredigits (int): Precision for benchmark scores. Defaults to 6.
            learning_rate_init (float): Initial learning rate. Defaults to 0.1.
            atol (float): Absolute tolerance for gradient optimization. Defaults to 1e-8.
        """
        self.k = k
        self.__kmax = kmax
        self.max_iter = max_iter
        self.__seed = seed
        self.__scoredigits = scoredigits
        self.learning_rate = abs(learning_rate_init)
        self.__atol = atol

        # Data preparation
        self.data_x = data_x
        self.data_xz = data_x if not zscale else self.zXform(data_x)
        self.data_y = data_y
        self.__ttsplit = ttsplit if 0 <= ttsplit <= 1 else 0.5
        self.X_train, self.X_test, self.y_train, self.y_test = self.traintestsplit()

        # Scaling initialization
        self.__xdim = self.X_train.shape[1]
        self.__scaleExpos = np.zeros(self.__xdim)
        self.__scaleFactors = np.ones(self.__xdim)

        # Initialize k-NN models
        self.__knnmodels = [np.nan, np.nan] + [
            KNeighborsRegressor(n_neighbors=i, weights='uniform').fit(
                self.X_train, self.y_train)
            for i in range(2, kmax + 1)
        ]

        # Benchmark scores
        self.benchmarkScores = [np.nan, np.nan] + [
            round(model.score(self.X_test, self.y_test), scoredigits) for model in self.__knnmodels[2:]
        ]
        print(f"Benchmark scores (R2) for k-values: {self.benchmarkScores}")

    def zXform(self, data_x):
        """Standardize features using z-score."""
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        return scaler.fit_transform(data_x)

    def traintestsplit(self):
        """Split data into train and test sets."""
        from sklearn.model_selection import train_test_split
        dy = self.data_y.values if isinstance(
            self.data_y, (pd.Series, pd.DataFrame)) else self.data_y
        return train_test_split(self.data_xz, dy, test_size=self.__ttsplit, random_state=self.__seed)

    def scorethis(self, scaleExpos=None, use='test', metric='R2'):
        """
        Score the model using the specified metric.

        Args:
            scaleExpos (list or ndarray): Scaling exponents for features.
            use (str): Dataset to evaluate ('train' or 'test'). Defaults to 'test'.
            metric (str): Evaluation metric ('R2', 'MSE', 'MAE'). Defaults to 'R2'.

        Returns:
            float: Score based on the specified metric.
        """
        # Ensure imports
        from sklearn.metrics import mean_squared_error, mean_absolute_error

        if scaleExpos is not None:
            self.__setExpos2Scales(scaleExpos)

        sfactors = self.__scaleFactors
        X = sfactors * (self.X_train if use == 'train' else self.X_test)
        y = self.y_train if use == 'train' else self.y_test
        self.__knnmodels[self.k].fit(sfactors * self.X_train, self.y_train)

        if metric == 'R2':
            return self.__knnmodels[self.k].score(X, y)
        elif metric == 'MSE':
            return -mean_squared_error(y, self.__knnmodels[self.k].predict(X))
        elif metric == 'MAE':
            return -mean_absolute_error(y, self.__knnmodels[self.k].predict(X))
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    def optimize(self, maxiter=None, learning_rate=None):
        """
        Optimize scaling factors using gradient-based approach.

        Args:
            maxiter (int): Maximum iterations. Defaults to self.max_iter.
            learning_rate (float): Learning rate. Defaults to self.learning_rate.
        """
        maxiter = maxiter or self.max_iter
        learning_rate = learning_rate or self.learning_rate

        for i in range(maxiter):
            grad = self.__evalGradients(learning_rate, use='train')
            if not self.__setNewExposFromGrad(grad):
                break
            if i % 10 == 0:
                print(f"Iteration {i}: Scaling Factors: {self.__scaleFactors}")

    def __setExpos2Scales(self, expos):
        """Update scaling factors based on exponents."""
        self.__scaleExpos = expos - np.mean(expos)
        self.__scaleFactors = np.exp(self.__scaleExpos)

    def __evalGradients(self, learning_rate, use='train'):
        """Evaluate gradients for scaling factors."""
        return np.array([self.__eval1Gradient(i, learning_rate, use) for i in range(self.__xdim)])

    def __eval1Gradient(self, i, learning_rate, use):
        """Evaluate gradient for a single feature."""
        thescale = self.__scaleExpos[i]
        step = max(learning_rate, abs(thescale) * learning_rate)
        max_expos = self.__scaleExpos.copy()
        min_expos = self.__scaleExpos.copy()
        max_expos[i] += step / 2
        min_expos[i] -= step / 2
        return (self.scorethis(scaleExpos=max_expos, use=use) - self.scorethis(scaleExpos=min_expos, use=use)) / step

    def __setNewExposFromGrad(self, grad):
        """Update exponents based on gradients."""
        if np.allclose(grad, 0, atol=self.__atol):
            return False
        self.__scaleExpos += grad / np.linalg.norm(grad) * self.learning_rate
        self.__setExpos2Scales(self.__scaleExpos)
        return True

    def get_scale_factors(self):
        """Return the scaling factors."""
        return self.__scaleFactors

    def get_knn_model(self, k=None):
        """Return the k-NN model for the given k."""
        if k is None:
            k = self.k
        return self.__knnmodels[k]


# %%

# Prepare data
# Handle NaN values in X
imputer = SimpleImputer(strategy='mean')  # Replace NaN with column mean
X_clean = imputer.fit_transform(X)  # Impute missing values

# Check if X_clean contains NaN (shouldn't after imputation)
assert not np.isnan(X_clean).any(), "NaN values remain in the dataset!"

# %%
X_clean = pd.DataFrame(X_clean, columns=X.columns)

# Confirm y (target) is ready
print(X_clean.head())  # View first few rows of features
print(y.head())        # View first few rows of the target

# %%
# Initialize the k-NN regression model
knn_model = knn(data_x=X_clean, data_y=y, k=5,
                zscale=True, ttsplit=0.8, max_iter=200)

# Optimize scaling factors
knn_model.optimize(maxiter=200, learning_rate=0.1)

# Evaluate the model
r2_score = knn_model.scorethis(use='test', metric='R2')
mse_score = knn_model.scorethis(use='test', metric='MSE')
mae_score = knn_model.scorethis(use='test', metric='MAE')

print(f"R^2 Score: {r2_score}")
# MSE is returned as negative for optimization purposes
print(f"Mean Squared Error: {-mse_score}")
# MAE is returned as negative for optimization purposes
print(f"Mean Absolute Error: {-mae_score}")

# %%
# Predict on the test data
# Ensure no missing values in test data
X_test_imputed = imputer.transform(X_test)

# Retrieve the scaling factors and apply them
X_test_scaled = knn_model.get_scale_factors() * X_test_imputed

# Retrieve the k-NN model and make predictions
knn_model_instance = knn_model.get_knn_model()
predictions = knn_model_instance.predict(X_test_scaled)

# Output predictions
print(predictions)

# %%[markdown]
# 3. Modify the sknn class as you see fit to improve the algorithm performance, logic, or presentations.


class sknn:
    """
    Scaling k-NN model with feature scaling optimization.
    v3 - Improved performance, logic, and presentation.
    """
    import numpy as np
    import math
    import pandas as pd
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    def __init__(self,
                 data_x,
                 data_y,
                 classifier=True,
                 k=7,
                 kmax=33,
                 zscale=True,
                 ttsplit=0.5,
                 max_iter=100,
                 seed=1,
                 scoredigits=6,
                 learning_rate_init=0.1,
                 atol=1e-8,
                 verbose=True):
        """
        Initialize the sknn model.

        Args:
            data_x (numpy ndarray or pandas DataFrame): Feature data.
            data_y (numpy ndarray or pandas Series): Target data.
            classifier (bool): Use classification if True; regression otherwise.
            k (int): Number of neighbors for k-NN.
            kmax (int): Maximum k-value for benchmarking.
            zscale (bool): Apply z-score normalization.
            ttsplit (float): Train-test split ratio.
            max_iter (int): Maximum iterations for optimization.
            seed (int): Random seed for reproducibility.
            scoredigits (int): Number of digits to round scores.
            learning_rate_init (float): Initial learning rate for optimization.
            atol (float): Absolute tolerance for gradient convergence.
            verbose (bool): Enable or disable debug output.
        """
        # Initialize properties
        self._classifier = classifier
        self.k = k
        self._kmax = kmax
        self.max_iter = max_iter
        self._seed = seed
        self._scoredigits = scoredigits
        self.learning_rate = abs(learning_rate_init)
        self._atol = atol
        self.verbose = verbose

        # Data preparation
        self.data_x = data_x
        self.data_y = data_y
        self.zscale = zscale
        self._validate_data()
        self.data_xz = self._z_transform(
            self.data_x) if self.zscale else self.data_x

        # Train-test split
        self.ttsplit = ttsplit
        self.X_train, self.X_test, self.y_train, self.y_test = self._train_test_split()

        # Scaling factors and exponents
        self._xdim = self.X_train.shape[1]
        self._scaleExpos = np.zeros(self._xdim)
        self._scaleFactors = np.ones(self._xdim)

        # Initialize k-NN model
        self._knnmodels = {}
        self._init_knn_models()

    def _validate_data(self):
        """Validate input data for compatibility."""
        if self.data_x.shape[0] != self.data_y.shape[0]:
            raise ValueError("Mismatched rows between X and y.")
        if self.data_x.isnull().any().any():
            raise ValueError(
                "X contains NaN values. Handle them before initialization.")
        if pd.isnull(self.data_y).any():
            raise ValueError(
                "y contains NaN values. Handle them before initialization.")

    def _z_transform(self, data_x):
        """Apply z-score normalization."""
        scaler = self.StandardScaler()
        return scaler.fit_transform(data_x)

    def _train_test_split(self):
        """Split data into train and test sets."""
        from sklearn.model_selection import train_test_split
        return train_test_split(self.data_xz, self.data_y, test_size=self.ttsplit, random_state=self._seed)

    def _init_knn_models(self):
        """Initialize k-NN models for multiple k-values."""
        for i in range(2, self._kmax + 1):
            if self._classifier:
                model = self.KNeighborsClassifier(
                    n_neighbors=i, weights='uniform')
            else:
                model = self.KNeighborsRegressor(
                    n_neighbors=i, weights='uniform')
            self._knnmodels[i] = model.fit(self.X_train, self.y_train)

    def optimize(self, maxiter=None, learning_rate=None):
        """
        Optimize scaling factors using gradient-based approach.

        Args:
            maxiter (int): Maximum number of iterations.
            learning_rate (float): Learning rate for optimization.
        """
        maxiter = maxiter or self.max_iter
        learning_rate = learning_rate or self.learning_rate

        for i in range(maxiter):
            grad = self._eval_gradients(learning_rate)
            grad_norm = np.linalg.norm(grad)

            if self.verbose:
                print(
                    f"Iteration {i}: Gradient Norm = {grad_norm}, Scaling Factors = {self._scaleFactors}")

            if grad_norm < self._atol:
                if self.verbose:
                    print(f"Optimization converged at iteration {i}.")
                break

            self._update_exponents_from_grad(grad)

    def _eval_gradients(self, learning_rate):
        """
        Evaluate gradients for scaling factors.

        Args:
            learning_rate (float): Learning rate for gradient computation.

        Returns:
            numpy.ndarray: Array of gradients for each feature.
        """
        grad = np.array([self._eval_single_gradient(i, learning_rate)
                        for i in range(self._xdim)])
        return grad

    def _eval_single_gradient(self, i, learning_rate):
        """Evaluate gradient for a single feature."""
        current_exponent = self._scaleExpos[i]
        step = max(learning_rate, abs(current_exponent) * learning_rate)

        # Perturb the scaling exponent
        self._scaleExpos[i] += step / 2
        self._scaleFactors = np.exp(self._scaleExpos)
        score_high = self.scorethis()

        self._scaleExpos[i] -= step
        self._scaleFactors = np.exp(self._scaleExpos)
        score_low = self.scorethis()

        # Restore original exponent
        self._scaleExpos[i] += step / 2

        gradient = (score_high - score_low) / step
        if self.verbose:
            print(
                f"Feature {i}: score_high={score_high}, score_low={score_low}, gradient={gradient}")
        return gradient

    def _update_exponents_from_grad(self, grad):
        """
        Update scaling exponents based on the computed gradient.

        Args:
            grad (numpy.ndarray): Array of gradients for each feature.

        Returns:
            bool: True if update occurred, False if gradient norm is below tolerance.
        """
        grad_norm = np.linalg.norm(grad)
        if grad_norm < self._atol:
            # Stop optimization if gradient norm is below tolerance
            return False

        # Update scaling exponents
        self._scaleExpos -= grad * self.learning_rate
        # Update scaling factors based on new exponents
        self._scaleFactors = np.exp(self._scaleExpos)

        if self.verbose:
            print(f"Updated scaling factors: {self._scaleFactors}")
        return True

    def scorethis(self, use='test', metric='R2'):
        """
        Score the model using the specified metric.

        Args:
            use (str): 'train' or 'test' dataset.
            metric (str): Metric for evaluation ('R2', 'MSE', 'MAE').

        Returns:
            float: Score for the chosen metric.
        """
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

        # Select dataset
        data = (self.X_train * self._scaleFactors, self.y_train) if use == 'train' else (
            self.X_test * self._scaleFactors, self.y_test)
        X, y = data

        # Retrieve model and make predictions
        model = self._knnmodels[self.k]
        preds = model.predict(X)

        # Evaluate with the selected metric
        if metric == 'R2':
            return r2_score(y, preds)
        elif metric == 'MSE':
            return -mean_squared_error(y, preds)  # Negative for optimization
        elif metric == 'MAE':
            return -mean_absolute_error(y, preds)  # Negative for optimization
        else:
            raise ValueError(f"Unsupported metric: {metric}")


# %%
# from sknn import sknn  # Import the updated sknn class

# Prepare Data
# Handle NaN values in X
imputer = SimpleImputer(strategy='mean')  # Replace NaN with column mean
X_clean = imputer.fit_transform(X)  # Impute missing values

# Check if X_clean contains NaN (shouldn't after imputation)
assert not np.isnan(X_clean).any(), "NaN values remain in the dataset!"

# Convert back to DataFrame for interpretability
X_clean = pd.DataFrame(X_clean, columns=X.columns)

# Confirm y (target) is ready
print(X_clean.head())  # View first few rows of features
print(y.head())        # View first few rows of the target

# %%
# Initialize the sknn regression model
sknn_model = sknn(data_x=X_clean, data_y=y, k=5,
                  zscale=True, ttsplit=0.8, max_iter=200, verbose=True)

# Optimize scaling factors
sknn_model.optimize(maxiter=200, learning_rate=0.5)

# Evaluate the model
r2 = sknn_model.scorethis(use='test', metric='R2')
mse = sknn_model.scorethis(use='test', metric='MSE')
mae = sknn_model.scorethis(use='test', metric='MAE')

print(f"R^2 Score: {r2}")
print(f"Mean Squared Error: {-mse}")  # Negative MSE for optimization purposes
print(f"Mean Absolute Error: {-mae}")  # Negative MAE for optimization purposes

# %%
# Predict on the test data
# Ensure no missing values in test data
X_test_imputed = imputer.transform(X_test)

# Retrieve the scaling factors and apply them
X_test_scaled = sknn_model._scaleFactors * X_test_imputed

# Make predictions using the k-NN model
predictions = sknn_model._knnmodels[sknn_model.k].predict(X_test_scaled)

# Output predictions
print(predictions)


# %%[markdown]
# 4. Find optimized scaling factors for the features for the best model score.

# %%
# Initialize the sknn regression model
sknn_model = sknn(
    data_x=X_clean,  # Feature matrix
    data_y=y,        # Target variable
    k=5,             # Number of neighbors
    zscale=True,     # Apply z-score normalization
    ttsplit=0.8,     # Train-test split ratio
    max_iter=500,    # Maximum iterations for optimization
    verbose=True     # Enable verbose output for optimization progress
)

# Run optimization to adjust scaling factors
sknn_model.optimize(maxiter=500, learning_rate=0.1)

# Retrieve optimized scaling factors
optimized_scaling_factors = sknn_model._scaleFactors

# Evaluate the optimized model
r2 = sknn_model.scorethis(use='test', metric='R2')
mse = sknn_model.scorethis(use='test', metric='MSE')
mae = sknn_model.scorethis(use='test', metric='MAE')

# Print the results
print("Optimized Scaling Factors:")
for feature, scale in zip(X_clean.columns, optimized_scaling_factors):
    print(f"{feature}: {scale:.4f}")

print(f"\nModel Performance after Optimization:")
print(f"R^2 Score: {r2:.4f}")
print(f"Mean Squared Error: {-mse:.4f}")  # Negative for optimization purposes
print(f"Mean Absolute Error: {-mae:.4f}")

# Model Performance after Optimization:
# R ^ 2 Score: 0.2204
# Mean Squared Error: 5050439663.7175
# Mean Absolute Error: 50106.9144

# %%
# iterate on different k values

# Range of k values to test
k_values = [3, 5, 7, 10, 15]

# Placeholder for results
best_k = None
best_score = -np.inf
results = []

for k in k_values:
    print(f"\nTesting k={k}...")
    sknn_model = sknn(data_x=X_clean, data_y=y, k=k, zscale=True,
                      ttsplit=0.8, max_iter=500, verbose=True)

    # Optimize scaling factors for this k
    sknn_model.optimize(maxiter=500, learning_rate=0.1)

    # Evaluate performance
    r2 = sknn_model.scorethis(use='test', metric='R2')
    # Convert negative MSE
    mse = -sknn_model.scorethis(use='test', metric='MSE')
    # Convert negative MAE
    mae = -sknn_model.scorethis(use='test', metric='MAE')

    print(f"Performance for k={k}:")
    print(f"R^2 Score: {r2}")
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")

    # Track the best k
    if r2 > best_score:
        best_score = r2
        best_k = k

    # Save results
    results.append({
        'k': k,
        'R2': r2,
        'MSE': mse,
        'MAE': mae
    })

print(f"\nBest k found: {best_k} with R^2 Score: {best_score}")

# Best k found: 3 with R ^ 2 Score: 0.39199295330427075

# %%[markdown]
# 5. Modify the sknn class to save some results (such as scores, scaling factors, gradients, etc, at various points, like every 100 epoch).


# %%
class sknn_save_more_results:
    """
    Scaling k-NN model with feature scaling optimization.
    v4 - Improved performance, logic, presentation, and result logging.
    """
    import numpy as np
    import math
    import pandas as pd
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    def __init__(self,
                 data_x,
                 data_y,
                 classifier=True,
                 k=7,
                 kmax=33,
                 zscale=True,
                 ttsplit=0.5,
                 max_iter=100,
                 seed=1,
                 scoredigits=6,
                 learning_rate_init=0.1,
                 atol=1e-8,
                 verbose=True):
        """
        Initialize the sknn model.
        """
        # Initialize properties
        self._classifier = classifier
        self.k = k
        self._kmax = kmax
        self.max_iter = max_iter
        self._seed = seed
        self._scoredigits = scoredigits
        self.learning_rate = abs(learning_rate_init)
        self._atol = atol
        self.verbose = verbose

        # Initialize logs
        self.logs = []  # Store iteration logs for debugging and visualization

        # Data preparation
        self.data_x = data_x
        self.data_y = data_y
        self.zscale = zscale
        self._validate_data()
        self.data_xz = self._z_transform(
            self.data_x) if self.zscale else self.data_x

        # Train-test split
        self.ttsplit = ttsplit
        self.X_train, self.X_test, self.y_train, self.y_test = self._train_test_split()

        # Scaling factors and exponents
        self._xdim = self.X_train.shape[1]
        self._scaleExpos = np.zeros(self._xdim)
        self._scaleFactors = np.ones(self._xdim)

        # Initialize k-NN model
        self._knnmodels = {}
        self._init_knn_models()

    def _validate_data(self):
        """Validate input data for compatibility."""
        if self.data_x.shape[0] != self.data_y.shape[0]:
            raise ValueError("Mismatched rows between X and y.")
        if self.data_x.isnull().any().any():
            raise ValueError(
                "X contains NaN values. Handle them before initialization.")
        if pd.isnull(self.data_y).any():
            raise ValueError(
                "y contains NaN values. Handle them before initialization.")

    def _z_transform(self, data_x):
        """Apply z-score normalization."""
        scaler = self.StandardScaler()
        return scaler.fit_transform(data_x)

    def _train_test_split(self):
        """Split data into train and test sets."""
        return train_test_split(self.data_xz, self.data_y, test_size=self.ttsplit, random_state=self._seed)

    def _init_knn_models(self):
        """Initialize k-NN models for multiple k-values."""
        for i in range(2, self._kmax + 1):
            if self._classifier:
                model = self.KNeighborsClassifier(
                    n_neighbors=i, weights='uniform')
            else:
                model = self.KNeighborsRegressor(
                    n_neighbors=i, weights='uniform')
            self._knnmodels[i] = model.fit(self.X_train, self.y_train)

    def optimize(self, maxiter=None, learning_rate=None):
        """
        Optimize scaling factors using gradient-based approach.

        Args:
            maxiter (int): Maximum number of iterations.
            learning_rate (float): Learning rate for optimization.
        """
        maxiter = maxiter or self.max_iter
        learning_rate = learning_rate or self.learning_rate

        for i in range(maxiter):
            grad = self._eval_gradients(learning_rate)
            grad_norm = np.linalg.norm(grad)

            # Record logs
            r2_train_score = self.scorethis(use='train', metric='R2')
            r2_test_score = self.scorethis(use='test', metric='R2')
            self.logs.append({
                'iteration': i,
                'R2_train': r2_train_score,
                'R2_test': r2_test_score,
                'scaling_factors': self._scaleFactors.tolist(),
                'gradient_norm': grad_norm
            })

            # Check for convergence
            if grad_norm < self._atol:
                if self.verbose:
                    print(f"Optimization converged at iteration {i}.")
                break

            # Update scaling factors
            self._update_exponents_from_grad(grad)

            if self.verbose and i % 10 == 0:
                print(
                    f"Iteration {i}: R2 Train={r2_train_score:.4f}, R2 Test={r2_test_score:.4f}, Gradient Norm={grad_norm:.4f}")

    def _eval_gradients(self, learning_rate):
        """Evaluate gradients for scaling factors."""
        grad = np.array([self._eval_single_gradient(i, learning_rate)
                        for i in range(self._xdim)])
        return grad

    def _eval_single_gradient(self, i, learning_rate):
        """Evaluate gradient for a single feature."""
        current_exponent = self._scaleExpos[i]
        step = max(learning_rate, abs(current_exponent) * learning_rate)
        self._scaleExpos[i] += step / 2
        score_high = self.scorethis()
        self._scaleExpos[i] -= step
        score_low = self.scorethis()
        self._scaleExpos[i] += step / 2
        return (score_high - score_low) / step

    def _update_exponents_from_grad(self, grad):
        """Update scaling exponents based on gradient."""
        if np.linalg.norm(grad) < self._atol:
            return False
        self._scaleExpos -= grad * self.learning_rate
        self._scaleFactors = np.exp(self._scaleExpos)
        return True

    def scorethis(self, use='test', metric='R2'):
        """
        Score the model using the specified metric.

        Args:
            use (str): 'train' or 'test' dataset.
            metric (str): Metric for evaluation ('R2', 'MSE', 'MAE').

        Returns:
            float: Score for the chosen metric.
        """
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

        # Select dataset
        data = (self.X_train, self.y_train) if use == 'train' else (
            self.X_test, self.y_test)
        X, y = data

        # Retrieve model and make predictions
        model = self._knnmodels[self.k]
        preds = model.predict(X)

        # Evaluate with the selected metric
        if metric == 'R2':
            return r2_score(y, preds)
        elif metric == 'MSE':
            return -mean_squared_error(y, preds)  # Negative for optimization
        elif metric == 'MAE':
            return -mean_absolute_error(y, preds)  # Negative for optimization
        else:
            raise ValueError(f"Unsupported metric: {metric}")


# %%
# Initialize the sknn regression model (k=3)
sknn_save_more_results_model = sknn_save_more_results(
    data_x=X_clean,  # Feature matrix
    data_y=y,        # Target variable
    k=3,             # k=3 is the best based on previous optimization
    zscale=True,     # Apply z-score normalization
    ttsplit=0.8,     # Train-test split ratio
    max_iter=500,    # Maximum iterations for optimization
    verbose=True     # Enable verbose output for optimization progress
)

# Run optimization to adjust scaling factors
sknn_save_more_results_model.optimize(maxiter=500, learning_rate=0.1)

# Access logs
logs_df = pd.DataFrame(sknn_save_more_results_model.logs)
print(logs_df.head())

# %%[markdown]
# 6. Compare the results of the optimized scaling factors to Feature Importance from other models, such as Tree regressor for example.

# %%

# Train the sknn model
sknn_model = sknn(
    data_x=X_clean,
    data_y=y,
    k=3,
    zscale=True,
    ttsplit=0.8,
    max_iter=500,
    verbose=True
)
sknn_model.optimize(maxiter=500, learning_rate=0.1)

# Retrieve optimized scaling factors
optimized_scaling_factors = sknn_model._scaleFactors
scaling_factors_df = pd.DataFrame({
    'Feature': X_clean.columns,
    'Scaling_Factor': optimized_scaling_factors
})

# Train a Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(sknn_model.X_train, sknn_model.y_train)

# Extract feature importances
feature_importances = rf_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X_clean.columns,
    'Feature_Importance': feature_importances
})

# Merge scaling factors and feature importances
comparison_df = scaling_factors_df.merge(importance_df, on='Feature')

# Print the comparison
print("\nComparison of Scaling Factors and Feature Importances:")
print(comparison_df)

# Visualize the comparison
comparison_df.set_index('Feature').plot(kind='bar', figsize=(12, 6))
plt.title(
    'Comparison of Scaling Factors (sknn) vs. Feature Importance (Random Forest)')
plt.ylabel('Value')
plt.xlabel('Feature')
plt.legend(['Scaling Factor', 'Feature Importance'])
plt.tight_layout()
plt.show()

# Evaluate Random Forest Performance
rf_preds = rf_model.predict(sknn_model.X_test)
rf_r2 = r2_score(sknn_model.y_test, rf_preds)

print(f"\nRandom Forest Performance:")
print(f"R^2 Score: {rf_r2:.4f}")

# image could be seen here: ~/final-project/src/image/scaling_factors_vs_feature_importance.png
