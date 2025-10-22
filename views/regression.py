#!./.venv/bin/python
# coding: utf-8

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn import svm
from sklearn.ensemble import (
    GradientBoostingRegressor,
    AdaBoostRegressor,
    RandomForestRegressor,
)
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split #cross_val_score, GridSearchCV, KFold, RandomizedSearchCV
from sklearn.linear_model import LassoCV, RidgeCV, HuberRegressor, LinearRegression
from functools import reduce
# Function that assigns the regression model to be used and its parameters
# import _pickle as cPickle
from _pickle import dumps
from utils import get_multiplestation_data


class ModelFactory:
    """Factory class for creating regression models.
    
    Follows the Factory Pattern and Open/Closed Principle - easy to extend
    with new models without modifying existing code.
    """
    
    @staticmethod
    def create_model(regression_model):
        """Create and return a regression model based on the specified type.
        
        Args:
            regression_model: String name of the regression model to create
            
        Returns:
            Configured sklearn/xgboost model instance
            
        Raises:
            ValueError: If an unsupported model type is specified
        """
        models = {
            "Linear": LinearRegression(),
            "Lasso": LassoCV(alphas=(0.001, 0.01, 0.1, 1, 10, 100, 1000)),
            "Huber": HuberRegressor(),
            "SVM": svm.SVR(
                kernel="rbf",
                degree=3,
                gamma="scale",
                coef0=0.0,
                tol=0.001,
                C=1.0,
                epsilon=0.1,
                shrinking=True,
                cache_size=200,
                verbose=False,
                max_iter=-1,
            ),
            "Random Forest": RandomForestRegressor(n_estimators=100),
            "AdaBoost": AdaBoostRegressor(random_state=0, n_estimators=100),
            "XGBoost": XGBRegressor(objective='reg:squarederror')
        }
        
        model = models.get(regression_model)
        if model is None:
            raise ValueError(
                f"Unsupported model: {regression_model}. "
                f"Choose from: {', '.join(models.keys())}"
            )
        return model


def regressionModel(regression_model):
    """Legacy function for backward compatibility. Use ModelFactory.create_model instead."""
    try:
        return ModelFactory.create_model(regression_model)
    except ValueError as e:
        print(str(e))
        return None


class RegressionVisualizer:
    """Handles all visualization for regression models.
    
    Follows Single Responsibility Principle - only responsible for creating plots.
    """
    
    def __init__(self, stationparameterpairs):
        """Initialize visualizer with station parameter pairs.
        
        Args:
            stationparameterpairs: List of (station, parameter) tuples
        """
        self.stationparameterpairs = stationparameterpairs
        self.stations = [i[0] for i in stationparameterpairs]
        self.parameters = [i[1] for i in stationparameterpairs]
    
    def create_training_plot(self, target_train, target_train_pred, target_test, 
                            target_test_pred, RMSE_train, RMSE_test):
        """Create plot showing training and test predictions.
        
        Args:
            target_train: Training target values
            target_train_pred: Predicted training values
            target_test: Test target values
            target_test_pred: Predicted test values
            RMSE_train: RMSE for training set
            RMSE_test: RMSE for test set
            
        Returns:
            Plotly Figure object
        """
        RMSE_train_text = f"RMSE for training set: {RMSE_train: .5}"
        RMSE_test_text = f"RMSE for test set: {RMSE_test: .5}"

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                y=target_train, x=target_train.index, mode="lines", name="Training Data"
            )
        )

        fig.add_trace(
            go.Scatter(
                y=target_train_pred,
                x=target_train.index,
                mode="lines",
                line_dash="dash",
                name="Model Predictions on Training Data",
            )
        )

        fig.add_trace(
            go.Scatter(
                y=target_test, x=target_test.index, mode="lines", name="Test Data"
            )
        )

        fig.add_trace(
            go.Scatter(
                y=target_test_pred,
                x=target_test.index,
                mode="lines",
                line_dash="dash",
                name="Model Predictions on Test Data",
            )
        )

        fig.add_annotation(
            xref="paper",
            x=0.5,
            yref="paper",
            y=1.12,
            text=f"{RMSE_train_text} - {RMSE_test_text}",
            font=dict(family="Courier New, monospace", size=14, color="black"),
            align="center",
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            bgcolor="lightgrey",
            opacity=0.7,
            showarrow=False,
        )

        fig.update_yaxes(title_text=f"{self.parameters[0]}")
        fig.update_xaxes(title_text="Date")

        fig.update_layout(
            showlegend=True,
            height=500,
            width=950,
            title={
                "text": "Model Predictions on Training and Test Data",
                "xanchor": "center",
                "yanchor": "top",
                "y": 0.9,
                "x": 0.4,
            },
        )
        return fig
    
    def create_model_fit_plot(self, features_train, target_train, target_train_pred, data):
        """Create plot showing model fit to data.
        
        Args:
            features_train: Training features
            target_train: Training target values
            target_train_pred: Predicted training values
            data: Original data with index
            
        Returns:
            Plotly Figure object
        """
        if len(self.stations) == 3:
            return self._create_3d_plot(features_train, target_train, target_train_pred, data)
        else:
            return self._create_2d_subplots(features_train, target_train, target_train_pred)
    
    def _create_3d_plot(self, features_train, target_train, target_train_pred, data):
        """Create 3D scatter plot for 3-station case."""
        customdata = data.reset_index()
        trace1 = go.Scatter3d(
            x=features_train.iloc[:, 0],
            y=features_train.iloc[:, 1],
            z=target_train,
            mode="markers",
            customdata=customdata,
            hovertemplate="Date: %{customdata[0]: .2f}"
            + "<br> x: %{customdata[1]: .2f}</br>"
            + "y: %{customdata[2]: .2f}"
            + "<br>z: %{customdata[3]: .2f}</br>",
            name="Response vs. Predictors",
        )

        trace2 = go.Scatter3d(
            x=features_train.iloc[:, 0],
            y=features_train.iloc[:, 1],
            z=pd.DataFrame(target_train_pred.tolist()).iloc[:, 0],
            mode="lines",
            name="Model Fit",
        )

        data = [trace1, trace2]
        layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))

        fig = go.Figure(data=data, layout=layout)

        fig.update_layout(
            showlegend=True,
            legend={"orientation": "h"},
            height=500,
            width=950,
            title={
                "text": "Regression Model",
                "xanchor": "center",
                "yanchor": "top",
                "y": 0.9,
                "x": 0.4,
            },
            scene=dict(
                xaxis_title=f"{self.stationparameterpairs[0]} x",
                yaxis_title=f"{self.stationparameterpairs[1]} y",
                zaxis_title=f"{self.stationparameterpairs[2]} z",
            ),
        )
        return fig
    
    def _create_2d_subplots(self, features_train, target_train, target_train_pred):
        """Create 2D subplot for multiple predictors."""
        fig = make_subplots(
            rows=len(self.stations),
            cols=1,
            vertical_spacing=0.1,
        )
        for i in range(0, len(self.stations) - 1):
            fig.append_trace(
                go.Scatter(
                    x=features_train.iloc[:, i],
                    y=target_train,
                    mode="markers",
                    marker_color="#1f77b4",
                    hovertext=features_train.index,
                    name="Response vs. Predictors",
                ),
                row=i + 1,
                col=1,
            )

            fig.append_trace(
                go.Scatter(
                    x=features_train.iloc[:, i],
                    y=pd.DataFrame(target_train_pred.tolist()).iloc[:, 0],
                    mode="lines",
                    marker_color="#ff7f0e",
                    hovertext=features_train.index,
                    name="Model Fit",
                ),
                row=i + 1,
                col=1,
            )

            fig.update_xaxes(
                title_text=f"{self.stationparameterpairs[i]}", row=i + 1, col=1
            )
            fig.update_yaxes(
                title_text=f"{self.stationparameterpairs[0]}", row=i + 1, col=1
            )
        fig.update_layout(
            showlegend=True,
            height=950,
            width=950,
            title={
                "text": "Regression Model Slice(s)",
                "xanchor": "center",
                "yanchor": "top",
                "x": 0.4,
            },
            font=dict(
                family="Courier New, monospace",
                size=12,
            ),
        )
        return fig
    
    def create_predictions_plot(self, predict_target, predictions, predict_features):
        """Create plot showing model predictions.
        
        Args:
            predict_target: Actual target values for prediction period
            predictions: Model predictions
            predict_features: Features for prediction period
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                y=predict_target,
                x=predict_target.index,
                mode="lines",
                name=f"{self.stations[0]} {self.parameters[0]}",
            )
        )

        fig.add_trace(
            go.Scatter(
                y=predictions,
                x=predict_features.index,
                mode="lines",
                name="Model Predictions",
            )
        )

        fig.update_layout(
            showlegend=True,
            height=500,
            width=950,
            title={
                "text": "Model Predictions",
                "xanchor": "center",
                "yanchor": "top",
                "y": 0.9,
                "x": 0.4,
            },
            xaxis_title="Date",
            yaxis_title=f"{self.stationparameterpairs[0]}",
        )
        return fig


class DataHandler:
    """Handles data loading and preparation.
    
    Follows Single Responsibility Principle - only responsible for data operations.
    """
    
    def __init__(self, stationparameterpairs, begindate, enddate):
        """Initialize data handler.
        
        Args:
            stationparameterpairs: List of (station, parameter) tuples
            begindate: Start date for data retrieval
            enddate: End date for data retrieval
        """
        self.stationparameterpairs = stationparameterpairs
        self.stations = [i[0] for i in stationparameterpairs]
        self.parameters = [i[1] for i in stationparameterpairs]
        self.begindate = begindate
        self.enddate = enddate
        self.data = None
    
    def load_data(self, orient="records"):
        """Load data from AWDB Web Service.
        
        Args:
            orient: Data orientation format
            
        Returns:
            DataFrame with loaded data
        """
        self.data = get_multiplestation_data(
            self.stationparameterpairs, self.begindate, self.enddate, orient
        ).dropna()
        return self.data
    
    def prepare_train_test_split(self, data, test_size):
        """Split data into training and test sets.
        
        Args:
            data: Input DataFrame
            test_size: Proportion of data to use for testing
            
        Returns:
            Tuple of (features_train, features_test, target_train, target_test)
        """
        target = data.iloc[:, 0]
        features = data.iloc[:, 1:]
        
        return train_test_split(
            features, target, test_size=test_size, shuffle=False
        )


class ModelTrainer:
    """Trains and evaluates regression models.
    
    Follows Single Responsibility Principle - focused on model training and evaluation.
    """
    
    def __init__(self, model_factory=None):
        """Initialize model trainer.
        
        Args:
            model_factory: Factory instance for creating models (defaults to ModelFactory)
        """
        self.model_factory = model_factory or ModelFactory()
        self.regr = None
        self.regressor_type = None
        self.RMSE_train = None
        self.RMSE_test = None
        self.regr_data_string = None
    
    def train(self, features_train, target_train, regression_model):
        """Train a regression model.
        
        Args:
            features_train: Training features
            target_train: Training target values
            regression_model: String name of model type to train
            
        Returns:
            Trained model
        """
        self.regr = self.model_factory.create_model(regression_model)
        self.regressor_type = regression_model
        self.regr.fit(features_train, target_train)
        self.regr_data_string = dumps(self.regr)
        return self.regr
    
    def evaluate(self, features_train, target_train, features_test, target_test):
        """Evaluate trained model on train and test sets.
        
        Args:
            features_train: Training features
            target_train: Training target values
            features_test: Test features
            target_test: Test target values
            
        Returns:
            Tuple of (target_train_pred, target_test_pred, RMSE_train, RMSE_test)
        """
        if self.regr is None:
            raise ValueError("Model must be trained before evaluation")
        
        target_train_pred = self.regr.predict(features_train)
        target_test_pred = self.regr.predict(features_test)
        
        self.RMSE_train = mean_squared_error(target_train, target_train_pred)
        self.RMSE_test = mean_squared_error(target_test, target_test_pred)
        
        return target_train_pred, target_test_pred, self.RMSE_train, self.RMSE_test


class ModelPredictor:
    """Makes predictions using trained models.
    
    Follows Single Responsibility Principle - focused on making predictions.
    """
    
    def __init__(self, model, stationparameterpairs):
        """Initialize predictor.
        
        Args:
            model: Trained regression model
            stationparameterpairs: List of (station, parameter) tuples
        """
        self.model = model
        self.stationparameterpairs = stationparameterpairs
        self.stations = [i[0] for i in stationparameterpairs]
        self.parameters = [i[1] for i in stationparameterpairs]
    
    def predict(self, predict_begindate, predict_enddate):
        """Make predictions for a specified date range.
        
        Args:
            predict_begindate: Start date for predictions
            predict_enddate: End date for predictions
            
        Returns:
            Tuple of (predictions, predict_target, predict_features, predict_data)
        """
        predict_data = get_multiplestation_data(
            self.stationparameterpairs, predict_begindate, predict_enddate
        )
        
        # Convert null values to 0, so they are obvious in the plot
        predict_target = predict_data.iloc[:, 0].fillna(0)
        predict_features = predict_data.iloc[:, 1:].fillna(0)
        
        # Run predictions
        predictions = self.model.predict(predict_features)
        
        # Combine predictions into a df with rest of data
        column_name = 'Predictions - ' + f'{self.stations[0]}' + '(' + f'{self.parameters[0]}' + ')'
        predict_data.insert(0, column_name, list(predictions))
        predict_data[column_name] = predict_data[column_name].astype(float).round(2)
        predict_data.reset_index(inplace=True)
        
        return predictions, predict_target, predict_features, predict_data



class RegressionFun:
    """Facade class that coordinates regression workflow.
    
    Refactored to follow SOLID principles by delegating responsibilities to specialized classes.
    Maintains backward compatibility with existing code while using composition over inheritance.
    """
    
    def __init__(
        self, stationparameterpairs, begindate, enddate, orient="records"
    ):
        """Initialize regression workflow coordinator.
        
        Args:
            stationparameterpairs: List of (station, parameter) tuples
            begindate: Start date for data retrieval
            enddate: End date for data retrieval
            orient: Data orientation format
        """
        self.stationparameterpairs = stationparameterpairs
        self.stations = [i[0] for i in stationparameterpairs]
        self.parameters = [i[1] for i in stationparameterpairs]
        self.begindate = begindate
        self.enddate = enddate

        # Initialize specialized components
        self.data_handler = DataHandler(stationparameterpairs, begindate, enddate)
        self.data = self.data_handler.load_data(orient)
        self.visualizer = RegressionVisualizer(stationparameterpairs)
        self.trainer = ModelTrainer()
        
        # Attributes for backward compatibility
        self.target = None
        self.features = None
        self.features_train = None
        self.features_test = None
        self.target_train = None
        self.target_test = None
        self.target_train_pred = None
        self.target_test_pred = None
        self.regr = None
        self.regressor_type = None
        self.regr_data_string = None
        self.RMSE_train = None
        self.RMSE_test = None
        self.traintest_fig = None
        self.modelfit_fig = None
        self.predictions_fig = None
        self.predict_data = None

    def train_model(self, regression_model, test_size):
        """Train and evaluate a regression model.
        
        Function checks model fit on train and test sets. Use to check which
        stations result in the best fitting model. Once the best model is found,
        it can be used in the make_predictions function to predict null values.

        Args:
            regression_model: Model type ('Linear', 'Lasso', 'Huber', 'SVM', 
                            'Random Forest', 'AdaBoost', 'XGBoost')
            test_size: Proportion of data to use for testing (0 to 1)
        """
        # Define Targets and Features (e.g. Response and Predictor Variables)
        self.target = self.data.iloc[:, 0]
        self.features = self.data.iloc[:, 1:]

        # Split into training and test sets
        (self.features_train, self.features_test, 
         self.target_train, self.target_test) = self.data_handler.prepare_train_test_split(
            self.data, test_size
        )

        # Train model
        self.regr = self.trainer.train(
            self.features_train, self.target_train, regression_model
        )
        self.regressor_type = self.trainer.regressor_type
        self.regr_data_string = self.trainer.regr_data_string

        # Evaluate model
        (self.target_train_pred, self.target_test_pred, 
         self.RMSE_train, self.RMSE_test) = self.trainer.evaluate(
            self.features_train, self.target_train,
            self.features_test, self.target_test
        )

        # Create visualizations
        self.traintest_fig = self.visualizer.create_training_plot(
            self.target_train, self.target_train_pred,
            self.target_test, self.target_test_pred,
            self.RMSE_train, self.RMSE_test
        )
        
        self.modelfit_fig = self.visualizer.create_model_fit_plot(
            self.features_train, self.target_train, 
            self.target_train_pred, self.data
        )

    def make_predictions(self, predict_begindate, predict_enddate):
        """Make predictions using the trained model.
        
        Args:
            predict_begindate: Start date for prediction period
            predict_enddate: End date for prediction period
        """
        if self.regr is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Create predictor and make predictions
        predictor = ModelPredictor(self.regr, self.stationparameterpairs)
        (predictions, predict_target, 
         predict_features, self.predict_data) = predictor.predict(
            predict_begindate, predict_enddate
        )
        
        # Create visualization
        self.predictions_fig = self.visualizer.create_predictions_plot(
            predict_target, predictions, predict_features
        )

