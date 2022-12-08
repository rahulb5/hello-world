"""

MAKE SURE TO RUN THIS WITH METAFLOW LOCAL FIRST

"""


from metaflow import FlowSpec, step, Parameter, IncludeFile, current
from datetime import datetime
import os
from dag_utils import utils, models
import comet_ml
from comet_ml import Experiment
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# make sure we are running locally for this
assert os.environ.get('METAFLOW_DEFAULT_DATASTORE', 'local') == 'local'
assert os.environ.get('METAFLOW_DEFAULT_ENVIRONMENT', 'local') == 'local'


class VolatilityAndExcessReturnPredictionFlow(FlowSpec):
    """
    VolatilityPredictionFlow is a DAG reading data from a file 
    and training a Regression model successfully.
    """
    
    # if a static file is part of the flow, 
    # it can be called in any downstream process,
    # gets versioned etc.
    # https://docs.metaflow.org/metaflow/data#data-in-local-files
    DATA_FILE = IncludeFile(
        'dataset',
        help='Text file with the dataset',
        is_text=True,
        default='final.csv')

    @step
    def start(self):
        """
        Start up and print out some info to make sure everything is ok metaflow-side
        """
        print("Starting up at {}".format(datetime.utcnow()))
        # debug printing - this is from https://docs.metaflow.org/metaflow/tagging
        # to show how information about the current run can be accessed programmatically
        print("flow name: %s" % current.flow_name)
        print("run id: %s" % current.run_id)
        print("username: %s" % current.username)
        self.next(self.load_data)

    @step
    def load_data(self): 
        """
        Read the data in from the static file
        """
        import pandas as pd
        from io import StringIO
        
        self.df = pd.read_csv(StringIO(self.DATA_FILE))
        self.df.set_index('Date', inplace = True)
        self.df = self.df.sort_index()
        self.next(self.clean_transform_dataset)
    
    @step
    def clean_transform_dataset(self) -> None:
        """
        Manipulate the dataframe to fix datatypes, fill missing rows and add any derived features
        """
        self.df['Mcap'] = self.df['Mcap'].apply(utils.value_to_float)

        # TODO: Normalize all values

        #creating two different dataframes for volatility and excess returns
        self.vol_df = self.df.copy()
        self.ret_df = self.df.copy()
        
        #Removing all the dates where we don't have volatility value
        self.vol_df.dropna(subset = ['Volatility'], inplace = True)

        #adding lagged variables based on exploratory data analysis
        lagged_data = []
        for i in range(1,10):
            col_label = "Volshift" + str(i)
            self.vol_df[col_label] = self.vol_df['Volatility'].shift(i)
        
        #forward fill all the rows
        self.vol_df.ffill(inplace = True)

        #drop all the rows with NA
        self.vol_df.dropna(inplace = True)

        #Removing all the dates where we don't have excess return value
        self.ret_df.dropna(subset = ['Mkt_rf'], inplace = True)

        #adding lagged variables based on 
        self.ret_df['Mkt_rf_shifted'] = self.ret_df['Mkt_rf'].shift(1)

        #forward fill all the rows
        self.ret_df.ffill(inplace = True)

        #drop all the rows with NA
        self.ret_df.dropna(inplace = True)
        
        
        # Shift both Volatility and Excess Return back by 1 week since our target is the value for next week
        self.vol_df['Volatility'] = self.vol_df.Volatility.shift(-1)
        self.ret_df['Mkt_rf'] = self.vol_df.Mkt_rf.shift(-1)
        self.vol_df.dropna(inplace = True)
        self.ret_df.dropna(inplace = True)

        self.next(self.check_dataset)
        
    @step
    def check_dataset(self):
        """
        Check data is ok before training starts
        """
        assert(self.vol_df.isnull().any().any() == False)
        assert(self.ret_df.isnull().any().any() == False)
        self.next(self.train_test_split)

    @step
    def train_test_split(self):
        import pandas as pd

        # split the dataframes in X and y dataframes
        vol_col = [x for x in self.vol_df.keys() if x != 'Volatility']
        self.vol_df_X = self.vol_df[vol_col]
        self.vol_df_y = self.vol_df['Volatility']

        ret_col = [x for x in self.ret_df.keys() if x != 'Mkt_rf']
        self.ret_df_X = self.ret_df[ret_col]
        self.ret_df_y = self.ret_df['Mkt_rf']
        
        # Train and Test Split lengths
        len_train_vol = round(len(self.vol_df)*0.7)
        len_train_ret = round(len(self.ret_df)*0.7)
        
        # Train Test split for Volatility dataset
        self.vol_train_X, self.vol_test_X = self.vol_df_X[:len_train_vol], self.vol_df_X[len_train_vol:]
        self.vol_train_y, self.vol_test_y = self.vol_df_y[:len_train_vol], self.vol_df_y[len_train_vol:]
        
        # Train Test split for Market Return dataset
        self.ret_train_X, self.ret_test_X = self.ret_df_X[:len_train_ret], self.ret_df_X[len_train_ret:]
        self.ret_train_y, self.ret_test_y = self.ret_df_y[:len_train_ret], self.ret_df_y[len_train_ret:]

        # Now invoke a sub-dag for each type of prediction task
        self.pipeline_types = ['VolatilityPrediction','ExcessReturnPrediction']
        self.next(self.begin_prediction_pipeline, foreach="pipeline_types")

    @step
    def begin_prediction_pipeline(self):
        self.pipeline_type = self.input
        print(f'Beginning {self.pipeline_type} Pipeline')

        # Choose Train and Test dataset to pass on based on choice of Pipeline
        if self.pipeline_type == 'VolatilityPrediction':
            self.X_train = self.vol_train_X
            self.y_train = self.vol_train_y
            self.X_test = self.vol_test_X
            self.y_test = self.vol_test_y
        else:
            self.X_train = self.ret_train_X
            self.y_train = self.ret_train_y
            self.X_test = self.ret_test_X
            self.y_test = self.ret_test_y
        
        #split the training data into training and validation sets
        val_len = round(len(self.X_train)*0.7)
        self.X_val, self.y_val = self.X_train, self.y_train
        self.X_train, self.y_train = self.X_train[:val_len], self.y_train[:val_len]

        # Train and evaluate dataset on each of the below algorithms
        self.classifier_types = ['RandomForest','ElasticNet']
        self.next(self.param_select, foreach="classifier_types")

    @step
    def param_select(self) -> None:
        """
        Uses metaflow parallelization to run models with different hyperparameters
        """
        self.classifier_type = self.input

        if self.classifier_type == 'RandomForest':
            self.param = [5,10,15]
        else:
            self.param = [0.5,1,1.5]
        self.next(self.train_with_walk_forward_validation, foreach = "param")

    @step
    def train_with_walk_forward_validation(self) -> None:
        """
        Trains a random forest model on the training set and validates using walk forward validation
        """
        print(f"Starting {self.classifier_type}")

        self.param = self.input
        predictions = list()
        self.real = list()
        history_X = [x for x in self.X_train.values]
        history_Y = [y for y in self.y_train.values]
        
        for i in range(len(self.X_test), len(self.X_val)):
            
            testX = self.X_val.iloc[i].values
            
            #store actual values in a list
            testY = self.y_val.iloc[i]
            self.real.append(testY)

            # fit model on history and make a prediction
            if self.classifier_type == 'RandomForest':
                yhat = models.random_forest_forecast(history_X, history_Y, testX, self.param)
            else:
                yhat = models.elasticnet_forecast(history_X, history_Y, testX, self.param)
            # store forecast in list of predictions
            predictions.append(yhat)
            # add actual observation to history for the next loop
            history_X.append(self.X_val.iloc[i].values)     
            history_Y.append(self.y_val.iloc[i])    
        
        self.y_predicted = predictions
        
        self.val_r2_score = metrics.r2_score(self.real, self.y_predicted)
        print(f"Finishing {self.classifier_type}")
        self.next(self.select_param)

    @step
    def select_param(self, inputs) -> None:
        """
        select the best model from hyperparameter tuning
        """
        #Merge all common artifacts
        self.merge_artifacts(inputs, exclude=['y_predicted','val_r2_score', 'real', 'param'])
        
        #select the best model
        self.best_model_param = max(inputs, key = lambda x: x.val_r2_score).param
        
        #changing training set to include the validation set
        # self.X_train = max(inputs, key = lambda x: x.val_r2_score).X_val
        # self.y_train = max(inputs, key = lambda x: x.val_r2_score).y_val
        # self.X_test = max(inputs, key = lambda x: x.val_r2_score).X_test
        # self.y_test = max(inputs, key = lambda x: x.val_r2_score).y_test
        # self.classifier_type = max(inputs, key = lambda x: x.val_r2_score).classifier_type
        

        print(f"Best R2 score for validation set: {max(inputs, key = lambda x: x.val_r2_score).val_r2_score}")
        
        self.next(self.test_with_walk_forward_validation)
        
    @step
    def test_with_walk_forward_validation(self):
        """
        Train a Regression model on train set and predict on test set in a walk forward fashion
        """        
        self.classifier_type = self.input

        print(f"Model: {self.classifier_type}: parameter: {self.best_model_param}")
                
        predictions = list()
        history_X = [x for x in self.X_train.values]
        history_Y = [y for y in self.y_train.values]
        
        for i in range(len(self.X_test)):
            testX = self.X_test.iloc[i].values
            testY = self.y_test.iloc[i]
            # fit model on history and make a prediction
            if self.classifier_type == 'RandomForest':
                yhat = models.random_forest_forecast(history_X, history_Y, testX, self.best_model_param)
            elif self.classifier_type == 'ElasticNet':
                yhat = models.elasticnet_forecast(history_X, history_Y, testX, self.best_model_param)
            else:
                print("\n\n\nfndfdfndf\n\n\n")

            # store forecast in list of predictions
            predictions.append(yhat)
            # add actual observation to history for the next loop
            history_X.append(self.X_test.iloc[i].values)     
            history_Y.append(self.y_test.iloc[i])    
        
        self.y_predicted = predictions
        if self.classifier_type == 'RandomForest':
            self.model = models.fit_random_forest_classifier(history_X, history_Y, self.best_model_param)
        else:
            self.model = models.fit_elasticnet_classifier(history_X, history_Y, self.best_model_param)
        
        # go to the evaluation phase
        self.next(self.evaluate_classifier)  

    @step
    def evaluate_classifier(self):
        """
        Calculate resulting metrics from predictions for given classifier
        """

        # Create an experiment with your api key
        # experiment = Experiment(
        #     api_key="utAEwSyzdABdikKjhUItXWeFQ",
        #     project_name="finalproject-volatilityandexcessreturn",
        #     workspace="nyu-fre-7773-2021",
        # )
        # experiment.add_tag("Pipeline:" + str(self.pipeline_type))
        # experiment.add_tag("Model:" + str(self.classifier_type))

        self.r2 = metrics.r2_score(self.y_test, self.y_predicted)
        print('R2 score is {}'.format(self.r2))
        
        #experiment.log_parameters({"max_depth":})
        # experiment.log_metrics({"r2": self.r2})


        self.next(self.evaluate_pipeline)

    @step
    def evaluate_pipeline(self, inputs):
        # Merge all common artifacts
        self.merge_artifacts(inputs, exclude=['y_predicted','classifier_type','r2','model','best_model_param'])

        # print and store results and best model/params
        for clf in inputs:
            print(f" {clf.classifier_type} Classifier's R2 score {clf.r2} for {self.pipeline_type} Pipeline")

        best_model = max(inputs, key=lambda x: x.r2)
        self.best_r2 = best_model.r2
        self.best_classifier = best_model.classifier_type
        self.best_model = best_model.model

        self.next(self.combine_and_save_pipeline_results)


    @step
    def combine_and_save_pipeline_results(self, inputs):
        # Store results and best model in artifacts to use in Flask app
        self.merge_artifacts(inputs, exclude=['y_predicted','classifier_type','r2','best_r2','best_classifier','pipeline_type','X_train','X_test','y_train','y_test','best_model','model','vol_train_y', 'vol_test_y', 'ret_train_y', 'ret_test_y'])
        for pipelineResult in inputs:
            # Store best model and results for Volatility prediction
            if pipelineResult.pipeline_type == 'VolatilityPrediction':
                self.vol_best_r2 = pipelineResult.best_r2
                self.vol_best_model_type = pipelineResult.best_classifier
                self.vol_best_model = pipelineResult.best_model

            # Store best model and results for Market Return prediction
            else:
                self.er_best_r2 = pipelineResult.best_r2
                self.er_best_model_type = pipelineResult.best_classifier
                self.er_best_model = pipelineResult.best_model

        print('Pipelines joined!')
        self.next(self.end)


    @step
    def end(self):
        # all done, just print goodbye
        print("All done at {}!\n See you, space cowboys!".format(datetime.utcnow()))


if __name__ == '__main__':
    VolatilityAndExcessReturnPredictionFlow()
