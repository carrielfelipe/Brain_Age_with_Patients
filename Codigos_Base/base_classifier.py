import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import shap
from sklearn.linear_model import LinearRegression
import seaborn as sns
import os
from joblib import dump, load
from nilearn import plotting
import statsmodels.api as sm
from sklearn.model_selection import StratifiedShuffleSplit

from scipy.stats import linregress


class BaseClassifier:
    def __init__(self,save_path=None, scaler=None, params=None, params_space=None, fit_params_search=None, model_params_search=None,fit_params_train=None, models_params_train=None, name_model=None):             
       
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.params = params if params is not None else {}
        self.params_space = params_space if params_space is not None else {}
        self.fit_params_search = fit_params_search if fit_params_search is not None else {}
        self.model_params_search = model_params_search if model_params_search is not None else {}
        self.fit_params_train = fit_params_train if fit_params_train is not None else {}
        self.model_params_train = models_params_train if models_params_train is not None else {}

        self.save_path = save_path
        self.model_ml = None
        self.name_model = name_model
        self.model = None
        self.opt_model = None
        self.explainer = None

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.x_train_kf = None
        self.y_train_kf= None

        self.early_stopping_rounds = 10

        self.residual_model = None


    def search_best_model(self,  X=None, y=None, param_space_=None, n_iter_=10, n_jobs_=-1, scoring_metric='accuracy', type_model=1):
       
        if X is None:
            X = self.X_train
        if y is None:
            y = self.y_train

        if param_space_ is None:
            param_space = self.params_space
        else:
            param_space = param_space_

        n_splits = 10
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=126)       
        
        if type_model == 1:
            model = self.model_ml(**self.model_params_search)
        if type_model == 2:
            model = self.model_ml 

        opt_model = BayesSearchCV(
            estimator=model,
            search_spaces=param_space,
            #fit_params=self.fit_param,
            cv=kf,
            n_iter=n_iter_,
            #scoring=scoring_metric,
            n_jobs=n_jobs_,
            random_state=42,
            verbose=1
        )                
        opt_model.fit(X, y, **self.fit_params_search)
        best_params_return = dict(opt_model.best_params_)  

        return opt_model, best_params_return
    

    def trainer(self, df, n_splits=10, n_iterations=20, params_=None, type_model=1, scaler=2, early_stop=False, id='ID-unique-2'):
        
        if params_ is None:
            params = self.params
        else:
            params = params_


        
        
        # Preparar el dataframe de controles
        X = df.iloc[:, :-2]  # Features
        y = df.iloc[:, -2]   # Labels (Age)
        ID = df.iloc[:, -1]  # IDs
        results_per_fold_train = []
        results_per_fold_test = []
                
        results_labels_df_train = pd.DataFrame(columns=['y_labels','y_pred','y_prob', id])
        results_labels_df_test = pd.DataFrame(columns=['y_labels', 'y_pred','y_prob', id])

        # Inicializar resultados
        results = {'model': [],
                    'mean_X_train_kf':[],
                    'std_X_train_kf':[],
                    'min_X_train_kf':[],
                    'max_X_train_kf':[],
                    'slope': [],
                    'intercept': [],
                    }
        
        # Bucle de iteraciones
        for i in range(n_iterations):
            # Crear validación cruzada para CN
            kf_CN = KFold(n_splits=n_splits, shuffle=True, random_state=i)
            kf_splits = list(kf_CN.split(X, y))
            
            for fold in range(n_splits):
                # Obtener índices de entrenamiento y prueba para CN
                train_index, test_index = kf_splits[fold]
                X_train_kf, X_test_kf_CN = X.iloc[train_index], X.iloc[test_index]
                y_train_kf, y_test_kf_CN = y.iloc[train_index], y.iloc[test_index]
                id_train_kf = ID.iloc[train_index]
                id_test_kf = ID.iloc[test_index]

                mean_X_train_kf = X_train_kf.mean()
                std_X_train_kf = X_train_kf.std()
                min_X_train_kf = X_train_kf.min()
                max_X_train_kf = X_train_kf.max()

                # Escalar los datos de acuerdo con el parámetro scaler
                if scaler == 1:
                    # No escalar
                    X_train_kf_scaled = X_train_kf
                    X_test_kf_scaled = X_test_kf_CN
                elif scaler == 2:
                    # Z-score scaling                    
                    X_train_kf_scaled = (X_train_kf - mean_X_train_kf) / std_X_train_kf
                    X_test_kf_scaled = (X_test_kf_CN - mean_X_train_kf) / std_X_train_kf
                elif scaler == 3:
                    # MinMax scaling (manual)                    
                    X_train_kf_scaled = (X_train_kf - min_X_train_kf) / (max_X_train_kf - min_X_train_kf)
                    X_test_kf_scaled = (X_test_kf_CN - min_X_train_kf) / (max_X_train_kf - min_X_train_kf)

                self.x_train_kf = X_train_kf_scaled
                self.y_train_kf=y_train_kf


                # Entrenar el modelo con CN
                if type_model == 1:
                    model = self.model_ml(**params, **self.model_params_train)
                if type_model == 2:
                    model = self.model_ml

                if early_stop:
                    self.fit_params_train = {
                    "early_stopping_rounds": self.early_stopping_rounds,
                    "eval_set": "mae",
                    "eval_set": self.get_eval_set(),
                    "verbose": False
                    }

                    
                model.fit(X_train_kf_scaled, y_train_kf,**self.fit_params_train)

                y_pred_CN_train = model.predict(X_train_kf_scaled)
                y_prob_CN_train = model.predict_proba(X_train_kf_scaled)[:, 1]

                # Hacer predicciones para el conjunto de prueba de CN
                y_pred_CN_test = model.predict(X_test_kf_scaled)
                y_prob_CN_test = model.predict_proba(X_test_kf_scaled)[:, 1]
                
                # Guardar resultados de CN 
                temp_CN_df_test = pd.DataFrame({
                    'y_labels': y_test_kf_CN,
                    'y_pred': y_pred_CN_test,
                    'y_prob':y_prob_CN_test,                    
                    id: id_test_kf
                })
                temp_CN_df_train = pd.DataFrame({                    
                    'y_labels': y_train_kf,
                    'y_pred': y_pred_CN_train, 
                    'y_prob':y_prob_CN_train  ,                 
                    id: id_train_kf
                })

                results_labels_df_train = pd.concat([results_labels_df_train, temp_CN_df_train], ignore_index=True)
                results_per_fold_train.append(temp_CN_df_train.copy())
                results_labels_df_test = pd.concat([results_labels_df_test, temp_CN_df_test], ignore_index=True)
                results_per_fold_test.append(temp_CN_df_test.copy())

                # Procesar cada dataframe de pacientes si lista_dfs no es None
                
                # Guardar el modelo entrenado
                results['model'].append(model)
                
                results['mean_X_train_kf'].append(mean_X_train_kf)
                results['std_X_train_kf'].append(std_X_train_kf)
            
                results['min_X_train_kf'].append(min_X_train_kf)
                results['max_X_train_kf'].append(max_X_train_kf)
                
                    

        return results_labels_df_train, results_labels_df_test, results, results_per_fold_train,results_per_fold_test


    def best_hyper(self, opt_model, num_best=10, num_max=400):
        """
        Obtiene los mejores hiperparámetros para las mejores puntuaciones de validación cruzada dentro de los primeros num_max resultados.
        
        """
        results = opt_model.cv_results_
        errors = results['mean_test_score'][:num_max]  # Considerar solo los primeros num_max resultados
        best_idx = np.argsort(errors)[-num_best:]  # Obtener los índices de las mejores puntuaciones
        best_hypers = []

        for idx in best_idx:
            hyper = {}
            for param, value in results['params'][idx].items():
                hyper[param] = value
            best_hypers.append(hyper)

        # Invertir el orden para que el mejor esté en el índice 0
        best_hypers = best_hypers[::-1]

        return best_hypers
    
    def avg_list(self, df_list, id='ID-unique-2'):
        results_avg = []
        for df in df_list:            
            df_avg = df.groupby(id).agg({
                'y_labels': 'mean',
                'y_pred': 'mean', 
                'y_prob': 'mean'               
            }).reset_index()
            results_avg.append(df_avg)
        return results_avg
    
    def avg_list_threshold(self, df_list, id='ID-unique-2'):
        results_avg = []
        for df in df_list:
            df_avg = df.groupby(id).agg({
                'y_labels': 'mean',
                'y_pred': 'mean',
                'y_prob': 'mean'
            }).reset_index()
            
            # Convertir el promedio de y_pred a 0 o 1 según el umbral de 0.5
            df_avg['y_pred'] = (df_avg['y_pred'] >= 0.5).astype(int)
            results_avg.append(df_avg)
        return results_avg
    
    

    def majority_vote(self, df_list, id='ID-unique-2'):
        results_avg = []
        for df in df_list:
            # Tomar la moda (valor más frecuente) para y_labels y y_pred
            df_avg = df.groupby(id).agg({
                'y_labels': lambda x: x.mode()[0],
                'y_pred': lambda x: x.mode()[0],
                'y_prob': lambda x: x.mode()[0]
            }).reset_index()
            results_avg.append(df_avg)
        return results_avg