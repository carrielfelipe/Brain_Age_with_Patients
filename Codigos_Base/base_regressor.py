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
from sklearn.linear_model import Lasso

class BaseRegressor:
    def __init__(self,save_path=None, scaler=None, params=None, params_space=None, fit_params_search=None, model_params_search=None,fit_params_train=None, models_params_train=None, name_model=None):             
        
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.params = params if params is not None else {}
        self.params_space = params_space if params_space is not None else {}
        self.fit_params_search = fit_params_search if fit_params_search is not None else {}
        self.model_params_search = model_params_search if model_params_search is not None else {}
        self.fit_params_train = fit_params_train if fit_params_train is not None else {}
        self.model_params_train = models_params_train if models_params_train is not None else {}

        self.fit_param={}

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


    def preprocess_data(self, X):        
        X_scaled = self.scaler.fit_transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns)
    
    def set_data(self,X,y):        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=1)
           
    def search_best_model(self,  X=None, y=None, param_space_=None, n_iter_=10, n_jobs_=-1, scoring_metric='neg_mean_absolute_error', type_model=1):
       
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

        self.opt_model = BayesSearchCV(
            estimator=model,
            search_spaces=param_space,
            fit_params=self.fit_param,
            cv=kf,
            n_iter=n_iter_,
            scoring=scoring_metric,
            n_jobs=n_jobs_,
            random_state=42,
            verbose=1
        )                
        self.opt_model.fit(X, y, **self.fit_params_search)
        best_params_return = dict(self.opt_model.best_params_)  

        return self.opt_model, best_params_return
    
    def get_eval_set(self):
        """Retorna el conjunto de evaluación actual."""
        return [(self.x_train_kf, self.y_train_kf)]
    
    def trainer(self, df_CN, df_patient=None, n_splits=10, n_iterations=20, params_=None, type_model=1, scaler=2, early_stopping_rounds=None):
    
        if params_ is None:
            params = self.params
        else:
            params = params_
        
        # Preparar el dataframe de controles
        X_CN = df_CN.iloc[:, :-2]  # Features
        y_CN = df_CN.iloc[:, -2]   # Labels (Age)
        ID_CN = df_CN.iloc[:, -1]  # IDs
        results_per_fold_CN_train = []
        results_per_fold_CN_test = []

        # Inicializar resultados
        results = {'model': [],
                   'mean_X_train_kf':[],
                   'std_X_train_kf':[],
                   'min_X_train_kf':[],
                   'max_X_train_kf':[],
                   'slope': [],
                   'intercept': [],
                   }
        
        results_labels_df_CN_train = pd.DataFrame(columns=['y_labels','y_pred','y_pred_corrected','GAP', 'GAP_corrected', 'ID-unique'])
        results_labels_df_CN_test = pd.DataFrame(columns=['y_labels', 'y_pred', 'y_pred_corrected', 'GAP', 'GAP_corrected', 'ID-unique'])

        if df_patient is not None:
            results_per_fold_patient = [[] for _ in df_patient]
        else:
            results_per_fold_patient = []

        results_labels_patient = []

        # Inicializar resultados por fold para pacientes
        # Si lista_dfs no es None, crear dataframes para almacenar resultados de pacientes
        if df_patient is not None:
            for _ in df_patient:
                results_labels_patient.append(pd.DataFrame(columns=['y_labels', 'y_pred', 'y_pred_corrected', 'GAP', 'GAP_corrected','ID-unique']))
                #results_per_fold_pat.append({})  # Diccionario por cada grupo de pacientes
        
        # Bucle de iteraciones
        for i in range(n_iterations):
            # Crear validación cruzada para CN
            kf_CN = KFold(n_splits=n_splits, shuffle=True, random_state=i)
            kf_CN_splits = list(kf_CN.split(X_CN, y_CN))

            # Crear validación cruzada para cada dataframe de pacientes si lista_dfs no es None
            if df_patient is not None:
                kf_splits_list = [list(KFold(n_splits=n_splits, shuffle=True, random_state=i).split(df.iloc[:, :-2], df.iloc[:, -2])) for df in df_patient]

            for fold in range(n_splits):
                # Obtener índices de entrenamiento y prueba para CN
                train_index_CN, test_index_CN = kf_CN_splits[fold]
                X_train_kf_CN, X_test_kf_CN = X_CN.iloc[train_index_CN], X_CN.iloc[test_index_CN]
                y_train_kf_CN, y_test_kf_CN = y_CN.iloc[train_index_CN], y_CN.iloc[test_index_CN]
                id_train_kf_CN = ID_CN.iloc[train_index_CN]
                id_test_kf_CN = ID_CN.iloc[test_index_CN]

                mean_X_train_kf = X_train_kf_CN.mean()
                std_X_train_kf = X_train_kf_CN.std()
                min_X_train_kf = X_train_kf_CN.min()
                max_X_train_kf = X_train_kf_CN.max()

                # Escalar los datos de acuerdo con el parámetro scaler
                if scaler == 1:
                    # No escalar
                    X_train_kf_CN_scaled = X_train_kf_CN
                    X_test_kf_CN_scaled = X_test_kf_CN
                elif scaler == 2:
                    # Z-score scaling                    
                    X_train_kf_CN_scaled = (X_train_kf_CN - mean_X_train_kf) / std_X_train_kf
                    X_test_kf_CN_scaled = (X_test_kf_CN - mean_X_train_kf) / std_X_train_kf
                elif scaler == 3:
                    # MinMax scaling (manual)                    
                    X_train_kf_CN_scaled = (X_train_kf_CN - min_X_train_kf) / (max_X_train_kf - min_X_train_kf)
                    X_test_kf_CN_scaled = (X_test_kf_CN - min_X_train_kf) / (max_X_train_kf - min_X_train_kf)         

                # Entrenar el modelo con CN
                if type_model == 1:
                    model = self.model_ml(**params, **self.model_params_train)
                if type_model == 2:
                    model = self.model_ml

                if early_stopping_rounds:
                    self.fit_params_train = {
                    "early_stopping_rounds": early_stopping_rounds,
                    "eval_set": "mae",
                    #"eval_set": self.get_eval_set(),
                    "eval_set": [(X_test_kf_CN_scaled, y_test_kf_CN)],
                    "verbose": False
                    }

                    
                model.fit(X_train_kf_CN_scaled, y_train_kf_CN,**self.fit_params_train)

                y_pred_CN_train = model.predict(X_train_kf_CN_scaled)
                gap_CN_train = y_pred_CN_train - y_train_kf_CN

                # Hacer predicciones para el conjunto de prueba de CN
                y_pred_CN_test = model.predict(X_test_kf_CN_scaled)
                gap_CN_test = y_pred_CN_test - y_test_kf_CN

                # Ajuste de GAP para CN
                slope, intercept, _, _, _ = linregress(y_train_kf_CN, gap_CN_train)
                corrected_gap_CN_train = gap_CN_train - (slope * y_train_kf_CN + intercept)
                corrected_gap_CN_test = gap_CN_test - (slope * y_test_kf_CN + intercept)
                y_pred_corrected_CN_test = y_pred_CN_test - (slope * y_test_kf_CN + intercept)
                y_pred_corrected_CN_train = y_pred_CN_train - (slope * y_train_kf_CN + intercept)

                # Guardar resultados de CN 
                temp_CN_df_test = pd.DataFrame({
                    'y_labels': y_test_kf_CN,
                    'y_pred': y_pred_CN_test,
                    'y_pred_corrected': y_pred_corrected_CN_test,
                    'GAP': gap_CN_test,
                    'GAP_corrected': corrected_gap_CN_test,
                    'ID-unique': id_test_kf_CN
                })
                temp_CN_df_train = pd.DataFrame({                    
                    'y_labels': y_train_kf_CN,
                    'y_pred': y_pred_CN_train,
                    'y_pred_corrected': y_pred_corrected_CN_train,
                    'GAP': gap_CN_train,
                    'GAP_corrected': corrected_gap_CN_train,
                    'ID-unique': id_train_kf_CN
                })

                results_labels_df_CN_train = pd.concat([results_labels_df_CN_train, temp_CN_df_train], ignore_index=True)
                results_per_fold_CN_train.append(temp_CN_df_train.copy())
                results_labels_df_CN_test = pd.concat([results_labels_df_CN_test, temp_CN_df_test], ignore_index=True)
                results_per_fold_CN_test.append(temp_CN_df_test.copy())

                # Procesar cada dataframe de pacientes si lista_dfs no es None
                if df_patient is not None:
                    for j, df in enumerate(df_patient):
                        train_index_pat, test_index_pat = kf_splits_list[j][fold]
                        X_train_pat = df.iloc[train_index_pat, :-2]
                        X_test_pat = df.iloc[test_index_pat, :-2]
                        y_test_pat = df.iloc[test_index_pat, -2]
                        id_test_pat = df.iloc[test_index_pat, -1]

                        # Escalar usando los parámetros de CN
                        X_test_pat_scaled = (X_test_pat - mean_X_train_kf) / std_X_train_kf

                        # Predicciones para el grupo de pacientes
                        y_pred_pat_test = model.predict(X_test_pat_scaled)
                        gap_pat = y_pred_pat_test - y_test_pat

                        # Ajuste de GAP para los pacientes
                        corrected_gap_pat = gap_pat - (slope * y_test_pat + intercept)
                        y_pred_corrected_pat = y_pred_pat_test - (slope * y_test_pat + intercept)

                        # Guardar resultados para cada grupo de pacientes
                        temp_pat_df = pd.DataFrame({
                            'y_labels': y_test_pat,
                            'y_pred': y_pred_pat_test,
                            'y_pred_corrected': y_pred_corrected_pat,
                            'GAP': gap_pat,
                            'GAP_corrected': corrected_gap_pat,
                            'ID-unique': id_test_pat
                        })
                        results_labels_patient[j] = pd.concat([results_labels_patient[j], temp_pat_df], ignore_index=True)
                        results_per_fold_patient[j].append(temp_pat_df.copy())  # Guardar en la lista simple

                # Guardar el modelo entrenado
                results['model'].append(model)
                
                results['mean_X_train_kf'].append(mean_X_train_kf)
                results['std_X_train_kf'].append(std_X_train_kf)
            
                results['min_X_train_kf'].append(min_X_train_kf)
                results['max_X_train_kf'].append(max_X_train_kf)
                
                results['slope'].append(slope)
                results['intercept'].append(intercept)

        return results_labels_df_CN_train, results_labels_df_CN_test, results_labels_patient, results, results_per_fold_CN_train,results_per_fold_CN_test, results_per_fold_patient


    def test(self):
        pass

    def avg_list(self, df_list):
        results_avg = []
        for df in df_list:            
            df_avg = df.groupby('ID-unique').agg({
                'y_labels': 'mean',
                'y_pred': 'mean',
                'y_pred_corrected': 'mean',
                'GAP': 'mean',
                'GAP_corrected': 'mean'
            }).reset_index()
            results_avg.append(df_avg)
        return results_avg

        
    def predicter(self, X_test=None):
        if X_test is None:
            X_test = self.X_test
        y_pred = self.model.predict(X_test)
        return y_pred   



    def regression_metrics(self, y_true, y_pred):
        """
        Calcula las métricas de regresión: MAE, MSE, RMSE y R2.
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return mae, mse, rmse, r2

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

    
    def calculate_simple_shap(self, X_train, X_test, model, random_seed=42):        
        try:
            self.explainer = shap.Explainer(model,X_train)
            shap_values = self.explainer.shap_values(X_test)
        except Exception as e:
            print("Fallo al usar shap.Explainer, intentando con shap.KernelExplainer:", e)
            try:
                np.random.seed(random_seed)
                self.explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 10), num_jobs=-1)
                shap_values = self.explainer.shap_values(X_test)
            except Exception as kernel_e:
                print("Fallo al usar shap.KernelExplainer:", kernel_e)
                return None, None 

        shap_sum = np.abs(shap_values).sum(axis=0)
        # Crear un diccionario para almacenar la suma de SHAP por característica
        shap_summary = {feature: shap_sum[i] for i, feature in enumerate(X_test.columns)}

        # Ordenar las características por su suma de SHAP
        shap_summary_sorted = sorted(shap_summary.items(), key=lambda x: x[1], reverse=True)

        # Imprimir el listado de importancia de características
        print("Importancia de características basada en suma de valores SHAP:")
        for feature, shap_sum in shap_summary_sorted:
            print(f"{feature}: {shap_sum}")
        
        return shap_values, shap_summary_sorted
    
    def calculate_multiple_shap(self, df_train, df_test, results_per_fold_train, results_per_fold_test, models_list, feature_col_range, iteration=20, kfolds_=10,  scaler=2, random_seed=42):
        shap_values_dict = {id_unique: [] for id_unique in df_test['ID-unique'].unique()}
        
        range_ = iteration*kfolds_
        
        for i in range(range_):

            # Train
            ID_train_fold = results_per_fold_train[i]['ID-unique']
            df_train_fold = df_train[df_train['ID-unique'].isin(ID_train_fold)]
            X_train_kf = df_train_fold.iloc[:, feature_col_range]  # Features
            y_train_kf = df_train_fold.iloc[:, -2]  # Labels

            # Test
            ID_test_fold = results_per_fold_test[i]['ID-unique']
            df_test_fold = df_test[df_test['ID-unique'].isin(ID_test_fold)]
            X_test_kf = df_test_fold.iloc[:, feature_col_range]  # Features
            y_test_kf = df_test_fold.iloc[:, -2]  # Labels

            if scaler == 1:
                # No escalar
                X_train_kf_scaled = X_train_kf
                X_test_kf_scaled = X_test_kf
            elif scaler == 2:
                # Z-score scaling
                mean_X_train_kf = X_train_kf.mean()
                std_X_train_kf = X_train_kf.std()
                X_train_kf_scaled = (X_train_kf - mean_X_train_kf) / std_X_train_kf
                X_test_kf_scaled = (X_test_kf - mean_X_train_kf) / std_X_train_kf
            elif scaler == 3:
                # MinMax scaling (manual)
                min_X_train_kf = X_train_kf.min()
                max_X_train_kf = X_train_kf.max()
                X_train_kf_scaled = (X_train_kf - min_X_train_kf) / (max_X_train_kf - min_X_train_kf)
                X_test_kf_scaled = (X_test_kf - min_X_train_kf) / (max_X_train_kf - min_X_train_kf)


            model_ = models_list[i]

            try:
                self.explainer = shap.Explainer(model_,X_train_kf_scaled)
                shap_values = self.explainer.shap_values(X_test_kf_scaled)
            except Exception as e:
                print("Fallo al usar shap.Explainer, intentando con shap.KernelExplainer:", e)
                try:
                    np.random.seed(random_seed)
                    self.explainer = shap.KernelExplainer(model_.predict, shap.sample(X_train_kf_scaled, 10), num_jobs=-1)
                    shap_values = self.explainer.shap_values(X_test_kf_scaled)
                except Exception as kernel_e:
                    print("Fallo al usar shap.KernelExplainer:", kernel_e)
                    return None, None 

            # SHAP calculation
            #explainer = shap.Explainer(model_, X_train_kf_scaled)
            #shap_values = explainer.shap_values(X_test_kf_scaled)

            # Store SHAP values
            for idx, id_unique in enumerate(df_test_fold['ID-unique']):
                shap_values_dict[id_unique].append(shap_values[idx])

        # Average SHAP values
        shap_values_avg_dict = {id_unique: np.mean(values, axis=0) for id_unique, values in shap_values_dict.items()}

        # Prepare SHAP summary matrix
        shap_values_avg_matrix = [shap_values_avg_dict[id_unique] for id_unique in df_test['ID-unique'].unique()]
        shap_values_avg_array = np.array(shap_values_avg_matrix)

        feature_names = X_test_kf_scaled.columns.tolist()

        shap_values_df = pd.DataFrame(shap_values_avg_array, columns=feature_names)
        shap_values_df['ID-unique'] = df_test['ID-unique'].unique()
        shap_values_df.set_index('ID-unique', inplace=True)

        # SHAP summary
        shap_sum = np.abs(shap_values_avg_array).sum(axis=0)
        shap_summary = {feature: shap_sum[i] for i, feature in enumerate(feature_names)}
        shap_summary_sorted = sorted(shap_summary.items(), key=lambda x: x[1], reverse=True)

        # Imprimir el listado de importancia de características
        print("Importancia de características basada en suma de valores SHAP:")
        for feature, shap_sum in shap_summary_sorted:
            print(f"{feature}: {shap_sum}")

        return shap_values_avg_array, shap_summary_sorted


    def shap_region(self, shap_summary_sorted, num_max=20):
        # Crear un diccionario para almacenar la suma de SHAP por región cerebral
        shap_por_region = {}

        # Recorrer la lista shap_summary_sorted
        for feature, shap_value in shap_summary_sorted[:num_max]:
            # Extraer la región cerebral (últimos dos textos separados por '_')
            region = feature.split('_')[-2] + '_' + feature.split('_')[-1]
            
            # Agregar la región cerebral al diccionario si no existe
            if region not in shap_por_region:
                shap_por_region[region] = 0.0
            
            # Sumar el valor SHAP al total de esa región cerebral
            shap_por_region[region] += shap_value
        
        max_value = max(shap_por_region.values())

        # Crear un diccionario para almacenar los valores normalizados
        resultado_normalizado = {}

        # Normalizar cada valor en el diccionario y almacenarlos en resultado_normalizado
        for region, suma_shap in shap_por_region.items():
            resultado_normalizado[region] = suma_shap / max_value

        # Ordenar shap_por_region y resultado_normalizado por valor descendente
        shap_por_region_sorted = {k: v for k, v in sorted(shap_por_region.items(), key=lambda item: item[1], reverse=True)}
        resultado_normalizado_sorted = {k: v for k, v in sorted(resultado_normalizado.items(), key=lambda item: item[1], reverse=True)}

        # Imprimir los valores normalizados ordenados
        for region, valor_normalizado in resultado_normalizado_sorted.items():
            print(f'{region}: {valor_normalizado:.6f}')

        return shap_por_region_sorted, resultado_normalizado_sorted
