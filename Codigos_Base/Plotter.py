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


class Plotter:
    def __init__(self):
        pass

    def regression_metrics(self, y_true, y_pred):
        """
        Calcula las métricas de regresión: MAE, MSE, RMSE y R2.
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return mae, mse, rmse, r2



    def plot_search_best_model(self, opt_model, num_max=100, x_size=4, y_size=4, color='navy', linewidth=1, legend_result=True, font='DejaVu Sans', fontsize='12'):
        """
        Imprime los mejores parámetros encontrados y la evolución del error absoluto medio negativo.
        """
        resultados = opt_model.cv_results_
        errores = resultados['mean_test_score'][:num_max]        
        iteraciones = list(range(1, len(errores) + 1))

        threshold = 1000
        array = np.array(errores)
        min_value = np.min(errores[np.abs(errores) <= threshold])
        array_replaced = np.where(np.abs(array) > threshold, min_value, array)
        replaced_mask = np.abs(array) > threshold

        best_indices = np.argsort(errores)[-10:]  # Obtener los índices de los 10 mejores resultados
        best_iteraciones = np.array(iteraciones)[best_indices]  # Obtener las iteraciones correspondientes a los mejores resultados
        best_errores = np.array(errores)[best_indices]  # Obtener los errores correspondientes a los mejores resultados

        plt.figure(figsize=(x_size, y_size))
        plt.plot(iteraciones, array_replaced,  linestyle='-', color=color, alpha=1,linewidth=linewidth)
        plt.scatter(np.array(iteraciones)[replaced_mask], array_replaced[replaced_mask], color='green', s=40, edgecolor='green', linewidth=2, label='Valor Atípico')  # Leyenda para outliers
        plt.xlabel('Iteración', fontweight='bold', fontsize=12)
        plt.ylabel('Error Absoluto Medio Negativo', fontweight='bold', fontsize=12)
        plt.title('Búsqueda de Hiperparámetros', fontweight='bold', fontsize=14)
        plt.grid(True)

        # Resaltar los 10 mejores resultados con puntos rojos
        plt.scatter(best_iteraciones, best_errores, color='red', s=40, edgecolor='red', linewidth=2, label='Mejor Iteración')  # Leyenda para mejores iteraciones
        
        # Ajustar el grosor y color del borde interior del cuadro
        for spine in plt.gca().spines.values():
            spine.set_linewidth(2)
            spine.set_color('black')
        # Configurar números de los ejes en negrita
        plt.xticks(fontweight='bold')
        plt.yticks(fontweight='bold')
        
        if legend_result:
        # Agregar texto con los mejores errores como leyenda en la esquina inferior derecha
            legend_text = "\n".join([f"{i}.- Iter: {it}, Error: {err:.2f}" for i, (it, err) in enumerate(zip(best_iteraciones[::-1], best_errores[::-1]), 1)])
            plt.text(0.98, 0.2, legend_text, transform=plt.gca().transAxes, fontsize=9, verticalalignment='bottom', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.2, edgecolor='black', boxstyle='round,pad=0.5'))
            # Color de fondo para el texto de leyenda
            bbox_props = dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3, edgecolor='black')
            plt.text(0.98, 0.2, legend_text, transform=plt.gca().transAxes, fontsize=9, verticalalignment='bottom', horizontalalignment='right', bbox=bbox_props)

        plt.legend(fontsize=9, loc='lower right', bbox_to_anchor=(1, 0))  # Ajustar tamaño y posición de la leyenda

        
        plt.show() 

        # Imprimir los 10 mejores errores y las iteraciones correspondientes en orden inverso
        for i, (it, err) in enumerate(zip(best_iteraciones[::-1], best_errores[::-1]), 1):
            print(f"Top {i}: Iteración {it}, Error {err}")



    def plot_iteration(self, y, x_size=4, y_size=4, color='navy',  linewidth=1, legend_result=False, best_result=False, xlabel='', ylabel='', title='', band_width=0.5, font='DejaVu Sans', fontsize=12, weight='normal',mode=1):
            """
            Imprime los mejores parámetros encontrados y la evolución del error absoluto medio negativo.
            """                 
            iteraciones = list(range(1, len(y) + 1))

            plt.figure(figsize=(x_size, y_size))

            # Definir los límites superior e inferior para la banda
            lower_bound = y - band_width
            upper_bound = y + band_width
            plt.fill_between(iteraciones, lower_bound, upper_bound, color=color, alpha=0.2)

            plt.plot(iteraciones, y,  linestyle='-', color=color, alpha=1, linewidth=linewidth)
            plt.xlabel(xlabel, fontweight=weight,fontname=font, fontsize=fontsize)
            plt.ylabel(ylabel, fontweight=weight,fontname=font, fontsize=fontsize)
            plt.title(title, fontweight=weight, fontname=font, fontsize=fontsize+2)
            plt.grid(True)

            # Ajustar el grosor y color del borde interior del cuadro
            for spine in plt.gca().spines.values():
                spine.set_linewidth(1)
                spine.set_color('black')

            # Configurar números de los ejes en negrita
            #plt.xticks(fontweight='bold')
            #plt.yticks(fontweight='bold')

            if best_result:

                best_indices = np.argsort(y)[-10:]  # Obtener los índices de los 10 mejores resultados
                best_iteraciones = np.array(iteraciones)[best_indices]  # Obtener las iteraciones correspondientes a los mejores resultados
                best_errores = np.array(y)[best_indices]  # Obtener los errores correspondientes a los mejores resultados
                # Resaltar los 10 mejores resultados con puntos rojos
                plt.scatter(best_iteraciones, best_errores, color='red', s=40, edgecolor='red', linewidth=2, label='Best Score')  

               
                if legend_result:
                    # Agregar texto con los mejores errores como leyenda en la esquina inferior derecha
                    legend_text = "\n".join([f"{i}.- Iter: {it}, Error: {err:.2f}" for i, (it, err) in enumerate(zip(best_iteraciones[::-1], best_errores[::-1]), 1)])
                    plt.text(0.98, 0.2, legend_text, transform=plt.gca().transAxes, fontsize=9, verticalalignment='bottom', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.2, edgecolor='black', boxstyle='round,pad=0.5'))
                    # Color de fondo para el texto de leyenda
                    bbox_props = dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3, edgecolor='black')
                    plt.text(0.98, 0.2, legend_text, transform=plt.gca().transAxes, fontsize=9, verticalalignment='bottom', horizontalalignment='right', bbox=bbox_props)

            plt.legend(fontsize=9, loc='lower right', bbox_to_anchor=(1, 0))  # Ajustar tamaño y posición de la leyenda

            if mode == 1 :
                # Ajustar el grosor y color del borde interior del cuadro
                for spine in plt.gca().spines.values():
                    spine.set_linewidth(1)
                    spine.set_color('black')
            if mode ==  2:
                # Ajustar el grosor y color del borde interior del cuadro
                ax = plt.gca()
                ax.spines['top'].set_visible(False)  # Ocultar borde superior
                ax.spines['right'].set_visible(False)  # Ocultar borde derecho
                ax.spines['left'].set_linewidth(1)
                ax.spines['left'].set_color('black')
                ax.spines['bottom'].set_linewidth(1)
                ax.spines['bottom'].set_color('black')

            plt.show() 

            if best_result:
                 # Imprimir los 10 mejores errores y las iteraciones correspondientes en orden inverso
                for i, (it, err) in enumerate(zip(best_iteraciones[::-1], best_errores[::-1]), 1):
                    print(f"Top {i}: Iteration {it}, Score {err}")




    


    def plot_regresion(self,x ,y , x_size=4, y_size=4, label_='', color='navy', alpha=0.5, color_line_fit='red', color_line_ideal='purple', color_confidence_interval='blue' , alpha_confidence_interval=0.6, title='',
                       xlabel='',ylabel='',label=True, line_fit=True, line_ideal=True , confidence_interval=True,  x_ticks_step=10, y_ticks_step=10, mode=1,
                        x_min_limit=None, x_max_limit=None, y_min_limit=None, y_max_limit=None, legend= True,legend_metrics=True, font='DejaVu Sans', fontsize=12, weight='normal', details = True,
                        xticks =1,yticks=1, print_metrics=True):
        """
        Grafica los resultados de la regresión, incluyendo las métricas de rendimiento para cada conjunto (entrenamiento, validación, prueba).
        """
        y_pred_test = y
        y_test = x
        
        # Ajustar modelo de regresión lineal a las edades cronológicas vs. las edades predichas
        x_test_with_constant = sm.add_constant(x)  # Agregar constante (intercepto)
        linear_model = sm.OLS(y, x_test_with_constant).fit()
        if details:
            print(linear_model.summary())

        # Obtener los límites del gráfico
        x_min = x_min_limit if x_min_limit is not None else int(min(y_test) // x_ticks_step * x_ticks_step)
        x_max = x_max_limit if x_max_limit is not None else int(max(y_test) // x_ticks_step * x_ticks_step + x_ticks_step)

        # Generar x_pred desde el límite inferior hasta el superior
        x_pred = np.linspace(x_min, x_max, 100)
        x_pred_with_constant = sm.add_constant(x_pred)
        y_pred_line = linear_model.predict(x_pred_with_constant)
        pred_ci = linear_model.get_prediction(x_pred_with_constant).conf_int()
        
        plt.figure(figsize=(x_size, y_size))
        plt.scatter(x, y, alpha=alpha, color=color,label=label_)
        
        if line_ideal:
            plt.plot([0,100], [0,100], color=color_line_ideal, linestyle='-', linewidth=2, label='Ideal Line')
        if line_fit:
            #plt.plot(y_test.values, m *y_test.values + c, color=color_line_fit, linestyle='-', linewidth=2, label='recta ajustada')
            plt.plot(x_pred, y_pred_line, color=color_line_fit,linestyle='-', linewidth=2, label='Fitted Line')
        if confidence_interval:   
            plt.fill_between(x_pred, pred_ci[:, 0], pred_ci[:, 1], color=color_confidence_interval, alpha=alpha_confidence_interval, label='Confidence Interval')

        plt.title(title, fontweight=weight, fontname=font, fontsize=fontsize+2)

        
        
        if(label):
            plt.xlabel(xlabel, fontweight=weight,fontname=font, fontsize=fontsize)
            plt.ylabel(ylabel, fontweight=weight,fontname=font, fontsize=fontsize)
        

        #p_value = linear_model.pvalues[1]
        
        # Mostrar las métricas de evaluación directamente en el gráfico
        mae_model, mse_model, rmse_model, r2_model = self.regression_metrics(y_true=x, y_pred=y)
        metrics_str = (
            f'MAE: {mae_model:.2f}\n'
            f'MSE: {mse_model:.2f}\n'
            f'RMSE: {rmse_model:.2f}\n'
            f'R²: {r2_model:.2f}'
        )

        if legend_metrics:
            plt.text(0.05, 0.95, metrics_str, transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8),fontsize=fontsize-3, font=font)
        

        if mode == 1 :
            # Ajustar el grosor y color del borde interior del cuadro
            for spine in plt.gca().spines.values():
                spine.set_linewidth(1)
                spine.set_color('black')
        if mode ==  2:
            # Ajustar el grosor y color del borde interior del cuadro
            ax = plt.gca()
            ax.spines['top'].set_visible(False)  # Ocultar borde superior
            ax.spines['right'].set_visible(False)  # Ocultar borde derecho
            ax.spines['left'].set_linewidth(1)
            ax.spines['left'].set_color('black')
            ax.spines['bottom'].set_linewidth(1)
            ax.spines['bottom'].set_color('black')

        if mode ==  3:
            for spine in plt.gca().spines.values():
                spine.set_linewidth(0.2)
                spine.set_color('black')
            
    

        #Configurar números de los ejes en negrita
        #plt.xticks(fontweight='bold')
        #plt.yticks(fontweight='bold')

        # Definir límites de los ejes
        x_min = x_min_limit if x_min_limit is not None else int(min(y_test) // x_ticks_step * x_ticks_step)
        x_max = x_max_limit if x_max_limit is not None else int(max(y_test) // x_ticks_step * x_ticks_step + x_ticks_step)
        y_min = y_min_limit if y_min_limit is not None else int(min(y_pred_test) // y_ticks_step * y_ticks_step)
        y_max = y_max_limit if y_max_limit is not None else int(max(y_pred_test) // y_ticks_step * y_ticks_step + y_ticks_step)

        plt.xticks(np.arange(x_min, x_max + x_ticks_step, x_ticks_step))
        plt.yticks(np.arange(y_min, y_max + y_ticks_step, y_ticks_step))

        # Establecer los límites de los ejes
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.grid(True)

        if xticks==2:
            plt.xticks(fontweight='bold')
        if xticks ==3:
            #plt.xticks(np.arange(x_min, x_max + x_ticks_step, x_ticks_step), [''] * len(np.arange(x_min, x_max + x_ticks_step, x_ticks_step)))
            plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)


        if yticks==2:
            plt.yticks(fontweight='bold')
        if yticks ==3:
            plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)




        if legend:
            plt.legend(loc='lower right')
        plt.show()

        if print_metrics:
            # Impresión de las métricas de evaluación en la consola
            print(f'Error medio absoluto: {mae_model}')
            print(f'Error cuadrático medio: {mse_model}')
            print(f'Raíz del error cuadrático medio: {rmse_model}')
            print(f'Coeficiente de determinación (R²): {r2_model}')
            




    def plot_feature_importance(self, shap_values, X_test, y_test, max_features=20, font='DejaVu Sans', fontsize=18, xlabel1='',xlabel2='',ylabel='', ylabel1='', ylabel2='', theme1="seismic",theme2='black', theme3="viridis"):
        # Calcular la suma de los valores absolutos de SHAP para cada característica
        shap_sum = np.abs(shap_values).sum(axis=0)
        
        # Obtener los índices de las características más importantes
        important_indices = np.argsort(shap_sum)[-max_features:]
        
        # Filtrar los valores SHAP y los datos de prueba para las características más importantes
        filtered_shap_values = shap_values[:, important_indices]
        filtered_X_test = X_test.iloc[:, important_indices]
        
        # Primer gráfico: resumen de SHAP
        shap.summary_plot(filtered_shap_values, filtered_X_test, cmap=sns.color_palette(theme1, as_cmap=True), show=False)
        plt.xlabel(xlabel1, fontsize=fontsize,font=font, fontweight='bold')  # Cambiar nombre del eje X
        plt.ylabel(ylabel, fontsize=fontsize,font=font, fontweight='bold')  # Cambiar nombre del eje Y
        plt.xticks(fontsize=fontsize-2)  # Ajustar tamaño de fuente de las etiquetas del eje X
        plt.yticks(fontsize=fontsize-2)  # Ajustar tamaño de fuente de las etiquetas del eje Y
        
        # Ajustar la barra de colores del primer gráfico
        cbar = plt.gcf().get_axes()[-1]  # Obtener la última axis que corresponde a la colorbar
        cbar.tick_params(labelsize=14)  # Ajustar tamaño de fuente de las etiquetas de la barra de color
        cbar.set_ylabel(ylabel1, fontsize=fontsize,font=font, fontweight='bold')  # Cambiar el nombre y ajustar el tamaño y peso del texto del eje Y de la barra de color
        
        plt.show()


        shap.summary_plot(filtered_shap_values, filtered_X_test, plot_type="bar", show=False)
        plt.xlabel(xlabel2,fontsize=fontsize,font=font, fontweight='bold')  # Cambiar nombre del eje X
        plt.ylabel(ylabel, fontsize=fontsize,font=font, fontweight='bold')  # Cambiar nombre del eje Y
        plt.xticks(fontsize=fontsize-6)  # Ajustar tamaño de fuente de las etiquetas del eje X
        plt.yticks(fontsize=fontsize-2)  # Ajustar tamaño de fuente de las etiquetas del eje Y
        # Obtener las posiciones de las marcas del eje X
        x_ticks = plt.gca().get_xticks()

        # Mostrar solo algunas de las marcas en el eje X
        plt.gca().set_xticks(x_ticks[::2])  # Cambia el valor '2' por el paso que desees

        #plt.gca().set_facecolor('black')  # Cambiar el color de fondo del gráfico a negro
        bars = plt.gca().patches
        for bar in bars:
            bar.set_edgecolor(theme2)  # Cambiar el borde de las barras a negro
            bar.set_facecolor(theme2)  # Cambiar el color de las barras a blanco
        plt.show()

        
        # Calcular correlaciones para las características más importantes
        X_ = filtered_X_test.values
        Scaler = StandardScaler()
        Scaler.fit(X_)
        X_scaled=Scaler.transform(X_)
        y_ = y_test.values
        correlations = np.array([np.corrcoef(X_scaled[:, i], y_.flatten())[0, 1] for i in range(X_.shape[1])])
        
        # Crear una paleta de colores basada en la correlación
        cmap = sns.color_palette(theme3, as_cmap=True)
        norm = plt.Normalize(vmin=np.min(correlations), vmax=np.max(correlations))
        colors = cmap(norm(correlations))
        
        # Segundo gráfico: resumen de SHAP en modo de barras
        bar_fig, bar_ax = plt.subplots()
        shap.summary_plot(filtered_shap_values, filtered_X_test, plot_type="bar", show=False, max_display=max_features)
        bars = bar_ax.patches

        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Añadir barra de colores (colorbar)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=bar_ax)
        cbar.set_label(ylabel2, fontsize=fontsize,font=font, fontweight='bold')
        cbar.ax.tick_params(labelsize=16)  # Ajustar tamaño de fuente de las etiquetas de la barra de color
        
        bar_ax.set_ylabel(ylabel, fontsize=fontsize,font=font, fontweight='bold')
        bar_ax.set_xlabel(xlabel2, fontsize=fontsize,font=font, fontweight='bold')  # Cambiar nombre del eje X
        bar_ax.tick_params(axis='y', labelsize=14)  # Ajustar tamaño de fuente de las etiquetas del eje Y
        plt.xticks(fontsize=fontsize-2)  # Ajustar tamaño de fuente de las etiquetas del eje X
        plt.yticks(fontsize=fontsize-2)  # Ajustar tamaño de fuente de las etiquetas del eje Y
        
        plt.show()







    def plot_normalized_values(self, region_values_dict, color='red', name_set='',x_size=4,y_size=4, xlabel='', ylabel='',font='DejaVu Sans', fontsize=12, weight='normal'):
        # Extraer nombres de las regiones y valores normalizados
        regiones = list(region_values_dict.keys())
        valores = list(region_values_dict.values())
        
        # Crear el gráfico de barras horizontal
        fig, ax = plt.subplots(figsize=(x_size, y_size))
        ax.barh(regiones, valores, color=color)
        
        # Añadir etiquetas y título
        ax.set_xlabel(xlabel,fontweight=weight,fontname=font, fontsize=fontsize)
        ax.set_ylabel(ylabel,fontweight=weight,fontname=font, fontsize=fontsize)
        ax.set_title(name_set,fontweight=weight,fontname=font, fontsize=fontsize+2)
        
        # Mostrar el gráfico
        plt.show()


    def plot_metricas_evaluacion(self, results,labels=None, name_set='Cross Validation'):
        metrics = ['mae', 'mse', 'rmse', 'r2']
        # Graficar las métricas para cada conjunto
        plt.figure(figsize=(7, 7))
        for i, metric in enumerate(metrics):
            plt.subplot(2, 2, i + 1)

            for i, df in enumerate(results):
                label = labels[i]
                plt.plot(range(1, 11), results[label][metric], label=label)
            #plt.plot(range(1, 11), results['val'][metric], label='Validation')
            #plt.plot(range(1, 11), results['test'][metric], label='Test')
            
            plt.title(f'{metric.upper()} - {name_set}')
            plt.xlabel('Fold', fontweight='bold', fontsize=12)
            plt.ylabel(f'{metric.upper()}', fontweight='bold', fontsize=12)
            plt.legend()
            # Ajustar el grosor y color del borde interior del cuadro
            for spine in plt.gca().spines.values():
                spine.set_linewidth(2)
                spine.set_color('black')
            
            #Configurar números de los ejes en negrita
            plt.xticks(fontweight='bold')
            plt.yticks(fontweight='bold')
        plt.tight_layout()        

        plt.show()


    def plot_metrica_evaluacion(self, results, metric='mae', name_set='Cross Validation',font='DejaVu Sans', fontsize=12, weight='normal', x_size=4, y_size=4, mode=1):
        if metric not in ['mae', 'mse', 'rmse', 'r2']:
            raise ValueError("La métrica debe ser una de 'mae', 'mse', 'rmse', 'r2'")
        
        # Graficar la métrica seleccionada
        plt.figure(figsize=(x_size, y_size))
        plt.plot(range(1, 11), results['train'][metric], label='Training')
        plt.plot(range(1, 11), results['val'][metric], label='Validation')
        plt.plot(range(1, 11), results['test'][metric], label='Test')
        plt.title(f'{metric.upper()} - {name_set}', fontweight=weight,fontname=font, fontsize=fontsize+2)
        plt.xlabel('Fold',  fontweight=weight,fontname=font, fontsize=fontsize)
        plt.ylabel(f'{metric.upper()}', fontweight=weight,fontname=font, fontsize=fontsize)
        plt.legend(prop={'family': font, 'size': fontsize-2, 'weight': weight})
        
        if mode == 1 :
            # Ajustar el grosor y color del borde interior del cuadro
            for spine in plt.gca().spines.values():
                spine.set_linewidth(1)
                spine.set_color('black')
        if mode ==  2:
            # Ajustar el grosor y color del borde interior del cuadro
            ax = plt.gca()
            ax.spines['top'].set_visible(False)  # Ocultar borde superior
            ax.spines['right'].set_visible(False)  # Ocultar borde derecho
            ax.spines['left'].set_linewidth(1)
            ax.spines['left'].set_color('black')
            ax.spines['bottom'].set_linewidth(1)
            ax.spines['bottom'].set_color('black')
        
        # Configurar números de los ejes en negrita
        plt.xticks(fontweight='bold')
        plt.yticks(fontweight='bold')
        
        plt.tight_layout()        

        plt.show()



        


       
    def plot_brain_regions(self, shap_values, base_path, color='red'):
        # Inicializar la imagen base con un umbral alto para crear una imagen en blanco
        display = plotting.plot_glass_brain(base_path + 'OCC.nii.gz', threshold=10)

        # Obtener el número de regiones
        num_regions = len(shap_values)

        # Definir los valores de linewidths para el borde negro
        linewidths = [1.5, 1, 0.5] + [0.2] * (num_regions - 3)

        # Iterar sobre los valores SHAP normalizados y agregar contornos y bordes
        for idx, (region, alpha) in enumerate(shap_values.items()):
            region_path = base_path + f'{region}.nii.gz'
            display.add_contours(region_path, levels=[1.], filled=True, alpha=alpha, colors=color)
            display.add_contours(region_path, levels=[1.], alpha=1, colors='black', linewidths=linewidths[idx])

        # Mostrar la imagen final
        plotting.show()






    def plot_gap_distribution(self, df_errors, colores_personalizados, x_limits=(-4, 4), title='GAP Distribution',
                           x_size=4, y_size=4, xlabel='', ylabel='', font='DejaVu Sans', fontsize=12, weight='normal'):
        # Crear el diagrama de cajas y los puntos individuales en forma horizontal
        plt.figure(figsize=(x_size, y_size))

        # Boxplot con la mediana (por defecto) y la media personalizada
        sns.boxplot(y='Grupo', x='Error', data=df_errors, showfliers=False, palette=colores_personalizados, 
                    showmeans=True, 
                    meanline=True, 
                    meanprops={"color": "blue", "ls": "--", "linewidth": 1.5})  # Líneas de la media personalizadas

        # Agregar puntos individuales
        sns.stripplot(y='Grupo', x='Error', data=df_errors, color='black', size=5, alpha=0.7, jitter=True)

        # Título y etiquetas
        plt.title(title, fontsize=fontsize+2, fontweight=weight, fontname=font)
        plt.ylabel(ylabel, fontsize=fontsize, fontname=font)
        plt.xlabel(xlabel, fontsize=fontsize, fontname=font)
        plt.xlim(x_limits)
        plt.xticks(np.arange(x_limits[0], x_limits[1], 1))
        plt.grid(True)
        
        # Modificar las etiquetas del eje y
        ax = plt.gca()  # Obtener el eje actual
        ax.set_yticklabels(ax.get_yticklabels(), fontname=font, fontsize=fontsize-2, fontweight=weight)

        # Mostrar gráfico
        plt.show()


    def plot_regression_diagnosis(self, df_list, colors, labels, title='',x_size=4, y_size=4, xlabel='',ylabel='',font='DejaVu Sans', fontsize=12, weight='normal', legend= True,
                                x_ticks_step=10, y_ticks_step=10, mode=1, line_ideal=True, alpha=0.5, color_line_ideal='gray',
                                x_min_limit=None, x_max_limit=None, y_min_limit=None, y_max_limit=None, xticks =1,yticks=1):

        plt.figure(figsize=(x_size, y_size))

        # Iterar sobre cada dataframe, color y etiqueta
        for i, df in enumerate(df_list):
            y_label = pd.to_numeric(df.iloc[:, 0])
            y_pred = pd.to_numeric(df.iloc[:, 1])
            label = labels[i]
            color = colors.get(label, 'black')  # Color predeterminado si no se encuentra en el diccionario
            
            # Graficar los puntos
            plt.scatter(y_label, y_pred, color=color, label=label, alpha=alpha)
            
            # Ajustar la recta de regresión
            slope, intercept = np.polyfit(y_label, y_pred, 1)
            regression_line = slope * y_label + intercept
            plt.plot(y_label, regression_line, color=color, linestyle='--')
            print(f"{label} - Pendiente: {slope}, Intercepto: {intercept}")

        if line_ideal:
            plt.plot([0,100], [0,100], color=color_line_ideal, linestyle='-', linewidth=2, label='Ideal Line')
        
        # Añadir detalles al gráfico
        plt.xlabel(xlabel, fontweight=weight,fontname=font, fontsize=fontsize)
        plt.ylabel(ylabel, fontweight=weight,fontname=font, fontsize=fontsize)
        plt.title(title, fontweight=weight, fontname=font, fontsize=fontsize+2)
        if legend:
                plt.legend()
        
        
        

        if mode == 1 :
            # Ajustar el grosor y color del borde interior del cuadro
            for spine in plt.gca().spines.values():
                spine.set_linewidth(1)
                spine.set_color('black')
        if mode ==  2:
            # Ajustar el grosor y color del borde interior del cuadro
            ax = plt.gca()
            ax.spines['top'].set_visible(False)  # Ocultar borde superior
            ax.spines['right'].set_visible(False)  # Ocultar borde derecho
            ax.spines['left'].set_linewidth(1)
            ax.spines['left'].set_color('black')
            ax.spines['bottom'].set_linewidth(1)
            ax.spines['bottom'].set_color('black')

        if mode ==  3:
            for spine in plt.gca().spines.values():
                spine.set_linewidth(0.2)
                spine.set_color('black')
            


        #Configurar números de los ejes en negrita
        #plt.xticks(fontweight='bold')
        #plt.yticks(fontweight='bold')

        # Definir límites de los ejes
        x_min = x_min_limit if x_min_limit is not None else int(min(y_label) // x_ticks_step * x_ticks_step)
        x_max = x_max_limit if x_max_limit is not None else int(max(y_label) // x_ticks_step * x_ticks_step + x_ticks_step)
        y_min = y_min_limit if y_min_limit is not None else int(min(y_pred) // y_ticks_step * y_ticks_step)
        y_max = y_max_limit if y_max_limit is not None else int(max(y_pred) // y_ticks_step * y_ticks_step + y_ticks_step)

        plt.xticks(np.arange(x_min, x_max + x_ticks_step, x_ticks_step))
        plt.yticks(np.arange(y_min, y_max + y_ticks_step, y_ticks_step))

        # Establecer los límites de los ejes
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.grid(True)

        if xticks==2:
            plt.xticks(fontweight='bold')
        if xticks ==3:
            #plt.xticks(np.arange(x_min, x_max + x_ticks_step, x_ticks_step), [''] * len(np.arange(x_min, x_max + x_ticks_step, x_ticks_step)))
            plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)


        if yticks==2:
            plt.yticks(fontweight='bold')
        if yticks ==3:
            plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

        # Mostrar el gráfico
        plt.show()







######### Plots de Clasificacioon

    def graph_roc(self, metrics_, title='', treshold= 0.75, individual=True,avg=True, color_avg='#F24405'):
        """
        Genera un gráfico ROC con las métricas almacenadas en metrics_.
        """
        # Inicializar valores promedio
        metric_tpr_mean = np.zeros(100)
        auc_values = []
        acc_list, prec_list, f1_list, rec_list = [], [], [], []
        plt.figure(figsize=(4, 4))


        # Iterar sobre las métricas por fold
        for fold, auc_roc in enumerate(metrics_["AUC"]):
            if auc_roc < treshold:  # Umbral para ignorar curvas con AUC < 0.75
                continue

            # Interpolación de TPR y cálculo de AUC
            tpr = np.interp(np.linspace(0, 1, 100), metrics_["FPR"][fold], metrics_["TPR"][fold])
            metric_tpr_mean += tpr
            auc_values.append(auc_roc)

            if individual:

                # Graficar cada curva individual
                plt.plot(
                    np.linspace(0, 1, 100), tpr,
                    color='#747E7E', alpha=0.5, lw=0.7,
                    label=None if len(metrics_["AUC"]) > 1 else f'AUC = {auc_roc:.2f}'
                )

            # Agregar métricas de evaluación por fold
            acc_list.append(metrics_["Accuracy"][fold])
            prec_list.append(metrics_["Precision"][fold])
            f1_list.append(metrics_["F1 Score"][fold])
            rec_list.append(metrics_["Recall"][fold])

        if avg:
            # Graficar la curva promedio si hay suficientes curvas válidas
            if auc_values:
                metric_tpr_mean /= len(auc_values)
                plt.plot(
                    np.linspace(0, 1, 100), metric_tpr_mean,
                    color=color_avg, lw=2, alpha=0.8,
                    label=f'(AUC = {np.mean(auc_values):.2f} ± {np.std(auc_values):.2f})'
                )

        # Línea de referencia
        plt.plot([0, 1], [0, 1], 'k--', lw=1.5)

        # Configuración de los ejes
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)

        # Título con métricas promedio
        plt.title(
            f'ROC Curve {title}\n'
            f'Acc={np.mean(acc_list):.2f} Prec={np.mean(prec_list):.2f} '
            f'F1={np.mean(f1_list):.2f} Rec={np.mean(rec_list):.2f}',
            fontsize=14
        )

        # Leyenda y mostrar gráfico
        plt.legend(loc="lower right", fontsize=10)
        #plt.grid(False, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()





    def plot_f_scores(self, f_scores, f_score_std, classes, colors=None,x_size=4,y_size=4):
                
        if colors is None:
            # Si no se pasan colores, usar colores predeterminados (uno para cada barra)
            colors = ['lightblue'] * len(f_scores)
        
        # Ordenar las barras de mayor a menor
        sorted_indices = np.argsort(f_scores)[::-1]  # Ordenar de mayor a menor
        f_scores = np.array(f_scores)[sorted_indices]
        f_score_std = np.array(f_score_std)[sorted_indices]
        classes = np.array(classes)[sorted_indices]
        colors = np.array(colors)[sorted_indices] if len(colors) == len(f_scores) else ['lightblue'] * len(f_scores)
        
        # Configuración de la figura
        fig, ax = plt.subplots(figsize=(x_size, y_size))
        
        # Crear las barras horizontales con error, asignando un color por cada barra
        bars = ax.barh(classes, f_scores, xerr=f_score_std, color=colors, edgecolor='black', capsize=5)
        
        # Etiquetas y título
        ax.set_xlabel('F-score')
        ax.set_title('F-scores con Desviación Estándar')
        
        # Invertir el eje Y para que la barra con el valor más alto quede en la parte superior
        ax.invert_yaxis()
        
        # Mostrar el gráfico
        plt.show()





    def C_Matrix_(self, metrics_, title='', threshold=0.75, individual=True, avg=True, color_avg='#F24405', classes=[], colors=[]):
        temp_matrix = []

        # Recorre los 200 modelos
        for i in range(len(metrics_['AUC'])):
            if metrics_["AUC"][i] < threshold:  # Umbral para ignorar curvas con AUC < 0.75
                continue
            temp_matrix.append(metrics_["Confusion Matrix"][i])

        avg_conf_matrix = np.mean(np.array(temp_matrix), axis=0)
        classes_ = classes = ['Low', 'High']

        # Crear la figura
        plt.figure(figsize=(4, 3))
        sns.heatmap(avg_conf_matrix, annot=True, fmt='.2f', cmap='Greys', 
                    cbar=True, linecolor='black', linewidths=1,
                    xticklabels=classes_, yticklabels=classes,
                    cbar_kws={'label': ''})
        
        plt.xlabel('Prediction')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()






    def C_Matrix(self, metrics_, title='', threshold=0.75, individual=True, avg=True, color_avg='#F24405', classes=[], colors=[]):
        temp_matrix = []

        # Recorre los 200 modelos
        for i in range(len(metrics_['AUC'])):
            if metrics_["AUC"][i] < threshold:  # Umbral para ignorar curvas con AUC < 0.75
                continue
            temp_matrix.append(metrics_["Confusion Matrix"][i])

        avg_conf_matrix = np.mean(np.array(temp_matrix), axis=0)
        std_conf_matrix = np.std(temp_matrix, axis=0)

        # Cálculo de porcentajes sobre la matriz promedio
        total = np.sum(avg_conf_matrix)
        if total > 0:
            percent_conf_matrix = (avg_conf_matrix / total) * 100
        else:
            percent_conf_matrix = np.zeros_like(avg_conf_matrix)

        classes_ = classes = ['Low', 'High']

        # Crear la figura
        plt.figure(figsize=(4, 3))

        # Crear un arreglo vacío para los textos dentro de cada celda
        text_matrix = np.empty(avg_conf_matrix.shape, dtype=object)

        # Llenar el arreglo de texto con los valores correspondientes
        for i in range(len(avg_conf_matrix)):
            for j in range(len(avg_conf_matrix[i])):
                text_matrix[i, j] = f"{avg_conf_matrix[i, j]:.2f}\n({percent_conf_matrix[i, j]:.1f}%)\n({std_conf_matrix[i, j]:.2f})"

        # Graficar el heatmap con los valores combinados
        # Graficar el heatmap con los valores basados en porcentaje
        sns.heatmap(percent_conf_matrix, annot=text_matrix, fmt='', cmap='Greys', 
                    cbar=True, linecolor='black', linewidths=1,
                    xticklabels=classes_, yticklabels=classes,
                    cbar_kws={'label': '%'})
        plt.xlabel('Prediction')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()




    def plot_metrics_clf(self,metrics_df):
        plt.figure(figsize=(5, 3))
        sns.boxplot(data=metrics_df, orient="h", palette="Set2")
        sns.stripplot(data=metrics_df, orient="h", color="black", alpha=0.5, jitter=True)

        # Añadir detalles al gráfico
        plt.title("Resultados de Métricas por Fold", fontsize=16)
        plt.xlabel("Valor de la Métrica", fontsize=12)
        plt.ylabel("Métrica", fontsize=12)
        plt.grid(axis="x", linestyle="--", alpha=0.7)
        plt.tight_layout()

        # Mostrar el gráfico
        plt.show()

