import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from evaluation import evaluate_knn


def plot_accuracy(train_accuracies, test_accuracies, parameter, param_values, title, xlabel, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, train_accuracies, marker='o', label='Точность на тренировочном множестве')
    plt.plot(param_values, test_accuracies, marker='s', label='Точность на тестовом множестве')
    plt.xlabel(xlabel)
    plt.ylabel('Точность')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_class_distribution(y, title='Распределение классов', save_path=None):
    unique, counts = np.unique(y, return_counts=True)
    data = pd.DataFrame({'Class': unique, 'Count': counts})
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Class', y='Count', data=data, hue='Class', palette='viridis', dodge=False)
    plt.xlabel('Класс')
    plt.ylabel('Количество объектов')
    plt.title(title)
    plt.xticks(unique)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_accuracy_over_parameters(results, parameter_type, save_path=None):
    df_results = pd.DataFrame(results)
    if parameter_type == 'n_neighbors':
        x_col = 'n_neighbors'
        xlabel = 'Число соседей k'
    else:
        x_col = 'radius'
        xlabel = 'Радиус окна'

    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df_results, x=x_col, y='accuracy', hue='metric', style='kernel', markers=True, dashes=False)
    plt.xlabel(xlabel)
    plt.ylabel('Точность')
    plt.title(f'Точность на разных метриках и ядрах в зависимости от {xlabel.lower()}')
    plt.legend(title='Метрика и Ядро', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_accuracy_results(X_train, y_train, X_test, y_test, best_params, results):
    if best_params['type'] == 'n_neighbors':
        neighbor_range_plot = range(1, 31, 2)
        train_accuracies, test_accuracies = [], []
        metric = best_params['metric']
        if 'minkowski' in metric:
            p = int(metric.split('_p')[-1])
            base_metric = 'minkowski'
        else:
            p = None
            base_metric = metric
        for k in neighbor_range_plot:
            acc_train = evaluate_knn(
                X_train=X_train, y_train=y_train, X_test=X_train, y_test=y_train,
                n_neighbors=k, radius=None, metric=base_metric, kernel=best_params['kernel'], p=p
            )
            acc_test = evaluate_knn(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                n_neighbors=k, radius=None, metric=base_metric, kernel=best_params['kernel'], p=p
            )
            train_accuracies.append(acc_train)
            test_accuracies.append(acc_test)
        plot_accuracy(train_accuracies, test_accuracies, parameter='k', param_values=neighbor_range_plot,
                      title='Зависимость точности от числа соседей', xlabel='Число соседей k')
    else:
        radius_range_plot = np.linspace(0.1, 2.0, 20)
        train_accuracies, test_accuracies = [], []
        metric = best_params['metric']
        if 'minkowski' in metric:
            p = int(metric.split('_p')[-1])
            base_metric = 'minkowski'
        else:
            p = None
            base_metric = metric
        for r in radius_range_plot:
            acc_train = evaluate_knn(
                X_train=X_train, y_train=y_train, X_test=X_train, y_test=y_train,
                n_neighbors=None, radius=r, metric=base_metric, kernel=best_params['kernel'], p=p
            )
            acc_test = evaluate_knn(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                n_neighbors=None, radius=r, metric=base_metric, kernel=best_params['kernel'], p=p
            )
            train_accuracies.append(acc_train)
            test_accuracies.append(acc_test)
        plot_accuracy(train_accuracies, test_accuracies, parameter='r', param_values=radius_range_plot,
                      title='Зависимость точности от радиуса окна', xlabel='Радиус окна')


def plot_weights_distribution(weights, title='Распределение весов после LOWESS', save_path=None):
    plt.figure(figsize=(8, 6))
    sns.histplot(weights, bins=30, kde=True, color='green')
    plt.xlabel('Вес')
    plt.ylabel('Количество объектов')
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_class_distribution_func(y_train):
    plot_class_distribution(
        y=y_train,
        title='Распределение классов в тренировочной выборке'
    )


def plot_accuracy_over_parameters_func(results):
    df_results_neighbors = [res for res in results if res['type'] == 'n_neighbors']
    plot_accuracy_over_parameters(results=df_results_neighbors, parameter_type='n_neighbors')

    df_results_radius = [res for res in results if res['type'] == 'radius']
    plot_accuracy_over_parameters(results=df_results_radius, parameter_type='radius')
