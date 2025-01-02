# lowess_custom.py
import numpy as np
from knn import KNN, KERNELS
from evaluation import evaluate_knn

def calculate_lowess_weights(X_train, y_train, X_test, y_test, best_params):
    # Извлекаем параметры метрики и ядра
    metric = best_params['metric']
    if 'minkowski' in metric:
        p = int(metric.split('_p')[-1])
        base_metric = 'minkowski'
    else:
        p = None
        base_metric = metric

    # Оцениваем начальную точность на тестовом множестве
    initial_accuracy = evaluate_knn(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_neighbors=best_params.get('n_neighbors'),
        radius=best_params.get('radius'),
        metric=base_metric,
        kernel=best_params['kernel'],
        p=p
    )
    print(f"Начальная точность на тестовом множестве: {initial_accuracy:.4f}")

    n_samples = X_train.shape[0]
    residuals = np.zeros(n_samples)
    weights = np.ones(n_samples)

    # Вычисляем остатки для каждого объекта обучающего множествац
    for i in range(n_samples):
        # Исключаем i-й объект из обучающего множества
        X_train_reduced = np.delete(X_train, i, axis=0)
        y_train_reduced = np.delete(y_train, i, axis=0)

        # Обучаем kNN на уменьшенном обучающем множестве
        knn = KNN(
            n_neighbors=best_params.get('n_neighbors'),
            radius=best_params.get('radius'),
            metric=base_metric,
            kernel=best_params['kernel'],
            p=p
        )
        knn.fit(X_train_reduced, y_train_reduced)

        # Предсказываем метку для исключенного объекта
        x_i = X_train[i].reshape(1, -1)
        y_hat_i = knn.predict(x_i)[0]

        # Вычисляем остаток
        residuals[i] = y_train[i] - y_hat_i

    # Выбираем функцию ядра для взвешивания остатков
    K = KERNELS.get(best_params['kernel'], 'uniform')

    # Используем абсолютные значения остатков
    residuals_abs = np.abs(residuals)

    # Вычисляем веса с помощью функции ядра
    weights = K(residuals_abs)

    # Обучаем kNN с новыми весами
    knn_weighted = KNN(
        n_neighbors=best_params.get('n_neighbors'),
        radius=best_params.get('radius'),
        metric=base_metric,
        kernel=best_params['kernel'],
        weights=weights,
        p=p
    )
    knn_weighted.fit(X_train, y_train)

    # Оцениваем точность на тестовом множестве
    y_pred = knn_weighted.predict(X_test)
    weighted_accuracy = np.mean(y_pred == y_test)
    print(f"Точность на тестовом множестве после взвешивания LOWESS: {weighted_accuracy:.4f}")
