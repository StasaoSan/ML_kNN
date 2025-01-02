import numpy as np
from knn import KNN


def evaluate_knn(X_train, y_train, X_test, y_test, n_neighbors=5, radius=None, metric='euclidean',
                kernel='uniform', weights=None, p=2):
    knn = KNN(n_neighbors=n_neighbors, radius=radius, metric=metric, kernel=kernel, weights=weights, p=p)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    return accuracy


def grid_search_knn(X_train, y_train, X_val, y_val, neighbor_range, radius_range, metrics, kernels, p_range):
    best_accuracy = 0
    best_params = {}
    results = []
    print("Точность для разных комбинаций параметров:")

    for n_neighbors in neighbor_range:
        for metric in metrics:
            if metric == 'minkowski':
                for p in p_range:
                    for kernel_name in kernels.keys():
                        accuracy = evaluate_knn(
                            X_train, y_train, X_val, y_val,
                            n_neighbors=n_neighbors, radius=None,
                            metric=metric, kernel=kernel_name, p=p
                        )
                        print(
                            f"[kNN] Метрика: {metric} (p={p}), Ядро: {kernel_name}, n_neighbors: {n_neighbors}, Точность: {accuracy:.4f}")
                        results.append({
                            'type': 'n_neighbors',
                            'n_neighbors': n_neighbors,
                            'metric': f"{metric}_p{p}",
                            'kernel': kernel_name,
                            'accuracy': accuracy
                        })
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_params = {
                                'type': 'n_neighbors',
                                'n_neighbors': n_neighbors,
                                'metric': f"{metric}_p{p}",
                                'kernel': kernel_name
                            }
            else:
                for kernel_name in kernels.keys():
                    accuracy = evaluate_knn(
                        X_train, y_train, X_val, y_val,
                        n_neighbors=n_neighbors, radius=None,
                        metric=metric, kernel=kernel_name
                    )
                    print(
                        f"[kNN] Метрика: {metric}, Ядро: {kernel_name}, n_neighbors: {n_neighbors}, Точность: {accuracy:.4f}")
                    results.append({
                        'type': 'n_neighbors',
                        'n_neighbors': n_neighbors,
                        'metric': metric,
                        'kernel': kernel_name,
                        'accuracy': accuracy
                    })
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = {
                            'type': 'n_neighbors',
                            'n_neighbors': n_neighbors,
                            'metric': metric,
                            'kernel': kernel_name
                        }

    for radius in radius_range:
        for metric in metrics:
            if metric == 'minkowski':
                for p in p_range:
                    for kernel_name in kernels.keys():
                        accuracy = evaluate_knn(
                            X_train, y_train, X_val, y_val,
                            n_neighbors=None, radius=radius,
                            metric=metric, kernel=kernel_name, p=p
                        )
                        print(
                            f"[RadiusNN] Метрика: {metric} (p={p}), Ядро: {kernel_name}, radius: {radius}, Точность: {accuracy:.4f}")
                        results.append({
                            'type': 'radius',
                            'radius': radius,
                            'metric': f"{metric}_p{p}",
                            'kernel': kernel_name,
                            'accuracy': accuracy
                        })
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_params = {
                                'type': 'radius',
                                'radius': radius,
                                'metric': f"{metric}_p{p}",
                                'kernel': kernel_name
                            }
            else:
                for kernel_name in kernels.keys():
                    accuracy = evaluate_knn(
                        X_train, y_train, X_val, y_val,
                        n_neighbors=None, radius=radius,
                        metric=metric, kernel=kernel_name
                    )
                    print(
                        f"[RadiusNN] Метрика: {metric}, Ядро: {kernel_name}, radius: {radius}, Точность: {accuracy:.4f}")
                    results.append({
                        'type': 'radius',
                        'radius': radius,
                        'metric': metric,
                        'kernel': kernel_name,
                        'accuracy': accuracy
                    })
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = {
                            'type': 'radius',
                            'radius': radius,
                            'metric': metric,
                            'kernel': kernel_name
                        }
    return best_params, best_accuracy, results
