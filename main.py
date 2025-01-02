from data_utils import load_and_preprocess_data
from evaluation import grid_search_knn, evaluate_knn
from knn import uniform_kernel, gaussian_kernel, triangular_kernel, polynomial_kernel
from plot_utils import plot_accuracy_results, plot_class_distribution_func, plot_accuracy_over_parameters_func
from lowess_custom import calculate_lowess_weights


def load_data(csv_path, target_column):
    return load_and_preprocess_data(csv_path, target_column)


def set_parameters():
    neighbor_range = range(1, 21, 2)
    radius_range = [0.5, 1.0, 1.5, 2.0]
    metrics = ['euclidean', 'manhattan', 'chebyshev', 'cosine', 'minkowski']
    kernels = {
        'uniform': uniform_kernel,
        'gaussian': gaussian_kernel,
        'triangular': triangular_kernel,
        'polynomial': lambda d: polynomial_kernel(d, a=2, b=1)
    }
    p_range = [2, 3, 4]
    return neighbor_range, radius_range, metrics, kernels, p_range


def find_best_parameters(X_train, y_train, X_val, y_val, neighbor_range, radius_range, metrics, kernels, p_range):
    best_params, best_accuracy, results = grid_search_knn(
        X_train, y_train, X_val, y_val,
        neighbor_range, radius_range, metrics, kernels, p_range
    )
    return best_params, best_accuracy, results


def evaluate_test_set(X_train, y_train, X_test, y_test, best_params):
    metric = best_params['metric']
    if 'minkowski' in metric:
        p = int(metric.split('_p')[-1])
        base_metric = 'minkowski'
    else:
        p = None
        base_metric = metric

    test_accuracy = evaluate_knn(
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
    return test_accuracy


def main():
    csv_path = "allrecipes_binarized_every.csv"
    target_column = 'Category'

    X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, scaler = load_data(csv_path, target_column)

    print(f"Размер тренировочной выборки: {X_train.shape}")
    print(f"Размер валидационной выборки: {X_val.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")

    neighbor_range, radius_range, metrics, kernels, p_range = set_parameters()

    best_params, best_accuracy, results = find_best_parameters(
        X_train, y_train, X_val, y_val,
        neighbor_range, radius_range, metrics, kernels, p_range
    )

    print("\nЛучшие параметры:")
    print(best_params)
    print(f"Лучшая точность на валидационной выборке: {best_accuracy:.4f}")

    test_accuracy = evaluate_test_set(X_train, y_train, X_test, y_test, best_params)
    print(f"Точность на тестовом множестве с лучшими гиперпараметрами: {test_accuracy:.4f}")

    plot_accuracy_results(X_train, y_train, X_test, y_test, best_params, results)

    plot_class_distribution_func(y_train)

    plot_accuracy_over_parameters_func(results)

    calculate_lowess_weights(X_train, y_train, X_test, y_test, best_params)


if __name__ == "__main__":
    main()
