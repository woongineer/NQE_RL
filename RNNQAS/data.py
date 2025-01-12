import tensorflow as tf
import torch
from pennylane import numpy as pnp
import numpy as np
from sklearn.decomposition import PCA


def data_catdog(reduction_sz=4):
    data_path = "/Users/jwheo/Desktop/Y/NQE/Neural-Quantum-Embedding/rl/kmnist"
    catdog_data_path = f"{data_path}/cat_dog_cifar10_pca.npz"
    catdog_data = np.load(catdog_data_path)

    x_train = catdog_data['X_train']
    y_train = catdog_data['y_train']
    x_test = catdog_data['X_test']
    y_test = catdog_data['y_test']

    y_train = y_train.flatten()
    y_test = y_test.flatten()

    # 1) Filter: cat(3), dog(5)만 사용
    train_filter = np.where((y_train == 3) | (y_train == 5))[0]
    test_filter = np.where((y_test == 3) | (y_test == 5))[0]

    x_train, y_train = x_train[train_filter], y_train[train_filter]
    x_test, y_test = x_test[test_filter], y_test[test_filter]

    # 2) 라벨을 0/1로 변환: cat -> 0, dog -> 1 (원하시는 대로)
    y_train = np.where(y_train == 3, 0, 1)
    y_test = np.where(y_test == 3, 0, 1)

    # 3) Grayscale 변환: [H, W, 3] -> [H, W], 평균값
    #   또는 RGB 3채널을 그대로 PCA에 넣어도 되지만, 여기서는 간단히 흑백으로.
    x_train_gray = np.mean(x_train, axis=-1)
    x_test_gray = np.mean(x_test, axis=-1)

    # 4) 리사이즈(예: 32×32 -> 256×1) 대신,
    #    우선 32×32 = 1024 픽셀을 flatten 하거나,
    #    혹은 tf.image.resize로 임의 크기로 바꿀 수도 있음.
    #    KMNIST 예제를 맞추려면 256×1으로 맞출 수도 있겠죠.
    x_train_gray = x_train_gray.reshape(-1, 32 * 32).astype(np.float32) / 255.0
    x_test_gray = x_test_gray.reshape(-1, 32 * 32).astype(np.float32) / 255.0

    # 5) PCA(reduction_sz)로 차원 축소
    #    여기선 reduction_sz=4면, shape: [n_samples, 4]
    pca = PCA(n_components=reduction_sz)
    x_train_pca = pca.fit_transform(x_train_gray)
    x_test_pca = pca.transform(x_test_gray)

    def scale_0_to_2pi(data):
        scaled = []
        for row in data:
            rmin, rmax = row.min(), row.max()
            scaled_row = (row - rmin) * (2 * np.pi / (rmax - rmin) + 1e-8)
            scaled.append(scaled_row)
        return np.array(scaled, dtype=np.float32)

    x_train_scaled = scale_0_to_2pi(x_train_pca)
    x_test_scaled = scale_0_to_2pi(x_test_pca)

    return x_train_scaled[:400], x_test_scaled[:100], y_train[:400], y_test[:100]


def data_load_and_process(dataset='mnist', reduction_sz: int = 4):
    data_path = "/Users/jwheo/Desktop/Y/NQE/Neural-Quantum-Embedding/rl/kmnist"
    if dataset == 'mnist':
        (x_train, y_train), (
            x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataset == 'kmnist':
        # Path to training images and corresponding labels provided as numpy arrays
        kmnist_train_images_path = f"{data_path}/kmnist-train-imgs.npz"
        kmnist_train_labels_path = f"{data_path}/kmnist-train-labels.npz"

        # Path to the test images and corresponding labels
        kmnist_test_images_path = f"{data_path}/kmnist-test-imgs.npz"
        kmnist_test_labels_path = f"{data_path}/kmnist-test-labels.npz"

        x_train = pnp.load(kmnist_train_images_path)['arr_0']
        y_train = pnp.load(kmnist_train_labels_path)['arr_0']

        # Load the test data from the corresponding npz files
        x_test = pnp.load(kmnist_test_images_path)['arr_0']
        y_test = pnp.load(kmnist_test_labels_path)['arr_0']

    x_train, x_test = x_train[..., pnp.newaxis] / 255.0, x_test[
        ..., pnp.newaxis] / 255.0
    train_filter_tf = pnp.where((y_train == 0) | (y_train == 1))
    test_filter_tf = pnp.where((y_test == 0) | (y_test == 1))

    x_train, y_train = x_train[train_filter_tf], y_train[train_filter_tf]
    x_test, y_test = x_test[test_filter_tf], y_test[test_filter_tf]

    x_train = tf.image.resize(x_train[:], (256, 1)).numpy()
    x_test = tf.image.resize(x_test[:], (256, 1)).numpy()
    x_train, x_test = tf.squeeze(x_train).numpy(), tf.squeeze(x_test).numpy()

    X_train = PCA(reduction_sz).fit_transform(x_train)
    X_test = PCA(reduction_sz).fit_transform(x_test)
    x_train, x_test = [], []
    for x in X_train:
        x = (x - x.min()) * (2 * pnp.pi / (x.max() - x.min()))
        x_train.append(x)
    for x in X_test:
        x = (x - x.min()) * (2 * pnp.pi / (x.max() - x.min()))
        x_test.append(x)
    return x_train[:400], x_test[:100], y_train[:400], y_test[:100]


def new_data(batch_sz, X, Y):
    X1_new, X2_new, Y_new = [], [], []
    for i in range(batch_sz):
        n, m = pnp.random.randint(len(X)), pnp.random.randint(len(X))
        X1_new.append(X[n])
        X2_new.append(X[m])
        Y_new.append(1 if Y[n] == Y[m] else 0)

    # X1_new 처리
    X1_new_array = pnp.array(X1_new)
    X1_new_tensor = torch.from_numpy(X1_new_array).float()

    # X2_new 처리
    X2_new_array = pnp.array(X2_new)
    X2_new_tensor = torch.from_numpy(X2_new_array).float()

    # Y_new 처리
    Y_new_array = pnp.array(Y_new)
    Y_new_tensor = torch.from_numpy(Y_new_array).float()
    return X1_new_tensor, X2_new_tensor, Y_new_tensor
