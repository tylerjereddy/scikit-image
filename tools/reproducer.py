import platform
import numpy as np
from skimage.feature import local_binary_pattern
from tqdm import tqdm
from sklearn.datasets import fetch_openml


def main():
    dataset = fetch_openml("Fashion-MNIST")
    X, y = dataset.data, dataset.target
    y = y.astype(np.int8)
    images = X.to_numpy().reshape(-1, 28, 28)
    radius = 2
    n_points = 8 * radius
    n_bins = n_points + 2
    hog_features = []
    for image in tqdm(images):
        new_feats = []
        LBP = local_binary_pattern(image,
                                   P=n_points,
                                   R=radius,
                                   method="uniform")
        lbp_hist, _ = np.histogram(LBP.ravel(),
                                   bins=n_bins,
                                   range=(0, n_bins))
        new_feats.append(lbp_hist)
        hog_features.append(np.atleast_2d(new_feats))
    X = np.asarray(hog_features).reshape(X.shape[0], -1)
    X_train = X[:60_000]
    sys_name = platform.system()
    np.save(f"{sys_name}_X_train.npy", X_train)


if __name__ == "__main__":
    main()
