import numpy as np
darwin = np.load("Darwin_X_train.npy")
linux = np.load("Linux_X_train.npy")
np.testing.assert_allclose(darwin, linux)
