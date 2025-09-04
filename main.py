import numpy as np
import pandas as pd

np.random.seed(1011)

x1 = np.random.uniform(-20, 20, size=3000)
x2 = np.random.uniform(-20, 20, size=3000)
x3 = np.random.uniform(-20, 20, size=3000)

df = pd.DataFrame({
    "x1": x1,
    "x2": x2,
    "x3": x3,
})

df["y1"] = (df["x3"] + df["x2"]) * df["x1"] / 10



x1 = np.random.uniform(-20, 20, size=100)
x2 = np.random.uniform(-20, 20, size=100)
x3 = np.random.uniform(-20, 20, size=100)

test_df = pd.DataFrame({
    "x1": x1,
    "x2": x2,
    "x3": x3,
})

test_df["y1"] = (test_df["x3"] + test_df["x2"]) * test_df["x1"] / 10


for col in ["x1","x2","x3"]:
    mean, std = df[col].mean(), df[col].std()
    df[col] = (df[col] - mean) / std
    mean, std = test_df[col].mean(), test_df[col].std()
    test_df[col] = (test_df[col] - mean) / std

inputDim = 3
midNeurons = 12
outputDim = 1

w1 = np.random.randn(midNeurons, inputDim) * np.sqrt(2.0 / inputDim)
b1 = np.zeros(midNeurons)

w2 = np.random.randn(outputDim, midNeurons) * np.sqrt(2.0 / midNeurons)
b2 = np.zeros(outputDim)

def predictor(df: pd.DataFrame, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray):
    X = df.iloc[:, :inputDim].to_numpy().T
    Z = w1 @ X + b1[:, None]
    A = np.maximum(Z, 0)
    Y_hat = w2 @ A + b2[:, None]
    df["y_hat"] = np.squeeze(Y_hat)
    return df


def lose(df: pd.DataFrame, w1: np.ndarray,b1: np.ndarray, w2: np.ndarray, b2: np.ndarray):
    predictor(df, w1, b1, w2, b2)
    diff = df["y_hat"] - df["y1"]
    return np.linalg.norm(diff)**2 / (2 * len(df))

def mse(df: pd.DataFrame, w1: np.ndarray,b1: np.ndarray, w2: np.ndarray, b2: np.ndarray):
    predictor(df, w1, b1, w2, b2)
    diff = df["y_hat"] - df["y1"]
    return np.mean((diff) ** 2)

def Gradient(df: pd.DataFrame, w1: np.ndarray,b1: np.ndarray, w2: np.ndarray, b2: np.ndarray):
    X = df.iloc[:, :inputDim].to_numpy().T
    Z = w1 @ X + b1[:, None]
    A = np.maximum(Z, 0)
    Y_hat = w2 @ A + b2[:, None]
    R = (np.squeeze(Y_hat) - df["y1"].to_numpy())[None, :]
    dw2 = (R @ A.T) / (len(df))
    db2 = (R @ np.ones((len(df),1))) / (len(df))
    delta = w2.T @ R * np.vectorize(lambda x: 1 if x > 0 else 0)(Z)
    dw1 = delta @ X.T / len(df)
    db1 = delta @ np.ones((len(df),1)) / len(df)
    return dw1, np.squeeze(db1), dw2, np.squeeze(db2)

def gradientDescent(df: pd.DataFrame, loss, gradient,
                    w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray,
                    learning_rate: float = 1e-3,
                    iterations: int = 10000,
                    threshold: float = 1e-5):
    last_w1, last_b1, last_w2, last_b2 = w1, b1, w2, b2
    last_loss = loss(df, w1, b1, w2, b2)
    for i in range(iterations):
        g_w1, g_b1, g_w2, g_b2 = gradient(df, last_w1, last_b1, last_w2, last_b2)
        cur_w1 = last_w1 - learning_rate * g_w1
        cur_b1 = last_b1 - learning_rate * g_b1
        cur_w2 = last_w2 - learning_rate * g_w2
        cur_b2 = last_b2 - learning_rate * g_b2
        cur_loss = loss(df, cur_w1, cur_b1, cur_w2, cur_b2)
        if (i % 100) == 0:
            print(i, cur_loss)
        diff_loss = last_loss - cur_loss
        if (diff_loss < threshold) or (diff_loss < 0):
            break
        else:
            last_w1 = cur_w1
            last_b1 = cur_b1
            last_w2 = cur_w2
            last_b2 = cur_b2
            last_loss = cur_loss
    return cur_w1, cur_b1, cur_w2, cur_b2



w1, b1, w2, b2 = gradientDescent(df, lose, Gradient, w1, b1, w2, b2)

print("df: \n", predictor(df, w1, b1, w2, b2))
print("Training mse:", mse(df, w1, b1, w2, b2))

print("Test df: \n", predictor(test_df, w1, b1, w2, b2))
print("Test mse:" , mse(test_df, w1, b1, w2, b2))
print(w1, b1, w2, b2)
