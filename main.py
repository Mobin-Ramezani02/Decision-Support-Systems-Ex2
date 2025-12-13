import numpy as np
import pandas as pd

def SSE(R_true, R_pred_round):
    mask = (R_true != 0)
    diff = R_true[mask] - R_pred_round[mask]
    return np.sum(diff ** 2)


def ALS(R, k, lambda_reg, n_iters):
    np.random.seed(42)

    m, n = R.shape

    U = np.random.rand(m, k) * 5.0
    V = np.random.rand(n, k) * 5.0

    observed_mask = (R != 0)

    for it in range(n_iters):

        # Update Users
        for i in range(m):
            idx_items = np.where(observed_mask[i, :])[0]
            if idx_items.size == 0:
                continue

            V_i = V[idx_items, :]
            r_i = R[i, idx_items]

            A = V_i.T @ V_i + lambda_reg * np.eye(k)
            b = V_i.T @ r_i

            U[i, :] = np.linalg.solve(A, b)

        # Update Items
        for j in range(n):
            idx_users = np.where(observed_mask[:, j])[0]
            if idx_users.size == 0:
                continue

            U_j = U[idx_users, :]
            r_j = R[idx_users, j]

            A = U_j.T @ U_j + lambda_reg * np.eye(k)
            b = U_j.T @ r_j

            V[j, :] = np.linalg.solve(A, b)

        R_hat = U @ V.T
        R_hat_clipped = np.clip(R_hat, 1, 5)
        R_hat_round = np.floor(R_hat_clipped + 0.5)
        sse = SSE(R, R_hat_round)
        print(f"Iteration {it+1}/{n_iters} => SSE : {sse:.2f}")

    return U, V


df = pd.read_csv("dataset/matrix_1000x200_sparse40.csv")
R = df.values.astype(float)
print(f"Shape of R: {R.shape[0]}x{R.shape[1]}")

K = 70     # تعداد فاکتورهای نهان
LAMBDA = 0.1  # ضریب منظم سازی (λ)
N_ITERS = 20  # تعداد تکرار ALS

U, V = ALS(R, k=K, lambda_reg=LAMBDA, n_iters=N_ITERS)

R_hat = U @ V.T

R_hat_clipped = np.clip(R_hat, 1, 5)

R_hat_round = np.floor(R_hat_clipped + 0.5)

sse = SSE(R, R_hat_round)
print("Final SSE:", sse)

df_out = pd.DataFrame(R_hat_round)
df_out.to_excel("output/R_completed_round.xlsx", index=False)
