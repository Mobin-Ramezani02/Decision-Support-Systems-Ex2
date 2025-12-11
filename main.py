import numpy as np
import pandas as pd

def compute_sse(R_true, R_pred_round):
    mask = (R_true != 0)
    diff = R_true[mask] - R_pred_round[mask]
    return np.sum(diff ** 2)


def ALS(R, k, lambda_reg, n_iters):
    np.random.seed(42)

    m, n = R.shape

    # مقداردهی اولیه‌ی تصادفی
    U = np.random.rand(m, k) * 5.0
    V = np.random.rand(n, k) * 5.0

    # ماسک درایه‌های مشاهده‌شده (غیرصفر)
    observed_mask = (R != 0)

    for it in range(n_iters):
        # --------
        # آپدیت U (برای هر کاربر)
        # --------
        for i in range(m):
            idx_items = np.where(observed_mask[i, :])[0]
            if idx_items.size == 0:
                continue

            V_i = V[idx_items, :]      # |Ω_i| × k
            r_i = R[i, idx_items]      # |Ω_i|

            A = V_i.T @ V_i + lambda_reg * np.eye(k)
            b = V_i.T @ r_i

            U[i, :] = np.linalg.solve(A, b)

        # --------
        # آپدیت V (برای هر آیتم)
        # --------
        for j in range(n):
            idx_users = np.where(observed_mask[:, j])[0]
            if idx_users.size == 0:
                continue

            U_j = U[idx_users, :]      # |Ω^j| × k
            r_j = R[idx_users, j]      # |Ω^j|

            A = U_j.T @ U_j + lambda_reg * np.eye(k)
            b = U_j.T @ r_j

            V[j, :] = np.linalg.solve(A, b)

        # (اختیاری) نمایش SSE در هر تکرار
        R_hat = U @ V.T
        R_hat_clipped = np.clip(R_hat, 1, 5)
        R_hat_round = np.floor(R_hat_clipped + 0.5)
        sse = compute_sse(R, R_hat_round)
        print(f"Iteration {it+1}/{n_iters} => SSE : {sse:.2f}")

    return U, V


# ۱) خواندن دیتاست با pandas
df = pd.read_csv("matrix_1000x200_sparse40.csv")  # اگر جداکننده فرق داشت، sep رو عوض کن
R = df.values.astype(float)
print(f"Shape of R: {R.shape[0]}x{R.shape[1]}")

K = 130     # تعداد فاکتورهای نهان
LAMBDA = 0.1  # ضریب منظم‌سازی (λ)
N_ITERS = 1  # تعداد تکرار ALS 15

# ۲) اجرای ALS
U, V = ALS(R, k=K, lambda_reg=LAMBDA, n_iters=N_ITERS)

# ۳) ماتریس پیش‌بینی شده
R_hat = U @ V.T

# ۴) محدود کردن به بازه‌ی [1,5]
R_hat_clipped = np.clip(R_hat, 1, 5)

# ۵) گرد کردن مطابق صورت سوال
R_hat_round = np.floor(R_hat_clipped + 0.5)

# ۶) محاسبه SSE روی درایه‌های غیر صفر ماتریس اصلی
sse = compute_sse(R, R_hat_round)
print("Final SSE (on non-zero entries):", sse)

# ۷) ذخیره‌ی ماتریس کامل‌شده‌ی گرد شده با pandas
df_out = pd.DataFrame(R_hat_round)
df_out.to_csv("output/R_completed_round.csv", header=False)

