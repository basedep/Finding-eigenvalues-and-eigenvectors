import numpy as np

def compute_eigenvalues_and_vectors(A):
    signs = np.array([1, -1, -1, 1, -1])
    np.set_printoptions(suppress=True, precision=4)

    print("Исходная матрица: \n", A, "\n")

    def compute_B_and_D(A, step):
        n = A.shape[0]
        B = np.eye(n)

        # Заполнение матрицы B
        for j in range(n):
            B[n - step - 1, j] = -A[n - step, j] / A[n - step, n - step - 1]
        B[n - step - 1, n - step - 1] = 1 / A[n - step, n - step - 1]

        B_inv = np.linalg.inv(B)
        D = B_inv @ A @ B

        return B, B_inv, D

    D = A.copy() #Для Фробениуса
    n = A.shape[0]
    B_total = [] #подобия

    for step in range(1, n):
        B, B_inv, D = compute_B_and_D(D, step)
        if step == 1:
            B_total = B 
        else:
            B_total = B_total @ B

    print(f"Матрица Фробениуса: \n", D, "\n")
    print(f"Матрица подобия B: \n", B_total, "\n")

    revert = np.insert(D[:1], 0, 1)
    roots_to_find = np.abs(revert) * signs

    eigenvalues = np.roots(roots_to_find)
    print("Собственные значения: ", eigenvalues, "\n")

    x = []

    for eigenvalue in eigenvalues:
        current_y = []
        
        for power in range(n-1, -1, -1):
            current_y.append(round(eigenvalue**power, 4))

        x_result = B_total @ current_y 
        x.append(x_result)

    x = np.array(x)

    print("Собственные вектора: \n", x, "\n")

    normalized_eigenvectors = x / np.linalg.norm(x, axis=1, keepdims=True)

    print("Нормализованные собственные вектора: \n", normalized_eigenvectors, "\n")
    
    return eigenvalues, normalized_eigenvectors


A = np.array([[2.2, 1, 0.5, 2],
              [1, 1.3, 2, 1],
              [0.5, 2, 0.5, 1.6],
              [2, 1, 1.6, 2]])

compute_eigenvalues_and_vectors(A)

print("\n\n")

noise = np.random.normal(0, 0.1, A.shape)  
A_noisy = A + noise

compute_eigenvalues_and_vectors(A_noisy)
