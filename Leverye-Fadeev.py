import numpy as np

def compute_eigenvalues_and_vectors(A):
    signs = np.array([1, -1, -1, 1, -1])

    print("Исходная матрица: ") 
    print(A, "\n")

    n = A.shape[0]
    E = np.eye(n)
    B = A.copy()
    p = np.trace(B)
    B -= p * E

    b = np.zeros((0, n))
    b = np.vstack((b, B[:1]))

    coefs = np.array([1, p])

    np.set_printoptions(suppress=True, precision=4)  

    print("B_1", "\n", B, "\n")
    print("p_1", np.round(p, 4), "\n")  

    for i in range(1, n):
        A_next = A @ B
        
        p = (1 / (i + 1)) * np.trace(A_next)  
        coefs = np.append(coefs, p)  
        
        if i != n - 1:  
            B = A_next - p * E
            b = np.vstack((b, B[:1]))  
            print(f"B_{i + 1}:")
            print(B, "\n")
        print(f"p_{i + 1}: {np.round(p, 4)}\n") 

    p_final = A_next[0][0]

    # Обратная A
    A_inv = B / p_final
    A_inv = np.round(A_inv, 4) 

    print("Обратная матрица A_inv:")
    print(A_inv, "\n")

    identity_check = A @ A_inv
    print("Проверка A @ A_inv:")
    print(np.round(identity_check, 1))

    print("\n", "Коэффициенты: ", coefs, "\n")

    roots_to_find = np.abs(coefs) * signs

    eigenvalues = np.roots(roots_to_find)

    print("Собственные значения матрицы A:", eigenvalues, "\n")

    eigenvectors_array = []

    for i in range(len(eigenvalues)):
        y0 = np.array([1, 0, 0, 0])
        
        eigenvalue = eigenvalues[i]
        
        y1 = eigenvalue * y0 + b[0]
        y2 = eigenvalue * y1 + b[1]
        x = eigenvalue * y2 + b[2]
        
        eigenvectors_array.append(x)
        
    eigenvectors = np.array(eigenvectors_array)

    print("Cобственные вектора:", "\n", eigenvectors, "\n")

    
    normalized_eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=1, keepdims=True)

    print("Нормализованные собственные вектора:", "\n", normalized_eigenvectors)
    
    return eigenvalues, normalized_eigenvectors

A = np.array([[2.2, 1, 0.5, 2],
              [1, 1.3, 2, 1],
              [0.5, 2, 0.5, 1.6],
              [2, 1, 1.6, 2]])

compute_eigenvalues_and_vectors(A)

print("\n\n")

#Добавление шума
noise = np.random.normal(0, 0.1, A.shape)  
A_noisy = A + noise

compute_eigenvalues_and_vectors(A_noisy)