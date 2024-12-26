import numpy as np

def compute_eigenvalues_and_vectors(A):
    signs = np.array([1, -1, -1, 1, -1])
    
    print("Исходная матрица: ") 
    print(A, "\n")

    np.set_printoptions(suppress=True, precision=4)

    n = A.shape[0] 

    y0 = np.array([1, 0, 0, 0])
    y_vectors = [y0]

    for i in range(n):  
        next_vector = np.dot(A, y_vectors[-1])  
        y_vectors.append(next_vector) 

    y1 = y_vectors[1] 
    y2 = y_vectors[2] 
    y3 = y_vectors[3] 
    y4 = y_vectors[4] 

    # Система
    coefficients = np.array([
        [y3[0], y2[0], y1[0], 1],
        [y3[1], y2[1], y1[1], 0],
        [y3[2], y2[2], y1[2], 0],
        [y3[3], y2[3], y1[3], 0] 
    ])

    free_terms = np.array([y4[0], y4[1], y4[2], y4[3]])

    p_solutions = np.linalg.solve(coefficients, free_terms)

    print("Решения системы уравнений p1, p2, p3, p4: ", p_solutions, "\n")

    revert = np.insert(p_solutions, 0, 1)
    roots_to_find = np.abs(revert) * signs

    roots = np.roots(roots_to_find)

    print("Собственные значения матрицы A:", roots, "\n")

    eigenvectors_array = []

    for i in range(len(roots)):
        eigenvalue = roots[i]
        
        q1 = 1
        q2 = eigenvalue * q1 - p_solutions[0]
        q3 = eigenvalue * q2 - p_solutions[1]
        q4 = eigenvalue * q3 - p_solutions[2]
        
        x = y3 + q2 * y2 + q3 * y1 + q4 * y0
        
        eigenvectors_array.append(x)

    eigenvectors = np.array(eigenvectors_array)

    print("Cобственные вектора:", "\n", eigenvectors, "\n")

    # Нормализация
    normalized_eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=1, keepdims=True)

    print("Нормализованные собственные вектора:", "\n", normalized_eigenvectors)
    
    return roots, normalized_eigenvectors

A = np.array([[2.2, 1, 0.5, 2],
              [1, 1.3, 2, 1],
              [0.5, 2, 0.5, 1.6],
              [2, 1, 1.6, 2]])

compute_eigenvalues_and_vectors(A)

print("\n\n")

noise = np.random.normal(0, 0.1, A.shape)  
A_noisy = A + noise

compute_eigenvalues_and_vectors(A_noisy)
