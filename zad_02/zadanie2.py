import numpy as np


# ZADANIE 1: Proszę zaimplementować metodę eliminacji Gaussa z częściowym wyborem elementu
# podstawowego (obowiązkowe).
def gauss_elimination_partial_pivoting(A, b):
    """
    Rozwiązuje układ równań Ax = b metodą eliminacji Gaussa z częściowym wyborem elementu podstawowego.
    """
    n = len(A)
    A = A.astype(float)  # Konwersja na float dla uniknięcia błędów numerycznych
    b = b.astype(float)
    
    # Eliminacja Gaussa z częściowym wyborem elementu podstawowego
    for k in range(n):
        # Znajdź wiersz z maksymalnym elementem w kolumnie k i zamień go z wierszem k
        max_row = np.argmax(abs(A[k:, k])) + k
        A[[k, max_row]] = A[[max_row, k]]
        b[[k, max_row]] = b[[max_row, k]]
        
        # Eliminacja
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]
    
    # Podstawianie wsteczne
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
    
    return x

# Przykładowe dane testowe
A = np.array([[2, -1, 1], [1, 3, 2], [1, -1, 2]], dtype=float)
b = np.array([8, 13, 7], dtype=float)

solution = gauss_elimination_partial_pivoting(A, b)
print("Rozwiązanie układu równań:", solution)


# ZADANIE 2: Proszę zbudować układ równań liniowych w zależności od zadanego parametru N
# opisujący funkcję φ(x, z, t) zgodnie z opisem powyżej i go rozwiązać (obowiązkowe).
def build_and_solve_wave_system(N, L, H, h, T, g=9.81):
    """
    Buduje i rozwiązuje układ równań dla funkcji φ(x, z, t) zgodnie z teorią falowania.
    N - liczba punktów siatki
    L - długość fali
    H - wysokość fali
    h - głębokość wody
    T - okres fali
    g - przyspieszenie ziemskie
    """
    k = 2 * np.pi / L  # Liczba falowa
    omega = 2 * np.pi / T  # Częstotliwość kołowa
    
    A = np.zeros((N, N))
    b = np.zeros(N)
    
    dz = h / (N - 1)  # Dzielimy głębokość na N-1 odcinków
    
    for i in range(1, N-1):
        A[i, i-1] = 1 / dz**2
        A[i, i] = -2 / dz**2
        A[i, i+1] = 1 / dz**2
    
    # Warunek brzegowy przy dnie (z = -h): dφ/dz = 0
    A[0, 0] = -1 / dz
    A[0, 1] = 1 / dz
    
    # Warunek brzegowy na powierzchni (z = 0): d²φ/dt² + g dφ/dz = 0
    A[-1, -2] = 1 / dz
    A[-1, -1] = -1 / dz - omega**2 / g
    
    # Rozwiązanie analityczne jako warunek brzegowy
    z_vals = np.linspace(-h, 0, N)
    phi_analytical = (g * H / (2 * omega)) * (np.cosh(k * (z_vals + h)) / np.cosh(k * h))
    b[-1] = phi_analytical[-1]
    
    # Rozwiązanie układu równań metodą eliminacji Gaussa
    phi_solution = gauss_elimination_partial_pivoting(A, b)
    
    return z_vals, phi_solution

# Przykładowe użycie
N = 10  # Liczba punktów siatki
L = 10  # Długość fali
H = 1   # Wysokość fali
h = 5   # Głębokość wody
T = 8   # Okres fali

z_vals, phi_solution = build_and_solve_wave_system(N, L, H, h, T)
print("Rozwiązanie φ(z) dla różnych głębokości:")
for z, phi in zip(z_vals, phi_solution):
    print(f"z = {z:.2f}, φ = {phi:.4f}")