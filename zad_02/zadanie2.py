import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

def gauss_elimination_partial_pivoting(A, b):
    """
    Rozwiązuje układ równań Ax = b metodą eliminacji Gaussa z częściowym wyborem elementu podstawowego.
    """
    n = len(A)
    A = A.astype(float)  # Konwersja na float dla uniknięcia błędów numerycznych
    b = b.astype(float)

    for k in range(n):
        max_row = np.argmax(abs(A[k:, k])) + k
        A[[k, max_row]] = A[[max_row, k]]
        b[[k, max_row]] = b[[max_row, k]]
        
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]
    
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]

    return x


def test_gauss_elimination():
    """
    Tworzy przykładowy układ równań Ax = b, wypisuje go w czytelnej formie
    i rozwiązuje metodą eliminacji Gaussa.
    """
    A = np.array([
        [2, -1,  1,  3],
        [1,  3,  2, -2],
        [3,  1, -3,  1],
        [2, -2,  4, -1]
    ], dtype=float)

    b = np.array([5, 3, -1, 4], dtype=float)

    print("\nPrzykładowy układ równań do rozwiązania:")
    for i in range(len(A)):
        row = " + ".join(f"{A[i, j]:.1f} * x{j+1}" for j in range(len(A[i])))
        print(f"{row} = {b[i]:.1f}")

    x = gauss_elimination_partial_pivoting(A, b)

    print("\nRozwiązanie układu równań:")
    for i, val in enumerate(x):
        print(f"x{i+1} = {val:.4f}")


def build_and_solve_wave_system(N, L, H, h, T):
    """
    Tworzy i rozwiązuje układ równań dla funkcji phi(z) na podstawie wzoru (7).
    
    Parametry:
    N  - liczba podziałów w pionie
    L  - długość fali
    H  - wysokość fali
    h  - głębokość zbiornika
    T  - okres fali
    
    Zwraca:
    z_vals - wartości z na siatce
    phi_solution - rozwiązanie phi(z)
    """
    g = 9.81  # Przyspieszenie ziemskie
    k = 2 * np.pi / L  # Liczba falowa
    omega = 2 * np.pi / T  # Częstotliwość kołowa

    dz = h / (N - 1)
    z_vals = np.linspace(-h, 0, N)  # Zakres od dna (-h) do powierzchni (0)

    # Tworzenie macierzy rzadkiej (tylko elementy niezerowe)
    A = np.zeros((N, N))
    b = np.zeros(N)

    for i in range(1, N - 1):
        A[i, i - 1] = 1 / dz**2
        A[i, i] = -2 / dz**2
        A[i, i + 1] = 1 / dz**2

    # Warunki brzegowe:
    A[0, 0] = 1
    b[0] = 0  # Warunek przy dnie: ∂φ/∂z = 0
    
    A[-1, -1] = 1
    b[-1] = g * H / (2 * omega) * np.cosh(k * h) / np.cosh(k * h)  # Warunek na powierzchni

    # Rozwiązanie układu równań metodą eliminacji Gaussa
    phi_solution = gauss_elimination_partial_pivoting(A, b)

    return z_vals, phi_solution


def solve_using_scipy(A_sparse, b):
    """
    Rozwiązuje układ równań Ax = b za pomocą gotowej funkcji spsolve z biblioteki scipy.
    """
    return spla.spsolve(A_sparse, b)


# Uruchomienie porównania
N = 50
L = 10
H = 1
h = 5
T = 8

test_gauss_elimination()

z_vals, phi_solution = build_and_solve_wave_system(N, L, H, h, T)

# Wypisanie rozwiązania
print("\nRozwiązanie φ(z) dla różnych głębokości:")
for z, phi in zip(z_vals, phi_solution):
    print(f"z = {z:.2f}, φ = {phi:.4f}")