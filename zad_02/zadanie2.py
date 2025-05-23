import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

def gauss_elimination_partial_pivoting(A, b):
    """
    Rozwiązuje układ równań Ax = b metodą eliminacji Gaussa z częściowym wyborem elementu podstawowego
    dla macierzy rzadkich.
    """
    n = A.shape[0]
    A = sp.lil_matrix(A)  # Konwersja na macierz rzadką LIL
    b = b.astype(float)

    for k in range(n):
        max_row = np.argmax(np.abs(A[k:, k].toarray().flatten())) + k
        A[[k, max_row], :] = A[[max_row, k], :].copy()
        b[[k, max_row]] = b[[max_row, k]]

        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:].toarray()
            b[i] -= factor * b[k]

    A = A.tocsr()  # Konwersja do formatu CSR
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = ((b[i] - A[i, i+1:].dot(x[i+1:])) / A[i, i]).item()
        
    return x

def test_gauss_elimination():
    """
    Tworzy przykładowy układ równań Ax = b, wypisuje go w czytelnej formie
    i rozwiązuje metodą eliminacji Gaussa.
    """
    A = sp.lil_matrix([
        [2, -1,  1,  3],
        [1,  3,  2, -2],
        [3,  1, -3,  1],
        [2, -2,  4, -1]
    ], dtype=float)

    b = np.array([5, 3, -1, 4], dtype=float)

    print("\nPrzykładowy układ równań do rozwiązania:")
    for i in range(A.shape[0]):
        row = " + ".join(f"{A[i, j]:.1f} * x{j+1}" for j in range(A.shape[1]))
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


def test_wave_system(N, L, H, h, T):
    """
    Wywołuje funkcję build_and_solve_wave_system, po czym wypisuje wartości funkcji
    φ dla poszczególnych argumentów z.
    """

    z_vals, phi_solution = build_and_solve_wave_system(N, L, H, h, T)

    # Wypisanie rozwiązania
    print("\nRozwiązanie φ(z) dla różnych głębokości:")
    for z, phi in zip(z_vals, phi_solution):
        print(f"z = {z:.2f}, φ = {phi:.4f}")
        
        
def solve_using_scipy(A_sparse, b):
    """
    Rozwiązuje układ równań Ax = b za pomocą gotowej funkcji spsolve z biblioteki scipy.
    """
    return spla.spsolve(A_sparse.tocsr(), b)


def compare_solutions(N, L, H, h, T):
    z_vals, phi_custom = build_and_solve_wave_system(N, L, H, h, T)

    dz = h / (N - 1)
    A = np.zeros((N, N))
    b = np.zeros(N)

    for i in range(1, N - 1):
        A[i, i - 1] = 1 / dz**2
        A[i, i] = -2 / dz**2
        A[i, i + 1] = 1 / dz**2

    A[0, 0] = 1
    b[0] = 0  
    g = 9.81  
    k = 2 * np.pi / L  
    omega = 2 * np.pi / T  
    A[-1, -1] = 1
    b[-1] = g * H / (2 * omega) * np.cosh(k * h) / np.cosh(k * h)

    A_sparse = sp.csr_matrix(A)
    phi_scipy = solve_using_scipy(A_sparse, b)

    # Obliczenie błędu
    error = np.abs(phi_custom - phi_scipy)

    # Wizualizacja błędu
    plt.figure(figsize=(10, 6))
    plt.plot(z_vals, error, 'g-', linewidth=2, label='Różnica między metodami')
    plt.xlabel('Głębokość z')
    plt.ylabel('|φ_Gauss - φ_Scipy|')
    plt.legend()
    plt.title('Błąd względny między metodami')
    plt.grid()
    plt.savefig('wykres_roznic_metod')

    print("\nMaksymalny błąd względny między metodami:", np.max(error))

# Testowanie
N = 50
L = 10
H = 1
h = 5
T = 8
test_gauss_elimination()
test_wave_system(N, L, H, h, T)
compare_solutions(N, L, H, h, T)