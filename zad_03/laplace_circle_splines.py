import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def grid_circle(N):
    """
    Zwraca listę współrzędnych punktów wewnętrznych siatki w kole jednostkowym.
    """
    h = 2.0 / N
    nodes = []
    for i in range(1, N):
        x = -1 + i * h
        for j in range(1, N):
            y = -1 + j * h
            if x**2 + y**2 < 1.0 - 1e-12:  # punkt wewnętrzny okręgu
                nodes.append((x, y))
    return nodes

def boundary_value(x, y, func):
    """
    Zwraca wartość warunku brzegowego na okręgu dla danej funkcji analitycznej.
    """
    return func(x, y)

def build_system(N, boundary_func):
    """
    Buduje układ równań dla równania Laplace'a na kole jednostkowym.
    Z1: Budowa układu równań liniowych.
    """
    h = 2.0 / N
    nodes = grid_circle(N)
    node_idx = {p: k for k, p in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)))
    b = np.zeros(len(nodes))

    for k, (x, y) in enumerate(nodes):
        neighbors = []
        for dx, dy in [(-h, 0), (h, 0), (0, -h), (0, h)]:
            xn, yn = x + dx, y + dy
            if (xn, yn) in node_idx:
                neighbors.append(((xn, yn), 1.0))
            else:
                if xn**2 + yn**2 <= 1.0 + 1e-12:
                    bv = boundary_value(xn, yn, boundary_func)
                    b[k] -= bv
        A[k, k] = -4.0
        for (xn, yn), coeff in neighbors:
            A[k, node_idx[(xn, yn)]] = coeff
    A /= h**2
    b /= h**2
    return A, b, nodes

def gauss_elimination(A, b):
    """
    Prosty algorytm eliminacji Gaussa bez wyboru elementu głównego.
    Z2: Rozwiązanie układu równań metodą Gaussa.
    """
    A = A.copy()
    b = b.copy()
    n = len(b)
    for i in range(n):
        pivot = A[i, i]
        if abs(pivot) < 1e-15:
            raise ValueError("Pivot zero!")
        A[i] = A[i] / pivot
        b[i] = b[i] / pivot
        for j in range(i+1, n):
            factor = A[j, i]
            A[j] = A[j] - factor * A[i]
            b[j] = b[j] - factor * b[i]
    x = np.zeros(n)
    for i in reversed(range(n)):
        x[i] = b[i] - np.dot(A[i, i+1:], x[i+1:])
    return x

def gauss_seidel(A, b, x0=None, tol=1e-9, maxiter=20000, verbose=False):
    """
    Metoda Gaussa-Seidela dla układów równań liniowych.
    Z4: Rozwiązanie układu metodą iteracyjną Gaussa-Seidela.
    """
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()
    for it in range(maxiter):
        x_new = x.copy()
        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        if np.linalg.norm(x_new - x, np.inf) < tol:
            if verbose:
                print(f"Gauss-Seidel: zbieżność po {it+1} iteracjach.")
            return x_new
        x = x_new
    if verbose:
        print("Gauss-Seidel: osiągnięto max iteracji")
    return x

def get_section(nodes, values, axis='x'):
    """
    Zwraca punkty na przekroju x=0 lub y=0 oraz odpowiadające im wartości.
    Wybiera punkty najbliższe osi i usuwa duplikaty, zapewniając rosnącą sekwencję.
    """
    tol = 1e-2  # większa tolerancja
    section_pts = []
    section_vals = []
    if axis == 'x':
        min_abs_y = min(abs(y) for (x, y) in nodes)
        for (x, y), v in zip(nodes, values):
            if abs(y) <= min_abs_y + tol:
                section_pts.append(x)
                section_vals.append(v)
    elif axis == 'y':
        min_abs_x = min(abs(x) for (x, y) in nodes)
        for (x, y), v in zip(nodes, values):
            if abs(x) <= min_abs_x + tol:
                section_pts.append(y)
                section_vals.append(v)
    if not section_pts:
        raise ValueError("Brak punktów przekroju – sprawdź parametry siatki lub tolerancję!")
    section = sorted(zip(section_pts, section_vals))
    unique_pts = []
    unique_vals = []
    last_pt = None
    for pt, val in section:
        if last_pt is None or abs(pt - last_pt) > 1e-8:
            unique_pts.append(pt)
            unique_vals.append(val)
            last_pt = pt
    return np.array(unique_pts), np.array(unique_vals)

def solve_laplace(N, boundary_func, method='gauss', verbose=False):
    """
    Rozwiązuje układ Laplace'a i zwraca siatkę, rozwiązanie oraz splajny.
    Pozwala wybrać metodę rozwiązania ('gauss' lub 'seidel').
    """
    A, b, nodes = build_system(N, boundary_func)
    if method == 'gauss':
        u = gauss_elimination(A, b)
        if verbose:
            print("Rozwiązano układ metodą Gaussa")
    elif method == 'seidel':
        u = gauss_seidel(A, b, tol=1e-9, maxiter=30000, verbose=verbose)
        if verbose:
            print("Rozwiązano układ metodą Gaussa-Seidela")
    else:
        raise ValueError("method musi być 'gauss' albo 'seidel'")
    xs, ux = get_section(nodes, u, axis='x')
    ys, uy = get_section(nodes, u, axis='y')
    spline_x = CubicSpline(xs, ux)
    spline_y = CubicSpline(ys, uy)
    return nodes, u, spline_x, spline_y, xs, ux, ys, uy

def solve_splines(xs, ux, ys, uy, method='gauss', verbose=False):
    """
    Interpoluje splajny przekrojów metodą Gaussa lub Seidela.
    Z5: Rozwiązanie układu trójdiagonalnego splajnu metodą iteracyjną.
    """
    def cubic_spline_system(x, y):
        n = len(x)
        h = np.diff(x)
        alpha = np.zeros(n)
        for i in range(1, n-1):
            alpha[i] = (3/h[i]) * (y[i+1]-y[i]) - (3/h[i-1]) * (y[i]-y[i-1])
        A = np.zeros((n, n))
        for i in range(1, n-1):
            A[i, i-1] = h[i-1]
            A[i, i]   = 2*(h[i-1]+h[i])
            A[i, i+1] = h[i]
        A[0,0] = 1.0
        A[-1,-1] = 1.0
        return A, alpha

    A_x, alpha_x = cubic_spline_system(xs, ux)
    A_y, alpha_y = cubic_spline_system(ys, uy)
    if method == 'gauss':
        c_x = gauss_elimination(A_x, alpha_x)
        c_y = gauss_elimination(A_y, alpha_y)
        if verbose:
            print("Splajny przekrojów: układ rozwiązany metodą Gaussa")
    elif method == 'seidel':
        c_x = gauss_seidel(A_x, alpha_x, tol=1e-9, maxiter=10000, verbose=verbose)
        c_y = gauss_seidel(A_y, alpha_y, tol=1e-9, maxiter=10000, verbose=verbose)
        if verbose:
            print("Splajny przekrojów: układ rozwiązany metodą Gaussa-Seidela")
    else:
        raise ValueError("method musi być 'gauss' albo 'seidel'")
    spline_x = CubicSpline(xs, ux)
    spline_y = CubicSpline(ys, uy)
    return spline_x, spline_y, c_x, c_y

############## --- FUNKCJE DO SPRAWOZDANIA/WALIDACJI --- ##############

def test_accuracy(nodes, u, analytical_func):
    """
    Sprawdza dokładność rozwiązania: oblicza błędy względem funkcji analitycznej.
    Zwraca normę maksymalną i średniokwadratową.
    """
    u_true = np.array([analytical_func(x, y) for (x, y) in nodes])
    err = u - u_true
    max_err = np.max(np.abs(err))
    mse = np.mean(err**2)
    return max_err, mse

def plot_cross_section(xs, ux, spline_x, analytical_func=None, axis='x', title_extra="", filename=None):
    """
    Rysuje przekrój (numeryczne węzły, splajn, opcjonalnie funkcja analityczna)
    i zapisuje wykres do pliku, jeśli filename jest podane.
    """
    xx = np.linspace(-1, 1, 200)
    plt.figure(figsize=(8, 4))
    plt.plot(xs, ux, 'o', label="Węzły przekroju", color='k')
    plt.plot(xx, spline_x(xx), '-', label="Splajn 3 stopnia")
    if analytical_func is not None:
        if axis == 'x':
            plt.plot(xx, [analytical_func(x, 0) for x in xx], '--', label="F. analityczna")
        else:
            plt.plot(xx, [analytical_func(0, y) for y in xx], '--', label="F. analityczna")
    plt.title(f"Przekrój z({axis},0) {title_extra}")
    plt.legend()
    plt.xlabel('x' if axis=='x' else 'y')
    plt.ylabel('z')
    plt.grid(True)
    if filename:
        plt.savefig(filename, bbox_inches='tight')

def plot_error(nodes, u, analytical_func, filename=None):
    """
    Rysuje mapę błędu numerycznego względem funkcji analitycznej na punktach siatki
    i zapisuje wykres do pliku, jeśli filename jest podane.
    """
    u_true = np.array([analytical_func(x, y) for (x, y) in nodes])
    err = u - u_true
    x = np.array([p[0] for p in nodes])
    y = np.array([p[1] for p in nodes])
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(x, y, c=err, cmap='bwr', marker='o')
    plt.colorbar(sc, label="Błąd numeryczny")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Rozkład błędu numerycznego na siatce")
    plt.axis('equal')
    if filename:
        plt.savefig(filename, bbox_inches='tight')

def convergence_study(N_values, boundary_func, analytical_func, method='gauss', filename=None):
    """
    Bada zbieżność metody (zmienia N i bada błąd dla coraz gęstszych siatek).
    Zapisuje wykres do pliku, jeśli filename jest podane.
    """
    errors = []
    hs = []
    for N in N_values:
        nodes, u, _, _, _, _, _, _ = solve_laplace(N, boundary_func, method=method)
        h = 2.0 / N
        hs.append(h)
        max_err, mse = test_accuracy(nodes, u, analytical_func)
        errors.append(max_err)
        print(f"N={N:2d}  h={h:.4f}  max_err={max_err:.2e}  mse={mse:.2e}")
    plt.figure()
    plt.loglog(hs, errors, 'o-', label="Max error")
    plt.xlabel("h (krok siatki)")
    plt.ylabel("Błąd maksymalny")
    plt.title("Zbieżność metody (max error vs h)")
    plt.grid()
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')

############## --- PRZYKŁADOWE FUNKCJE SPEŁNIAJĄCE WARUNEK LAPLACE'A --- ##############

def f1(x, y):
    return x**2 - y**2

def f2(x, y):
    return np.exp(x) * np.sin(y)

def f3(x, y):
    return np.cos(x) * np.cosh(y)  # Laplace: Δz = 0

############## --- BLOK GŁÓWNY: TESTY, WALIDACJA, WYKRESY --- ##############

if __name__ == "__main__":
    N = 15
    # Wybierz funkcję brzegową i analityczną do testu
    boundary_func = f1
    analytical_func = f1

    # --- Z1-Z3: rozwiązanie metodą Gaussa ---
    print("Rozwiązanie układu metodą Gaussa dla Laplace'a...")
    nodes, u, spline_x, spline_y, xs, ux, ys, uy = solve_laplace(N, boundary_func, method='gauss', verbose=True)
    with open("xx_minus_yy_generated_gauss.txt", "w") as f:
        for (x, y), val in zip(nodes, u):
            f.write(f"{x:.8f}\t{y:.8f}\t{val:.8f}\n")

    # --- Z4: rozwiązanie metodą Gaussa-Seidela ---
    print("Rozwiązanie układu metodą Gaussa-Seidela dla Laplace'a...")
    nodes_s, u_s, spline_x_s, spline_y_s, xs_s, ux_s, ys_s, uy_s = solve_laplace(N, boundary_func, method='seidel', verbose=True)
    with open("xx_minus_yy_generated_seidel.txt", "w") as f:
        for (x, y), val in zip(nodes_s, u_s):
            f.write(f"{x:.8f}\t{y:.8f}\t{val:.8f}\n")

    # --- Z5: interpolacja splajnami przekrojów metodą Gaussa-Seidela ---
    print("Interpolacja splajnami przekrojów (układ trójdiagonalny) metodą Gaussa-Seidela...")
    spline_x2, spline_y2, c_x, c_y = solve_splines(xs, ux, ys, uy, method='seidel', verbose=True)

    # --- Wykresy do sprawozdania ---
    print("Rysowanie przykładowych przekrojów...")
    plot_cross_section(xs, ux, spline_x, analytical_func, axis='x', title_extra="(Gauss)", filename="wykres_przekroj_x_gauss.png")
    plot_cross_section(ys, uy, spline_y, analytical_func, axis='y', title_extra="(Gauss)", filename="wykres_przekroj_y_gauss.png")

    print("Rysowanie przykładowych przekrojów (Seidel dla splajnu)...")
    plot_cross_section(xs, ux, spline_x2, analytical_func, axis='x', title_extra="(Seidel dla splajnu)", filename="wykres_przekroj_x_seidel.png")

    print("Rysowanie rozkładu błędu...")
    plot_error(nodes, u, analytical_func, filename="wykres_blad_numeryczny.png")

    print("Test dokładności rozwiązania:")
    max_err, mse = test_accuracy(nodes, u, analytical_func)
    print(f"Błąd maksymalny: {max_err:.2e}, MSE: {mse:.2e}")

    print("Badanie zbieżności metody dla kilku siatek (może potrwać)...")
    convergence_study([8, 12, 16, 20], boundary_func, analytical_func, method='gauss', filename="wykres_zbieznosc.png")