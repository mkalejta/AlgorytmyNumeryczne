from math import cos, sin, sqrt, pi
import matplotlib.pyplot as plt
from heapq import heappop, heappush, heapify
import time

# H2
def suma_wektorow(n):
    theta = 2 * pi / n
    w0 = [cos(theta) - 1, sin(theta)]
    macierz_obrotu_wektora = [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]
    suma = w0
    poprzedni_wektor = w0
    for _ in range(1, n):
        nastepny_wektor = [macierz_obrotu_wektora[0][0]*poprzedni_wektor[0] + macierz_obrotu_wektora[0][1]*poprzedni_wektor[1],
                  macierz_obrotu_wektora[1][0]*poprzedni_wektor[0] + macierz_obrotu_wektora[1][1]*poprzedni_wektor[1]]
        suma = [suma[0] + nastepny_wektor[0], suma[1] + nastepny_wektor[1]]
        poprzedni_wektor = nastepny_wektor
    return suma

# H3
def sumy_wspolrzednych_wektorow(n):
    start_time = time.time()

    theta = 2 * pi / n
    w0 = [cos(theta) - 1, sin(theta)]
    macierz_obrotu_wektora = [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]
    
    x_plus, x_minus, y_plus, y_minus = [], [], [], []
    poprzedni_wektor = w0

    for _ in range(1, n):
        nastepny_wektor = [macierz_obrotu_wektora[0][0]*poprzedni_wektor[0] + macierz_obrotu_wektora[0][1]*poprzedni_wektor[1],
                  macierz_obrotu_wektora[1][0]*poprzedni_wektor[0] + macierz_obrotu_wektora[1][1]*poprzedni_wektor[1]]
        if nastepny_wektor[0] >= 0:
            x_plus.append(nastepny_wektor[0])
        else:
            x_minus.append(nastepny_wektor[0])
        if nastepny_wektor[1] >= 0:
            y_plus.append(nastepny_wektor[1])
        else:
            y_minus.append(nastepny_wektor[1])
        poprzedni_wektor = nastepny_wektor
    
    suma_x_plus = sum(sorted(x_plus))
    suma_x_minus = sum(sorted(x_minus, reverse=True))

    suma_y_plus = sum(sorted(y_plus))
    suma_y_minus = sum(sorted(y_minus, reverse=True))

    suma_x = suma_x_plus + suma_x_minus
    suma_y = suma_y_plus + suma_y_minus

    end_time = time.time()
    execution_time = end_time - start_time
    
    # print(f"Czas wykonania dla n={n}: {execution_time:.6f} sekundy")

    # print(f"Wynik: ({suma_x:.20f}, {suma_y:.20f})")
    return suma_x, suma_y, execution_time

# H4
def reduce_heap(heap, reverse=False):
    if reverse:
        heap = [-x for x in heap]  # Odwracamy znaki, by działać na max-heap
        heapify(heap)
    
    while len(heap) > 1:
        a = heappop(heap)
        b = heappop(heap)
        heappush(heap, a + b)
    
    return -heap[0] if reverse else heap[0]

def sumy_wspolrzednych_wektorow_heap(n):
    start_time = time.time()

    theta = 2 * pi / n
    w0 = [cos(theta) - 1, sin(theta)]
    macierz_obrotu_wektora = [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]
    
    x_plus, x_minus, y_plus, y_minus = [], [], [], []
    poprzedni_wektor = w0
    
    for _ in range(1, n):
        nastepny_wektor = [
            macierz_obrotu_wektora[0][0] * poprzedni_wektor[0] + macierz_obrotu_wektora[0][1] * poprzedni_wektor[1],
            macierz_obrotu_wektora[1][0] * poprzedni_wektor[0] + macierz_obrotu_wektora[1][1] * poprzedni_wektor[1]
        ]
        
        if nastepny_wektor[0] >= 0:
            x_plus.append(nastepny_wektor[0])
        else:
            x_minus.append(nastepny_wektor[0])
        
        if nastepny_wektor[1] >= 0:
            y_plus.append(nastepny_wektor[1])
        else:
            y_minus.append(nastepny_wektor[1])
        
        poprzedni_wektor = nastepny_wektor
    
    # Redukcja z użyciem kopca
    suma_x_plus = reduce_heap(x_plus) if x_plus else 0
    suma_x_minus = reduce_heap(x_minus, reverse=True) if x_minus else 0
    suma_y_plus = reduce_heap(y_plus) if y_plus else 0
    suma_y_minus = reduce_heap(y_minus, reverse=True) if y_minus else 0
    
    suma_x = suma_x_plus + suma_x_minus
    suma_y = suma_y_plus + suma_y_minus

    end_time = time.time()
    execution_time = end_time - start_time
    
    # print(f"Czas wykonania dla n={n}: {execution_time:.6f} sekundy")
    
    return suma_x, suma_y, execution_time

# print(sumy_wspolrzednych_wektorow(10))
# print(sumy_wspolrzednych_wektorow_heap(10))

def porownaj_metody(n_values):
    print("n | Δx | Δy | Δt (sekundy)")
    print("-" * 40)
    for n in n_values:
        x1, y1, t1 = sumy_wspolrzednych_wektorow(n)
        x2, y2, t2 = sumy_wspolrzednych_wektorow_heap(n)
        delta_x = abs(x1 - x2)
        delta_y = abs(y1 - y2)
        delta_t = abs(t1 - t2)
        print(f"{n} | {delta_x:.20f} | {delta_y:.20f} | {delta_t:.10f}")

n_values = [4, 10, 100, 500, 1000, 5000, 10000, 100000, 1000000]
porownaj_metody(n_values)

# wynik wychodzi taki
# 4 | 0.00000000000000000000 | 0.00000000000000000000 | 0.0000000000
# 10 | 0.00000000000000022204 | 0.00000000000000022204 | 0.0000000000
# 100 | 0.00000000000000111022 | 0.00000000000000022204 | 0.0000000000
# 500 | 0.00000000000000177636 | 0.00000000000000044409 | 0.0010049343
# 1000 | 0.00000000000000022204 | 0.00000000000000044409 | 0.0010008812
# 5000 | 0.00000000000000066613 | 0.00000000000000333067 | 0.0029997826
# 10000 | 0.00000000000000865974 | 0.00000000000000044409 | 0.0054781437
# 100000 | 0.00000000000001243450 | 0.00000000000001065814 | 0.0949854851
# 1000000 | 0.00000000000000288658 | 0.00000000000001243450 | 1.5011379719

# i patrząc na to że robie takie działanie x1-x2 to wychodzi że wynik po prawej jest zawsze mniejszy.
# Wniosek jest taki że algorytm z kopcem daje dokładniejszy wynik ale wraz ze zwiekszeniem wartości n działa on dłużej