import heapq
import numpy as np
import matplotlib.ticker as ticker
import time
import matplotlib.pyplot as plt


cache = {}

def calculateAllVectors(n):
    if n in cache:
        return cache[n]
    
    Theta = np.float64(2 * np.pi / n)
    rotation_matrix = np.array([[np.cos(Theta), -np.sin(Theta)], 
                                [np.sin(Theta), np.cos(Theta)]], dtype=np.float64)
    
    vectors = np.zeros((n, 2), dtype=np.float64)
    vectors[0] = np.array([np.cos(Theta) - 1, np.sin(Theta)], dtype=np.float64)

    for i in range(1, n):
        vectors[i] = rotation_matrix @ vectors[i - 1]

    cache[n] = vectors
    return vectors

def calculateError(n, sumMethod):
    vectors = calculateAllVectors(n)
    vector_norms = np.linalg.norm(vectors, axis=1).sum()
    vector_sum = np.linalg.norm(sumMethod(vectors))
    return vector_sum / vector_norms

def sumOfVectorsDifferent(vectors):
    x_values, y_values = vectors[:, 0], vectors[:, 1]  # Ekstrakcja kolumn
    
    # Podział na wartości dodatnie i ujemne
    x_pos, x_neg = x_values[x_values >= 0], x_values[x_values < 0]
    y_pos, y_neg = y_values[y_values >= 0], y_values[y_values < 0]

    # Sortowanie stabilne (bez konieczności odwracania kolejności)
    x_pos_sorted = np.sort(x_pos, kind="stable")
    x_neg_sorted = np.sort(x_neg, kind="stable")[::-1]  # Sortowanie malejące
    y_pos_sorted = np.sort(y_pos, kind="stable")
    y_neg_sorted = np.sort(y_neg, kind="stable")[::-1]  # Sortowanie malejące

    # Sumowanie
    x_sum = np.sum(x_pos_sorted) + np.sum(x_neg_sorted)
    y_sum = np.sum(y_pos_sorted) + np.sum(y_neg_sorted)

    return x_sum, y_sum

def heap_sum(values, reverse=False):

    if len(values) == 0:
        return 0
    
    heap = list(values)
    if reverse:
        heap = [-x for x in heap]  # Odwrócenie znaków, aby symulować kopiec max
    heapq.heapify(heap)  # Tworzenie kopca min
    
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        heapq.heappush(heap, a + b)  # Wstawiamy sumę z powrotem
    
    return -heap[0] if reverse else heap[0]  # Odwracamy znak jeśli to kopiec max

def sumOfVectorsHeap(vectors):
    x_values, y_values = vectors[:, 0], vectors[:, 1]

    # Podział na wartości dodatnie i ujemne
    x_pos, x_neg = x_values[x_values >= 0], x_values[x_values < 0]
    y_pos, y_neg = y_values[y_values >= 0], y_values[y_values < 0]

    # Sumowanie przy użyciu kopców
    x_sum = heap_sum(x_pos) + heap_sum(x_neg, reverse=True)
    y_sum = heap_sum(y_pos) + heap_sum(y_neg, reverse=True)

    return x_sum, y_sum

def compare_methods_percentage(errors1, errors2):
    count_method1_better = sum(e1 < e2 for e1, e2 in zip(errors1, errors2))
    count_method2_better = sum(e1 > e2 for e1, e2 in zip(errors1, errors2))
    count_equal = sum(e1 == e2 for e1, e2 in zip(errors1, errors2))
    
    total_cases = len(errors1)
    
    percentage_method1 = (count_method1_better / total_cases) * 100
    percentage_method2 = (count_method2_better / total_cases) * 100
    percentage_equal = (count_equal / total_cases) * 100
    
    print(f"Metoda H3 była lepsza w {count_method1_better} przypadkach ({percentage_method1:.2f}% całości).")
    print(f"Metoda H4(kopiec) była lepsza w {count_method2_better} przypadkach ({percentage_method2:.2f}% całości).")
    print(f"Równość wystąpiła w {count_equal} przypadkach ({percentage_equal:.2f}% całości).")

# ns = list(range(10, 1001, 10))
ns = list(range(1000, 200000, 200))
diff_times = []
heap_times = []

diff_results = []
heap_results = []

diff_errors =[]
heap_errors = []

for n in ns:
    vectors = calculateAllVectors(n)
    
    start = time.time()
    diff_result = sumOfVectorsDifferent(vectors)
    diff_times.append(time.time() - start)
    diff_results.append(diff_result)
    diff_errors.append(calculateError(n, sumOfVectorsDifferent))
    
    start = time.time()
    heap_result = sumOfVectorsHeap(vectors)
    heap_times.append(time.time() - start)
    heap_results.append(heap_result)
    heap_errors.append(calculateError(n, sumOfVectorsHeap))

# print(diff_results)
# print(heap_results)

error_difference = np.abs(np.array(diff_errors) - np.array(heap_errors))

# Wykres czasu działania
plt.figure(figsize=(18, 8))
plt.plot(ns, diff_times, label='Sortowanie')
plt.plot(ns, heap_times, label='Kopiec')
plt.xlabel('n')
plt.ylabel('Czas (s)')
plt.legend()
plt.title('Porównanie czasu działania')
plt.show()

# Wykres błędów
plt.figure(figsize=(18, 8))
plt.plot(ns, error_difference, label='Różnica błędów (Sortowanie - Kopiec)', marker="o")
plt.xlabel('n')
plt.ylabel('Różnica błędów')

ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.16f"))

plt.legend()
plt.title('Porównanie różnicy błędów (Sortowanie vs Kopiec)')
plt.show()

compare_methods_percentage(diff_errors, heap_errors)