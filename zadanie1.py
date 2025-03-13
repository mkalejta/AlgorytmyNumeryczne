import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

cache = {}

def formatFloat(value):
    return format(value, ".15f").rstrip("0").rstrip(".")

def calculateNextVector(prevVector, Theta):
    cosT, sinT = np.cos(Theta), np.sin(Theta)
    return np.array([
        prevVector[0] * cosT - prevVector[1] * sinT,
        prevVector[0] * sinT + prevVector[1] * cosT
    ], dtype=np.float64)

def calculateNextPoint(prevPoint, vector):
    return np.add(prevPoint, vector)

def calculateAllPoints(startVector, startPoint, n, Theta):
    allPoints = [startPoint]
    currentVector = startVector
    currentPoint = calculateNextPoint(startPoint, startVector)
    for _ in range(n):
        allPoints.append(currentPoint)
        currentVector = calculateNextVector(currentVector, Theta)
        currentPoint = calculateNextPoint(currentPoint, currentVector)
    return allPoints

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

def sumOfVectors(vectors):
    return np.sum(vectors, axis=0)

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

def calculateError(n, sumMethod):
    vectors = calculateAllVectors(n)
    vector_norms = np.linalg.norm(vectors, axis=1).sum()
    vector_sum = np.linalg.norm(sumMethod(vectors))
    return vector_sum / vector_norms

def drawPlot(points, offset=0.1, margin=0.5):
    # Rozdzielenie współrzędnych wszystkich punktów
    xValues = [x for x, _ in points]
    yValues = [y for _, y in points]
    
    # Obliczenie środka okręgu (dla wielokąta foremnego to środek masy)
    xCenter = sum(xValues[:-1]) / len(points)
    yCenter = sum(yValues[:-1]) / len(points)
    
    # Promień okręgu
    radius = 1
    
    # Tworzenie wykresów
    plt.plot(xValues, yValues, marker = 'o', mfc = 'red')
    
    # **Dostosowanie zakresu osi**
    xMin, xMax = min(xValues), max(xValues)
    yMin, yMax = min(yValues), max(yValues)
    plt.xlim(xMin - margin, xMax + margin)
    plt.ylim(yMin - margin, yMax + margin)
    
    # Etykiety do punktów
    for i, (x, y) in enumerate(points):
        # Obliczenie wektora przesunięcia
        dx = x - xCenter
        dy = y - yCenter
        norm = np.sqrt(dx**2 + dy**2)
        xLabel = x + offset * dx / norm  # Przesunięcie w kierunku radialnym
        yLabel = y + offset * dy / norm
        if i == len(points):
            xLabel = x - offset * dx / norm
            yLabel = y / norm
        plt.text(xLabel, yLabel, "v" + str(i), fontsize=10, ha='center', va='center', color='black')
        
    circle = plt.Circle((xCenter, yCenter), radius, color='red', fill=False)
    plt.gca().add_patch(circle)
    
    # Ustawienie osi i legendy
    plt.axis("equal")
    plt.legend()
    plt.show()
    
def drawPlot2(n_values, errors):
    plt.figure(figsize=(18, 8))
    plt.plot(n_values, errors, "o")
    plt.xlabel("Liczba kątów n")
    plt.ylabel("Norma wektora sumy |sum(wi)|")
    
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.18f"))

    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.show()

def drawPlot3(n_values, errors_std, errors_dif):
    errors_differences = np.abs(errors_std - errors_dif)

    plt.figure(figsize=(16, 8))
    plt.plot(n_values, errors_differences, marker="o")
    plt.xlabel("Liczba kątów n")
    plt.ylabel("Norma wektora sumy |sum(wi)|")
    
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.16f"))

    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.show()

def printAllVectors(n):
    for i, (value_x, value_y) in enumerate(calculateAllVectors(n)):
        print(f"w_{i}. ", [formatFloat(value_x), formatFloat(value_y)])
        
def printAllPoints(points):
    for i, (point_x, point_y) in enumerate(points):
        print(f"v_{i}. ", [formatFloat(point_x), formatFloat(point_y)])
    
def calculateAccurancy(errors_std, errors_dif):
    return np.mean(errors_std > errors_dif)

def main():
    N = 10
    Theta = np.float64(2 * np.pi / N)
    w_0 = np.array([np.cos(Theta) - 1, np.sin(Theta)], dtype=np.float64)
    v_0 = [np.float64(1), np.float64(0)]
    points = calculateAllPoints(w_0, v_0, N, Theta)
    vectors = calculateAllVectors(N)
    print(f"N = {N}")
    print("Vector of sum v_i {i = 0, 1, ... , N-1}: ", [formatFloat(sum) for sum in sumOfVectors(vectors)])
    print("---------------- All vectors ----------------")
    printAllVectors(N)
    print("---------------- All points ----------------")
    printAllPoints(points)
    # drawPlot(points)
    
    n_values = np.arange(10, 1001, 10)
    n_values2 = np.arange(1000, 100001, 1000)
    errors_std = np.array([calculateError(n, sumOfVectors) for n in n_values])
    errors_dif = np.array([calculateError(n, sumOfVectorsDifferent) for n in n_values])
    errors_std2 = np.array([calculateError(n, sumOfVectors) for n in n_values2])
    errors_dif2 = np.array([calculateError(n, sumOfVectorsDifferent) for n in n_values2])
    
    drawPlot2(n_values, errors_std)
    drawPlot2(n_values2, errors_std2)
    
    drawPlot3(n_values, errors_std, errors_dif)
    drawPlot3(n_values2, errors_std2, errors_dif2)
    
    print(calculateAccurancy(errors_std, errors_dif))
    print(calculateAccurancy(errors_std2, errors_dif2))

main()