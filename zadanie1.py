import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def formatFloat(value):
    return format(value, ".15f").rstrip("0").rstrip(".")

def calculateNextVector(prevVector, Theta):
    matrix = np.array([[math.cos(Theta), -math.sin(Theta)], [math.sin(Theta), math.cos(Theta)]], dtype=np.float64)
    return np.dot(matrix, prevVector)

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
    Theta = np.float64(2 * math.pi / n)
    w_0 = np.array([math.cos(Theta) - 1, math.sin(Theta)], dtype=np.float64)
    vectors = [w_0]
    currVector = w_0
    
    while len(vectors) < n:
        nextVector = calculateNextVector(currVector, Theta)
        vectors.append(nextVector)
        currVector = nextVector
        
    return vectors

def sumOfVectors(n):
    vectors = calculateAllVectors(n)
    x_sum = math.fsum(v[0] for v in vectors)
    y_sum = math.fsum(v[1] for v in vectors)
    
    return (x_sum, y_sum)

def sumOfVectorsDifferent(n):
    vectors = calculateAllVectors(n)
    vectors_x_pos, vectors_x_neg, vectors_y_pos, vectors_y_neg = [], [], [], []
    
    for vector in vectors:
        if vector[0] < 0.0:
            vectors_x_neg.append(vector[0])
        else:
            vectors_x_pos.append(vector[0])
        
        if vector[1] < 0.0:
            vectors_y_neg.append(vector[1])
        else:
            vectors_y_pos.append(vector[1])
    
    vectors_x_neg.sort(reverse=True)
    vectors_x_pos.sort()
    vectors_y_neg.sort(reverse=True)
    vectors_y_pos.sort()
    
    x_sum = math.fsum(vectors_x_pos) + math.fsum(vectors_x_neg)
    y_sum = math.fsum(vectors_y_pos) + math.fsum(vectors_y_neg)
    
    return (x_sum, y_sum)

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
    
def drawPlot2(n_values):
    errors = []
    
    for val in n_values:
        vectors = calculateAllVectors(val)
        total_norm = math.fsum(np.linalg.norm(v) for v in vectors)  # Suma norm wektorów
        vector_sum = sumOfVectors(val)
        error = np.linalg.norm(vector_sum) / total_norm  # Błąd względny
        print(error)
        errors.append(error)
    
    print("--------------------------------------------------------------------------------")

    plt.plot(n_values, errors, linestyle="--")
    plt.xlabel("Liczba kątów n")
    plt.ylabel("Norma wektora sumy |sum(wi)|")
    
    # **Formatowanie osi Y do postaci a × 10^b**
    ax = plt.gca()  # Pobiera aktualną oś
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.yaxis.get_offset_text().set_fontsize(12)  # Rozmiar tekstu wykładnika
    ax.yaxis.get_offset_text().set_position((0, 1.02))  # Przesunięcie wykładnika w górę

    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.show()
    
def drawPlot3(n_values):
    errors = []
    
    for val in n_values:
        vectors = calculateAllVectors(val)
        total_norm = math.fsum(np.linalg.norm(v) for v in vectors)  # Suma norm wektorów
        vector_sum = sumOfVectorsDifferent(val)
        error = np.linalg.norm(vector_sum) / total_norm  # Błąd względny
        print(error)
        errors.append(error)
    
    print("--------------------------------------------------------------------------------")

    plt.plot(n_values, errors, linestyle="--")
    plt.xlabel("Liczba kątów n")
    plt.ylabel("Norma wektora sumy |sum(wi)|")
    
    # **Formatowanie osi Y do postaci a × 10^b**
    ax = plt.gca()  # Pobiera aktualną oś
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.yaxis.get_offset_text().set_fontsize(12)  # Rozmiar tekstu wykładnika
    ax.yaxis.get_offset_text().set_position((0, 1.02))  # Przesunięcie wykładnika w górę

    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.show()

def printAllVectors(n):
    for i, (value_x, value_y) in enumerate(calculateAllVectors(n)):
        print(f"w_{i}. ", [formatFloat(value_x), formatFloat(value_y)])
        
def printAllPoints(points):
    for i, (point_x, point_y) in enumerate(points):
        print(f"v_{i}. ", [formatFloat(point_x), formatFloat(point_y)])

def comparision(list1, list2):
    return [a > b for a, b in zip(list1, list2)]
    
def main():
    N = 10
    Theta = np.float64(2 * math.pi / N)
    w_0 = np.array([math.cos(Theta) - 1, math.sin(Theta)], dtype=np.float64)
    v_0 = [np.float64(1), np.float64(0)]
    points = calculateAllPoints(w_0, v_0, N, Theta)
    print(f"N = {N}")
    print("Vector of sum v_i {i = 0, 1, ... , N-1}: ", [formatFloat(sum) for sum in sumOfVectors(N)])
    print("---------------- All vectors ----------------")
    printAllVectors(N)
    print("---------------- All points ----------------")
    printAllPoints(points)
    drawPlot(points)
    
    n_values = np.arange(10, 101, 10)
    drawPlot2(n_values)
    
    n_values2 = np.arange(1000, 100001, 10000)
    drawPlot2(n_values2)
    
    drawPlot3(n_values)
    drawPlot3(n_values2)
    
    list1 = [
        5.776865447512006e-17,
        2.178246710040794e-16,
        2.7749593374671904e-16,
        4.366389559043215e-16,
        3.7521762514616853e-16,
        7.170942648917674e-16,
        5.802501942811156e-16,
        1.4549659367009322e-16,
        9.653602161551593e-16,
        2.2852089674716046e-16
    ]

    list2 = [
        3.592757177872429e-17,
        2.1291146251044555e-16,
        2.7651540219112816e-16,
        4.3036584552857323e-16,
        4.302065385085116e-16,
        7.10639753250698e-16,
        5.667254573082696e-16,
        1.802431047354073e-16,
        9.569748928461308e-16,
        2.499290855688125e-16
    ]
    
    list3 = [
        2.9253904180694534e-15,
        6.818348870251833e-14,
        1.4423995840287497e-13,
        2.553266298773119e-13,
        2.1946977254424066e-13,
        2.040375048142444e-13,
        2.5427590289815345e-13,
        1.2594022013679212e-13,
        5.869653531626222e-13,
        7.468362569979524e-13
    ]
    
    list4 = [
        2.8987052887423823e-15,
        6.821050235112466e-14,
        1.442597208668806e-13,
        2.553013282394499e-13,
        2.194599121626314e-13,
        2.0405026493393717e-13,
        2.54271606456453e-13,
        1.2595385231501354e-13,
        5.869909129018201e-13,
        7.468651323840133e-13
    ]
    
    print("Percentage of vector sums calculated using the separation method that are closer to zero (n = [10, 100]): ", comparision(list1, list2).count(True)/len(comparision(list1, list2)))
    print("Percentage of vector sums calculated using the separation method that are closer to zero (n = [1000, 91000]): ", comparision(list3, list4).count(True)/len(comparision(list3, list4)))

main()