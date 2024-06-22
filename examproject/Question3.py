import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt

class BuildingBlockII:
    def __init__(self, X, y, f):
        """
        Initializes class with points X and y
        """
        rng = np.random.default_rng(2024)
        self.X = rng.uniform(size=(50,2))
        self.y = rng.uniform(size=(2,))
        self.f = lambda x: x[0]*x[1]

    def find_distance(self, p1, p2):
        """
        Args: 
            p1 (tuple): point 1
            p2 (tuple): point 2
        Returns:
            float: Euclidean distance between two points
        """
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def points(self):
        """
        Args: 
            X (array): uniformly distributed random points
            y (tuple): point y
        Returns:
            tuple: A, B, C, and D
        """
        # Initial setup
        # a. set initial values of A, B, C, and D to None
        A = B = C = D = None
        X, y = self.X, self.y

        # b. Set initial values of minimum distances to infinity
        min_dist_A = min_dist_B = min_dist_C = min_dist_D = np.inf
    
        # Calculate the minimum distances
        # a. initiate loop through all points in X
        for point in X:
            # b. calculate distance if x1 > y1 and x2 > y2
            if point[0] > y[0] and point[1] > y[1]:
                dist = self.find_distance(point, y)
                # c. update minimum distance and point if new distance is smaller
                if dist < min_dist_A:
                    min_dist_A = dist
                    A = point
            # d. calculate distance if x1 > y1 and x2 < y2
            elif point[0] > y[0] and point[1] < y[1]:
                dist = self.find_distance(point, y)
                if dist < min_dist_B:
                    min_dist_B = dist
                    B = point
            # e. calculate distance if x1 < y1 and x2 < y2
            elif point[0] < y[0] and point[1] < y[1]:
                dist = self.find_distance(point, y)
                if dist < min_dist_C:
                    min_dist_C = dist
                    C = point
            # f. calculate distance if x1 < y1 and x2 > y2
            elif point[0] < y[0] and point[1] > y[1]:
                dist = self.find_distance(point, y)
                if dist < min_dist_D:
                    min_dist_D = dist
                    D = point 
        return A, B, C, D

    def barycentric_coordinates_ABC(self, y, A, B, C, D):
        """
        Args:
            y (tuple): point y
            A (tuple): point A
            B (tuple): point B
            C (tuple): point C
        Returns:
            tuple: Barycentric coordinates (r1, r2, r3) of point y with respect to triangle ABC
        """
        # a. Calculate the denominator in the fraction
        denom = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])

        # b. Calculate the barycentric coordinates
        r1 = ((B[1] - C[1]) * (y[0] - C[0]) + (C[0] - B[0]) * (y[1] - C[1])) / denom
        r2 = ((C[1] - A[1]) * (y[0] - C[0]) + (A[0] - C[0]) * (y[1] - C[1])) / denom
        r3 = 1 - r1 - r2
        return r1, r2, r3
 
    def barycentric_coordinates_CDA(self, y, A, B, C, D):
        """
        Args:
            y (tuple): point y
            C (tuple): point C
            D (tuple): point D
            A (tuple): point A
        Returns:
            tuple: Barycentric coordinates (r1, r2, r3) of point y with respect to triangle CDA
        """
        # a. Calculate the denominator in the fraction
        denom = (D[1] - A[1]) * (C[0] - A[0]) + (A[0] - D[0]) * (C[1] - A[1])
        # b. Calculate the barycentric coordinates
        r1 = ((D[1] - A[1]) * (y[0] - A[0]) + (A[0] - D[0]) * (y[1] - A[1])) / denom
        r2 = ((A[1] - C[1]) * (y[0] - A[0]) + (C[0] - A[0]) * (y[1] - A[1])) / denom
        r3 = 1 - r1 - r2
        return r1, r2, r3

    def check_containment(self, A, B, C, D):
        """
        Args:
            A (tuple): point A
            B (tuple): point B
            C (tuple): point C
            D (tuple): point D
        Returns:
            tuple: Barycentric coordinates with respect to ABC, CDA and which triangle contains y
        """
        y = self.y

        # a. Compute barycentric coordinates for y with respect to triangle ABC
        if A is not None and B is not None and C is not None:
            r_ABC = self.barycentric_coordinates_ABC(y, A, B, C, D)
        else:
            r_ABC = (None, None, None)

        # b. Compute barycentric coordinates for y with respect to triangle CDA
        if C is not None and D is not None and A is not None:
            r_CDA = self.barycentric_coordinates_CDA(y, A, B, C, D)
        else:
            r_CDA = (None, None, None)

        # c. Check if y is inside triangle ABC
        inside_ABC = all(0 <= r <= 1 for r in r_ABC)

        # d. Check if y is inside triangle CDA
        inside_CDA = all(0 <= r <= 1 for r in r_CDA)

        # e. update containing_y
        containing_y = None
        if inside_ABC:
            containing_y = "ABC"
        elif inside_CDA:
            containing_y = "CDA"

        return r_ABC, r_CDA, containing_y

    def plot(self, A, B, C, D):
        """
        Args:
            X (array): uniformly distributed random points
            y (tuple): point y
            A (tuple): point A
            B (tuple): point B
            C (tuple): point C
            D (tuple): point D

        Returns:
            a plot of the points and the outlinead triangles ABC and CDA
        """
        X, y = self.X, self.y
        plt.figure(figsize=(8, 8))
        
        # Plot points
        # a. scatter X
        plt.scatter(X[:, 0], X[:, 1], c='blue', label='X')
        # b. scatter y
        plt.scatter(y[0], y[1], c='red', label='y', zorder=5)

        # c. scatter A, B, C, and D if they are not None
        if A is not None:
            plt.scatter(A[0], A[1], c='green', label='A', zorder=5)
        if B is not None:
            plt.scatter(B[0], B[1], c='purple', label='B', zorder=5)
        if C is not None:
            plt.scatter(C[0], C[1], c='orange', label='C', zorder=5)
        if D is not None:
            plt.scatter(D[0], D[1], c='brown', label='D', zorder=5)

        # d. Draw triangles
        if A is not None and B is not None and C is not None:
            plt.plot([A[0], B[0]], [A[1], B[1]], 'r-')
            plt.plot([B[0], C[0]], [B[1], C[1]], 'r-')
            plt.plot([C[0], A[0]], [C[1], A[1]], 'r-')
        if C is not None and D is not None and A is not None:
            plt.plot([C[0], D[0]], [C[1], D[1]], 'k-')
            plt.plot([D[0], A[0]], [D[1], A[1]], 'k-')
            plt.plot([A[0], C[0]], [A[1], C[1]], 'k-')

        # e. Set labels and title
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Plot of points and solutions')
        plt.legend()
        plt.grid(True)
        plt.show()

    def approx(self, A, B, C, D, r_ABC, r_CDA):
        """
        Args:
            A (tuple): point A
            B (tuple): point B
            C (tuple): point C
            D (tuple): point D
            r_ABC (tuple): barycentric coordinates of y with respect to ABC
            r_CDA (tuple): barycentric coordinates of y with respect to CDA
        Returns:
            float: approximation of f(y)
        """
        # Compute the approximation of f(y)

        # a. set initial value of approx to None
        approx = None

        # b. calculate approximation for
        if A is not None and B is not None and C is not None:
            approx = r_ABC[0] * self.f(A) + r_ABC[1] * self.f(B) + r_ABC[2] * self.f(C)
        elif C is not None and D is not None and A is not None:
            approx = r_CDA[0] * self.f(C) + r_CDA[1] * self.f(D) + r_CDA[2] * self.f(A)
        return approx

    def truevalue(self):
        """
        Returns:
            float: true value of f(y)
        """
        true = self.f(self.y)
        return true
 
    def error(self, approx, true):
        """
        Args:
            approx (float): approximation of f(y)
            true (float): true value of f(y)
        Returns:
            float: error
        """
        return np.abs(approx - true)            

    def algorithm(self, y):
        """
        Args:
            y (tuple): point y
        Returns:
            tuple: A, B, C, D, r_ABC, r_CDA, containing_y, approx, true, error
        """
        # Compute points A, B, C, and D
        A, B, C, D = self.points()

        # Compute barycentric coordinates for y with respect to triangle ABC
        if A is not None and B is not None and C is not None:
            r_ABC = self.barycentric_coordinates_ABC(y, A, B, C, D)
        else:
            r_ABC = (None, None, None)

        # Compute barycentric coordinates for y with respect to triangle CDA
        if C is not None and D is not None and A is not None:
            r_CDA = self.barycentric_coordinates_CDA(y, A, B, C, D)
        else:
            r_CDA = (None, None, None)

        # Determine which triangle y is located inside
        triangle_containing_y = None
        if r_ABC != (None, None, None) and all(0 <= r <= 1 for r in r_ABC):
            triangle_containing_y = "ABC"
        elif r_CDA != (None, None, None) and all(0 <= r <= 1 for r in r_CDA):
            triangle_containing_y = "CDA"

        # Compute the approximation of f(y)
        approx = None
        if triangle_containing_y == "ABC":
            approx = r_ABC[0] * self.f(A) + r_ABC[1] * self.f(B) + r_ABC[2] * self.f(C)
        elif triangle_containing_y == "CDA":
            approx = r_CDA[0] * self.f(C) + r_CDA[1] * self.f(D) + r_CDA[2] * self.f(A)

        # Compute the true value of f(y)
        true = self.f(y)

        # Compare the approximation with the true value
        error = None
        if approx is not None:
            error = np.abs(approx - true)

        return A, B, C, D, r_ABC, r_CDA, triangle_containing_y, approx, true, error
