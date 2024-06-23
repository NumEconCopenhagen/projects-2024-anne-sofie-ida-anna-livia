import numpy as np
from types import SimpleNamespace
from scipy import optimize
import matplotlib.pyplot as plt
import contextlib
import os
from mpl_toolkits.mplot3d import Axes3D

class ProductionEconomyClass:

    def __init__(self):
        """
        Baseline parameters
        """
        par = self.par = SimpleNamespace()

        # firms
        par.A = 1.0
        par.gamma = 0.5

        # households
        par.alpha = 0.3
        par.nu = 1.0
        par.epsilon = 2.0

        # government
        par.tau = 0.0
        par.T = 0.0

        # Question 3
        par.kappa = 0.1

        # wage
        par.w=1

    def optimal_firm(self,p):
        """
        Finds the optimal behavior of a firm with given prices.

        Args: p, w, gamma, A
        Returns: optimal labor demand, optimal production and profit
        """
        par= self.par

        # Calcalate labor demand, production and profit from given functions. 
        labor_demand=(p*par.A*par.gamma/par.w)**(1/(1-par.gamma))
        production=par.A*(labor_demand**par.gamma)
        profit=((1-par.gamma)/par.gamma)*par.w*((p*par.A*par.gamma)/par.w)**(1/(1-par.gamma))

        return labor_demand, production, profit

    def labor_supply(self, p1, p2):
        """ 
        Finds the optimal labor supply from the consumer using a minimizer.

        Args: p, alpha, w, T, tau, nu, epsilon
        Returns: optimal supply of labor
        """
        par = self.par

        # Prepare for solution
        self.sol = SimpleNamespace()

        profit_1 = self.optimal_firm(p1)[2]
        profit_2 = self.optimal_firm(p2)[2]

        # Define objective function
        def obj(l):
            c1 = par.alpha * (par.w * l + par.T + profit_1 + profit_2) / p1
            c2 = (1 - par.alpha) * (par.w * l + par.T + profit_1 + profit_2) / (p2 + par.tau)
            return -(np.log(c1 ** par.alpha * c2 ** (1 - par.alpha)) - par.nu * (l ** (1 + par.epsilon) / (1 + par.epsilon)))

        # Define bounds and initial guess
        bounds = [(1e-8, 1)]
        x0 = [0.1]

        # Optimize using scipy
        result = optimize.minimize(obj, x0, method='SLSQP', bounds=bounds)

        # Extract solution
        self.sol.l = result.x[0] if result.success else None

        return self.sol.l

    def optimal_consumption(self,p1,p2):
        """
        Optimal behavior of consumer for given prices. 

        Args: p1, p2, w, alpha, T, l, profit_1, profit_2, tau
        Returns: optimal consumption of good 1 and good 2
        """
        par= self.par

        # Calculate optimal consumption of good 1 and good 2 using profit from optimal firm behavior and labour supply from the consumer. 
        profit_1=self.optimal_firm(p=p1)[2]
        profit_2=self.optimal_firm(p=p2)[2]
        l=self.labor_supply(p1,p2)

        c1=par.alpha*(par.w*l+par.T+profit_1+profit_2)/p1
        c2=(1-par.alpha)*(par.w*l+par.T+profit_1+profit_2)/(p2+par.tau)

        return c1, c2
    
    def check_market_clearing(self, p1, p2): 
        """
        Returns excess demand of good 1 and 2 for given prices. 

        Args: p1, p2
        Returns: Excess demand
        """

        # Calculates labor demand, labor supply, production and demand for each good using prior defined functions
        labor_demand1, production1=self.optimal_firm(p1)[0:2]
        labor_demand2, production2=self.optimal_firm(p2)[0:2]
        total_labor_demand=labor_demand1+labor_demand2
        labor_supply=self.labor_supply(p1,p2)

        c1, c2 = self.optimal_consumption(p1,p2)

        # Calculating excess demand
        excess_1=c1-production1
        excess_2=c2-production2
        excess_labor=total_labor_demand-labor_supply

        return excess_1, excess_2, excess_labor 
    
    def plot_excess_demand(self, N=10):
        """
        Plots excess demand of good 1 and 2 for given prices (Here we have used ChatGPT to write a first draft of the code, which we have then adjusted.) 

        Args: N (number of points in the price range) 
        Returns: Plot of excess demand for combinations of p1 and p2 
        """

        price_range = np.linspace(0.1, 2.0, N)
        P1, P2 = np.meshgrid(price_range, price_range)

        excess_1 = np.zeros_like(P1)
        excess_2 = np.zeros_like(P2)
        excess_l = np.zeros_like(P1)

        for i in range(N):
            for j in range(N):
                excess_1[i, j], excess_2[i, j], excess_l[i,j] = self.check_market_clearing(P1[i, j], P2[i, j])

        fig = plt.figure(figsize=(20, 6))

        # 3D plot for excess production of good 1
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot_surface(P1, P2, excess_1, cmap='viridis')
        ax1.set_title('Excess demand of Good 1')
        ax1.set_xlabel('Price of Good 1 (p1)')
        ax1.set_ylabel('Price of Good 2 (p2)')
        ax1.set_zlabel('Excess demand for good 1')
        ax1.view_init(elev=30, azim=45)

        # 3D plot for excess production of good 2
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.plot_surface(P1, P2, excess_2, cmap='viridis')
        ax2.set_title('Excess demand of Good 2')
        ax2.set_xlabel('Price of Good 1 (p1)')
        ax2.set_ylabel('Price of Good 2 (p2)')
        ax2.set_zlabel('Excess demand for good 2')
        ax2.view_init(elev=30, azim=45)

        # 3D plot for excess demand for labor
        ax1 = fig.add_subplot(133, projection='3d')
        ax1.plot_surface(P1, P2, excess_l, cmap='viridis')
        ax1.set_title('Excess demand for labor')
        ax1.set_xlabel('Price of Good 1 (p1)')
        ax1.set_ylabel('Price of Good 2 (p2)')
        ax1.set_zlabel('Excess demand for labor')
        ax1.view_init(elev=30, azim=45) 

        plt.tight_layout()
        plt.show()

    
    def walras(self, p1, p2, eps=1e-8, maxiter=500):
        """
        We find the Walras equilibrium prices using the market clearing function. 

        Args: p1, p2, eps, maxiter
        Returns: p1, p2
        """
        # set t to 0
        t = 0

        # While loop that breaks when the excess demand for good 1 and good 2 is smaller than epsilon or t is higher than maxiter
        while True:
            # Excess demand
            excess1, excess2 = self.check_market_clearing(p1, p2)[0:2]

            # Checking the conditions for the loop 
            if np.abs(excess1) < eps and np.abs(excess2) < eps or t >= maxiter:
                print(f'{t:3d}: p1, p2 = {p1:.8f}, {p2:.8f} -> excess demand of good 1 and good 2-> {excess1:.8f}, {excess2:.8f}')
                break

            # Update p1 and p2 
            p1 += excess1
            p2 += excess2

            # Print during the loop 
            if t < 5 or t % 25 == 0:
                print(f'{t:3d}: p1, p2 = {p1:.8f}, {p2:.8f} -> excess demand of good 1 and good 2-> {excess1:.8f}, {excess2:.8f}')
            elif t == 5:
                print('   ...')

            # Update t (iteration counter)
            t += 1

        return p1, p2
    
    def social_welfare(self, x):
        """
        Define the social welfare function (SWF) utilizing previously defined functions.
        (Here we have used ChatGPT to write a first draft of the code, which we have then adjusted.) 

        Args: tau, p1, p2, w, T, alpha, epsilon, nu, kappa. 
        Returns: social welfare function
        
        """

        # Setting values for variables of interest
        tau, p1, p2 = x

        par = self.par

        # Updating tau
        par.tau = tau

        # Calculate the components of the SWF
        l = self.labor_supply(p1, p2)
        profit_1 = self.optimal_firm(p1)[2]
        profit_2 = self.optimal_firm(p2)[2]
        income = par.w * l + par.T + profit_1 + profit_2
        c1 = par.alpha * income / p1
        c2 = (1 - par.alpha) * income / (p2 + tau)

        # Define T from tau and c2
        par.T = tau * c2

        y2 = self.optimal_firm(p2)[1]

        # Define utility function of the consumer and the SWF
        utility = np.log(c1**par.alpha * c2**(1 - par.alpha)) - par.nu * (l**(1 + par.epsilon) / (1 + par.epsilon))

        swf = utility - par.kappa * y2

        # We return the negative SWF as we will later use a minimizer
        return -swf  

    def optimal_tax(self):
        """
        Finds the optimal value of tau using a minimizer.

        Args: 
        Returns: Optimal value of tau and resulting prices and T

        """
        # Initial guess for tau, p1, and p2
        x0 = [0.1, 1.0, 1.0]

        # Bounds for tau, p1, and p2
        bounds = [(0, 1), (0.1, 2), (0.1, 2)]

        # Optimizing using a minimizer
        result = optimize.minimize(self.social_welfare, x0, method='SLSQP', bounds=bounds)

        # Updating tau and prices if optimal value is found 
        if result.success:
            optimal_tau, optimal_p1, optimal_p2 = result.x
            optimal_T = optimal_tau * self.optimal_consumption(optimal_p1, optimal_p2)[1]
        else:
            optimal_tau, optimal_p1, optimal_p2, optimal_T = None, None, None, None

        return optimal_tau, optimal_p1, optimal_p2, optimal_T
