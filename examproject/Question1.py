import numpy as np
from types import SimpleNamespace
from scipy import optimize
import matplotlib.pyplot as plt
import contextlib
import os

# For the 3D plot 
from mpl_toolkits.mplot3d import Axes3D

from scipy.optimize import minimize

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
        Finds the optimal behavior of a firm with given price

        Args: p, w, gamma, A
        Returns: optimal labor demand, optimal production
        """
        par= self.par
        labor_demand=(p*par.A*par.gamma/par.w)**(1/(1-par.gamma))
        production=par.A*(labor_demand**par.gamma)
        profit=((1-par.gamma)/par.gamma)*par.w*((p*par.A*par.gamma)/par.w)**(1/(1-par.gamma))
        return labor_demand, production, profit

    def labor_supply(self, p1, p2):
        par = self.par

        # Prepare for solution
        self.sol = SimpleNamespace()

        labor_demand_1, production_1, profit_1 = self.optimal_firm(p1)
        labor_demand_2, production_2, profit_2 = self.optimal_firm(p2)

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
        """
        par= self.par

        labor_1, y_1, profit_1=self.optimal_firm(p=p1)
        labor_2, y_2, profit_2=self.optimal_firm(p=p2)
        l=self.labor_supply(p1,p2)

        c1=par.alpha*(par.w*l+par.T+profit_1+profit_2)/p1
        c2=(1-par.alpha)*(par.w*l+par.T+profit_1+profit_2)/(p2+par.tau)

        return c1, c2
    
    def check_market_clearing_A(self, p1, p2): 
        """
        Returns excess demand of good 1 and 2 for given prices 
        Args: p1, p2
        Returns: Excess demand
        """
        par= self.par

        labor_demand1, production1, profit1=self.optimal_firm(p1)
        labor_demand2, production2, profit2=self.optimal_firm(p2)
        total_labor_demand=labor_demand1+labor_demand2
        labor_supply=self.labor_supply(p1,p2)

        c1, c2 = self.optimal_consumption(p1,p2)

        excess_1=c1-production1
        excess_2=c2-production2
        excess_labor=total_labor_demand-labor_supply

        return excess_1, excess_2, excess_labor 
    
    def plot_excess_demand(self, N=10):
        """
        Plots excess demand of good 1 and 2 for given prices 
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
                excess_1[i, j], excess_2[i, j], excess_l[i,j] = self.check_market_clearing_A(P1[i, j], P2[i, j])

        fig = plt.figure(figsize=(14, 6))

        # 3D plot for excess production of good 1
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot_surface(P1, P2, excess_1, cmap='viridis')
        ax1.set_title('Excess demand of Good 1')
        ax1.set_xlabel('Price of Good 1 (p1)')
        ax1.set_ylabel('Price of Good 2 (p2)')
        ax1.set_zlabel('Excess demand 1')
        ax1.view_init(elev=30, azim=45)

        # 3D plot for excess production of good 2
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.plot_surface(P1, P2, excess_2, cmap='viridis')
        ax2.set_title('Excess demand of Good 2')
        ax2.set_xlabel('Price of Good 1 (p1)')
        ax2.set_ylabel('Price of Good 2 (p2)')
        ax2.set_zlabel('Excess demand 2')
        ax2.view_init(elev=30, azim=45)

        # 3D plot for excess production of good 1
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
        Args: p1, p2, eps, maxiter
        Returns: p1, p2
        """
        # a. set t to 0
        t = 0
        # b. initiate a while loop that breaks when the excess demand for good 1 is smaller than epsilon or t is higher than maxiter
        while True:
            # i. excess demand
            excess1, excess2, excess_l = self.check_market_clearing_A(p1, p2)

            # ii: ensures that the break conditions hold, i.e. that the excess demand for good 1 is not smaller then epsilon
            # and that the number of iterations (t) isn't higher than 500 
            if np.abs(excess1) < eps and np.abs(excess2) < eps or t >= maxiter:
                print(f'{t:3d}: p1, p2 = {p1:.8f}, {p2:.8f} -> excess demand of good 1 and good 2-> {excess1:.8f}, {excess2:.8f}')
                break

            # iii. updates p1 and p2 
            p1 += excess1
            p2 += excess2

            # iv. Printing during the loop 
            if t < 5 or t % 25 == 0:
                print(f'{t:3d}: p1, p2 = {p1:.8f}, {p2:.8f} -> excess demand of good 1 and good 2-> {excess1:.8f}, {excess2:.8f}')
            elif t == 5:
                print('   ...')

            # v. update t (iteration counter)
            t += 1

        return p1, p2
    
    def social_welfare(self): 

        self.sol = SimpleNamespace()

        par= self.par

        l1, y1, profit1=self.optimal_firm(p1)
        l2, y2, profit2=self.optimal_firm(p2)

        def obj(tax, p1, p2):
            l=self.labor_supply(p1, p2)
            c2 = (1 - par.alpha) * (par.w * l + tax*y2 + profit1 + profit2) / (p2 + tax)
            c1 = par.alpha * (par.w * l + tax*y2 + profit1 + profit2) / p1
            return -(np.log(c1**par.alpha*c2**(1-par.alpha))-par.nu*(l**(1+par.epsilon))/(1+par.epsilon)-par.kappa*y2)

        # Define bounds and initial guess
        bounds = [(1e-8, 1)]
        x0 = [0.15]

        # Optimize using scipy
        result = optimize.minimize(obj, x0, method='SLSQP', bounds=bounds)

        # Extract solution
        self.sol.tax = result.x[0] if result.success else None

        return self.sol.tax
    
    def social_welfare_GPT(self, x):
        tau, p1, p2 = x

        par = self.par
        par.tau = tau

        # Calculate labor supply, profits, and consumption with the given tau
        l = self.labor_supply(p1, p2)
        profit_1 = self.optimal_firm(p1)[2]
        profit_2 = self.optimal_firm(p2)[2]
        income = par.w * l + par.T + profit_1 + profit_2
        c1 = par.alpha * income / p1
        c2 = (1 - par.alpha) * income / (p2 + tau)

        # Update the transfer T based on the new c2
        par.T = tau * c2

        y2 = self.optimal_firm(p2)[1]

        utility = np.log(c1**par.alpha * c2**(1 - par.alpha)) - par.nu * (l**(1 + par.epsilon) / (1 + par.epsilon))
        swf = utility - par.kappa * y2
        return -swf  # Negate because we use a minimizer to maximize

    def find_optimal_tax_and_prices_GPT(self):
        # Initial guess for tau, p1, and p2
        x0 = [0.1, 1.0, 1.0]
        # Bounds for tau, p1, and p2
        bounds = [(0, 1), (0.1, 2), (0.1, 2)]

        # Optimize using scipy
        result = minimize(self.social_welfare_GPT, x0, method='SLSQP', bounds=bounds)

        if result.success:
            optimal_tau, optimal_p1, optimal_p2 = result.x
            optimal_T = optimal_tau * self.optimal_consumption(optimal_p1, optimal_p2)[1]
        else:
            optimal_tau, optimal_p1, optimal_p2, optimal_T = None, None, None, None

        return optimal_tau, optimal_p1, optimal_p2, optimal_T
