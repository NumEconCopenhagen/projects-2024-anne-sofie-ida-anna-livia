from types import SimpleNamespace
import numpy as np
from scipy import optimize

class ExchangeEconomyClass:

    def __init__(self):
        par = self.par = SimpleNamespace()

        # a. preferences
        par.alpha = 1/3
        par.beta = 2/3

        # b. endowments
        par.w1A = 0.8
        par.w2A = 0.3
        par.w1B = 1 - par.w1A
        par.w2B = 1 - par.w2A

    def utility_A(self,x1A,x2A):
        """
        Returns the utility of agent A. Takes x1A and x2A as arguments.
        """
        par= self.par
        return  x1A**par.alpha*x2A**(1-par.alpha)

    def utility_B(self,x1B,x2B):
        """
        Returns the utility of agent B. Takes x1A and x2A as arguments.
        """
        par=self.par
        return x1B**par.beta*x2B**(1-par.beta)

    def demand_A(self,p1):
        """
        Returns the demand of agent A. Takes the prices of good 1 and good 2 respectively as arguments.
        """
        p2 = 1 #p2 is numeraire
        par = self.par
        x1A = par.alpha*(par.w1A*p1+par.w2A*p2)/p1
        x2A = (1-par.alpha)*(par.w1A*p1+par.w2A*p2)/p2
        return x1A, x2A

    def demand_B(self,p1):
        """
        Returns the demand of agent B. Takes the prices of good 1 and good 2 respectively as arguments.
        """
        p2 = 1 #p2 is numeraire
        par = self.par
        x1B = par.beta*(par.w1B*p1+par.w2B*p2)/p1
        x2B = (1-par.beta)*(par.w1B*p1+par.w2B*p2)/p2
        return x1B, x2B

    def check_market_clearing(self,p1):
        """
        Calculates the market error on the market for good 1 and good 2. Takes p1 as argument.
        """

        par = self.par

        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)

        eps1 = x1A-par.w1A + x1B-(1-par.w1A)
        eps2 = x2A-par.w2A + x2B-(1-par.w2A)

        return eps1,eps2

    def walras(self, p1, eps=1e-8, maxiter=500):
        """
        Returns the market clearing price based on the lowest market error. takes p1 as argument.
        """
        t = 0
        while True:

            # i. excess demand
            excess = self.check_market_clearing(p1)

            # ii: ensures that the break conditions hold, i.e. that the excess demand for good 1 is not smaller then epsilon
            # and that the number of iterations (t) isn't higher than 500 
            if np.abs(excess[0]) < eps or t >= maxiter:
                print(f'{t:3d}: p1 = {p1:12.8f} -> excess demand -> {excess[0]:14.8f}')
                break

            # iii. updates p1
            p1 += excess[0]

            # iv. return and a lot of formatting for printing
            if t < 5 or t % 25 == 0:
                print(f'{t:3d}: p1 = {p1:12.8f} -> excess demand -> {excess[0]:14.8f}')
            elif t == 5:
                print('   ...')

            # v. update t (interation counter)
            t += 1

        return p1


    def solve(self):
        """
        """
        # Prepare for solution
        self.sol = SimpleNamespace(x1=np.nan, x2=np.nan, u=np.nan)
        par = self.par 

        # b. set objective function, constraints and bounds
        obj = lambda x: -self.utility_A(x[0], x[1])
        bounds = ((1e-8, 1), (1e-8, 1))

        # c. optimize using scipy
        x0 = [par.w1A, par.w2A]
        constraint = lambda x: self.utility_B(1 - x[0], 1 - x[1]) - self.utility_B(1 - par.w1A, 1 - par.w2A)
        constraints = {'type': 'ineq', 'fun': constraint}
        result = optimize.minimize(obj, x0, method='SLSQP', bounds=bounds,constraints=constraints)

        # Update solution
        self.sol.x1 = result.x[0]
        self.sol.x2 = result.x[1]
        self.sol.u = self.utility_A(self.sol.x1, self.sol.x2)
        
    def solve2(self):
        # Prepare for solution
        self.sol = SimpleNamespace(x1=np.nan, x2=np.nan, u=np.nan)
        par = self.par 

        # b. set objective function, constraints and bounds
        obj1 = lambda x: -(self.utility_A(x[0], x[1])+self.utility_B(1-x[0], 1-x[1]))
        bounds = ((1e-8, 1), (1e-8, 1))

        # c. optimize using scipy
        x00 = [par.w1A, par.w2A]
        result_2 = optimize.minimize(obj1, x00, method='SLSQP', bounds=bounds)


        # Update solution
        self.sol.x1 = result_2.x[0]
        self.sol.x2 = result_2.x[1]
        self.sol.u = self.utility_A(self.sol.x1, self.sol.x2)
    
    def market_clearing_p(self, P1):
        """
        finds the market clearing price based on the lowest market error.
        Takes a price vector P1 as input.
        """
        eps1,eps2 = self.check_market_clearing(P1)

        #difference between vectors
        diff = eps1-eps2

        #find minimum error
        error_min = abs(diff).min()

        #pass solutions to vector
        vec = abs(diff) == error_min

        #calculate price of P1 price vector
        market_price = P1[vec][0]

        return market_price

 
    
        



        


        

