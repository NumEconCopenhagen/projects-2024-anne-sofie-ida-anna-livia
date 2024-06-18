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
        Args: x1A, x2A
        Returns: utility of agent A
        """
        par= self.par
        return  x1A**par.alpha*x2A**(1-par.alpha)

    def utility_B(self,x1B,x2B):
        """
        Args: x1B, x2B
        Returns: utility of agent B
        """
        par=self.par
        return x1B**par.beta*x2B**(1-par.beta)

    def demand_A(self,p1):
        """
        Args: p1
        Returns: demand for good 1 and good 2 for agent A

        """
        # a. set numeraire
        p2 = 1 

        # b. set parameters
        par = self.par

        # c. calculate demand for good 1 and good 2
        x1A = par.alpha*(par.w1A*p1+par.w2A*p2)/p1
        x2A = (1-par.alpha)*(par.w1A*p1+par.w2A*p2)/p2
        return x1A, x2A

    def demand_B(self,p1):
        """
        Args: p1
        Returns: demand for good 1 and good 2 for agent B

        """
        # a. set numeraire
        p2 = 1 

        # b. set parameters
        par = self.par

        # c. calculate demand for good 1 and good 2
        x1B = par.beta*(par.w1B*p1+par.w2B*p2)/p1
        x2B = (1-par.beta)*(par.w1B*p1+par.w2B*p2)/p2
        return x1B, x2B

    def check_market_clearing(self,p1):
        """
        Args: p1
        Returns: excess demand for good 1 and good 2

        """
        # a. set parameters
        par = self.par

        # b. calculate demand for good 1 and good 2 for agent A and B
        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)

        # c. calculate excess demand for good 1 and good 2
        eps1 = x1A-par.w1A + x1B-(1-par.w1A)
        eps2 = x2A-par.w2A + x2B-(1-par.w2A)

        return eps1,eps2

    def walras(self, p1, eps=1e-8, maxiter=500):
        """
        Args: p1, eps, maxiter
        Returns: p1

        """
        # a. set t to 0
        t = 0
        # b. initiate a while loop that breaks when
        # the excess demand for good 1 is smaller than epsilon or t is higher than maxiter
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
        Args: 
        Returns: solution for x1A, x2A and u_A

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
        
    def solve_socialplanner(self):
        """
        Args: 
        Returns: solution for x1A, x2A and u_A

        """
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
        Args: P1
        Returns: market clearing price
    
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
    
    def draw_pairs(self):
        """
        Args:
        Returns: 50 pairs of endowments
        """
        np.random.seed(2000)
        # Generate 50 pairs of endowments
        W_pairs = []
        for _ in range(50):
            pair = (np.random.uniform(0, 1), np.random.uniform(0, 1))
            W_pairs.append(pair)
        return W_pairs
    
    def equilibrium(self):
        """
        Args: p1, eps, maxiter
        Returns: equilibrium price and allocation if in set C
        """
        # a. use draw_pairs to get 50 pairs of endowments
        W_pairs = self.draw_pairs()
        # b. initiate empty list for equilibrium prices
        equilibrium_prices = []
        # c. initiate empty list for equilibrium allocations
        equilibrium_allocations = []
        # d. initiate empty list for equilibrium utilities
        equilibrium_utilities = []
        # e. loop through pairs of endowments
        for W in W_pairs:
            # i. set endowments
            self.par.w1A = W[0]
            self.par.w2A = W[1]
            self.par.w1B = 1 - W[0]
            self.par.w2B = 1 - W[1]
            # ii. find equilibrium price using inital guess of 2.5
            p1 = self.walras(2.5, eps=1e-8, maxiter=500)
            # iii. find equilibrium allocation for A
            self.solve()
            # iv. append equilibrium price to equilibrium_prices
            equilibrium_prices.append(p1)
            # v. append equilibrium allocation to equilibrium_allocations
            equilibrium_allocations.append((self.sol.x1, self.sol.x2))
            # vi. append equilibrium utility to equilibrium_utilities
            equilibrium_utilities.append(self.sol.u)
        
    


 
    
        



        


        

