from types import SimpleNamespace
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import contextlib
import os


class ExchangeEconomyClass:

    def __init__(self):
        """
        Baseline parameters
        """
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

    def solve_4a(self):
        """
        Args:
        Returns: solution for optimal price and allocation for agent A
        """
        # Find the optimal price and allocation
        # a. Define empty function to replace with the found solution.
        N = 75
        P1 = np.arange(0.5, 2.5+ (1/(N)*2), 1/(N)*2)
        optimal_price = ()
        optimal_allocation = ()
        max_utility = float("-inf")

        # b. Loop over all prices in the price vector 
        for p1 in P1:

        # i. Call on demand function for B
            x1B, x2B = self.demand_B(p1)

        # ii. Make sure that the amount of each good left for A is positive 
            if 1-x1B > 0 and 1-x2B > 0:
                utility_A = self.utility_A(1-x1B, 1-x2B)
                if utility_A > max_utility:
                    max_utility = utility_A
                    optimal_price = p1
                    optimal_allocation = (1-x1B, 1-x2B)

        # c. Print the solution
        print(f"The optimal price is {optimal_price:.3f}, and the optimal allocation for A is x1 = {optimal_allocation[0]:.3f} and x2 = {optimal_allocation[1]:.3f}.")
        print(f"Resulting in utility {max_utility:.3f} for A.")
        return optimal_price, optimal_allocation

    def solve_5b(self):
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
        return self.sol.x1, self.sol.x2
        
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
        return self.sol.x1, self.sol.x2
    
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
        Returns: 50 pairs of random endowments
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
        Returns: equilibrium price and allocation for agent A

        """
        # a. use draw_pairs to get 50 pairs of endowments
        W_pairs = self.draw_pairs()
        # c. initiate empty list for equilibrium allocations
        equilibrium_allocations = []
        # e. loop through pairs of endowments
        for W in W_pairs:
            # i. set endowments
            self.par.w1A = W[0]
            self.par.w2A = W[1]
            self.par.w1B = 1 - W[0]
            self.par.w2B = 1 - W[1]
            # ii. find equilibrium price using initial guess of 1
            init_guess = 1
            p1 = self.walras(init_guess, eps=1e-8, maxiter=500)
            # iii. find demand for A
            x1A, x2A = self.demand_A(p1)
            # v. append equilibrium allocation for A to equilibrium_allocations
            equilibrium_allocations.append((x1A, x2A))
            #convert to x and y
            x, y = zip(*equilibrium_allocations)
            # vi. convert equilibrium_allocations to numpy array
            x_list = np.array(x)
            y_list = np.array(y)

        return x_list, y_list
        
    def plot_walras_all(self):
        """
        Args:
        Returns: plot of 50 pairs of endowments and equilibrium allocations for agent A

        """
        # a. use equilibrium to get equilibrium allocations
        x_list, y_list = self.equilibrium()
        # b. initiate figure
        # a. total endowment
        w1bar = 1.0
        w2bar = 1.0

        # b. figure set up
        fig = plt.figure(frameon=False,figsize=(6,6), dpi=100)
        ax_A = fig.add_subplot(1, 1, 1)

        ax_A.set_xlabel("$x_1^A$",color="white")
        ax_A.set_ylabel("$x_2^A$", color="white")

        temp = ax_A.twinx()
        temp.set_ylabel("$x_2^B$", color="white")
        ax_B = temp.twiny()
        ax_B.set_xlabel("$x_1^B$", color="white")

        ax_A.tick_params(axis='x', colors='white')
        ax_A.tick_params(axis='y', colors='white')
        ax_B.tick_params(axis='x', colors='white')
        temp.tick_params(axis='y', colors='white')

        # limits
        ax_B.plot([0,w1bar],[0,0],lw=2,color='black')
        ax_B.plot([0,w1bar],[w2bar,w2bar],lw=2,color='black')
        ax_B.plot([0,0],[0,w2bar],lw=2,color='black')
        ax_B.plot([w1bar,w1bar],[0,w2bar],lw=2,color='black')

        ax_A.set_xlim([-0.1, w1bar + 0.1])
        ax_A.set_ylim([-0.1, w2bar + 0.1])    
        ax_B.set_xlim([w1bar + 0.1, -0.1])
        ax_B.set_ylim([w2bar + 0.1, -0.1])
        
        # d. scatter plot of equilibrium allocations
        ax_A.scatter(x_list, y_list, label='Equilibrium allocations')
       
        # f. show plot
        plt.show()

    def plot_walras(self):
        # Make plot of convergence to equilibrium
        # a. create empty list to store the errors
        N = 75
        excess_demands = []

        # b. create a price range
        price_range = np.linspace(0.1,2,N)

        # c. loop over the price range
        for p1 in price_range:
            excess_demand = self.check_market_clearing(p1)
            excess_demands.append(excess_demand)

        # d. convert to numpy array
        excess_demands = np.array(excess_demands)

        # Plot the figure
        plt.figure(figsize=(6, 6))

        # Plot excess demand for good 1
        plt.plot(np.linspace(0,2,N), excess_demands[:, 0], label='Excess demand for good 1')

        # Plot excess demand for good 2
        plt.plot(np.linspace(0,2,N), excess_demands[:, 1],label='Excess demand for good 2', linestyle='dashed')

        # Add a horizontal line at zero to indicate market clearing
        plt.axhline(0, color='black', linewidth=0.5)

        plt.title('Walrasian Equilibrium')
        plt.xlabel('p1')
        plt.ylabel('Market error')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_pareto(self):
        utility_init_A = self.utility_A(self.par.w1A, self.par.w2A) #calling utility function in py-file
        print(f"The utility of the bundle (x1A = {self.par.w1A}, x2A = {self.par.w2A}) is: {utility_init_A:.3f}")

        # b. consumer B
        utility_init_B = self.utility_B(self.par.w1B, self.par.w2B) #calling utility function in py-file
        print(f"The utility of the bundle (x1B = {self.par.w1B:.1f}, x2B = {self.par.w2B}) is: {utility_init_B:.3f}")
        #create grid for x1a and x2a
        N = 75
        x1A_val = np.linspace(0,1,N)
        x2A_val = np.linspace(0,1,N)

        #create a set C
        C = []

        #loop through grid and append pareto optimal values to C
        for x1A in x1A_val:
            for x2A in x2A_val:
                # a. calculate the corresponding consume for B that follows from walras' law:
                x1B = 1-x1A
                x2B = 1-x2A
                # b. calculate utility at every point
                utility_A=self.utility_A(x1A,x2A)
                utility_B=self.utility_B(x1B,x2B)
                # c. check if value is pareto
                if utility_A >= utility_init_A and utility_B >= utility_init_B:
                    # d. append to C if condition is satisfied
                    C.append((x1A,x2A))
        # Plotting the Edgeworth box
        C = np.array(C)

        # a. total endowment
        w1bar = 1.0
        w2bar = 1.0

        # b. figure set up
        fig = plt.figure(frameon=False,figsize=(6,6), dpi=100)
        ax1 = fig.add_subplot(1, 1, 1)

        ax1.set_xlabel("$x_1^A$", color="white")
        ax1.set_ylabel("$x_2^A$", color="white")

        temp = ax1.twinx()
        temp.set_ylabel("$x_2^B$", color="white")
        ax2 = temp.twiny()
        ax2.set_xlabel("$x_1^B$", color="white")

        # a. plot the pareto set
        ax1.scatter(C[:, 0], C[:, 1], s=10, alpha=0.2, color='blue') 

        # Initial endowments
        ax1.scatter(self.par.w1A, self.par.w2A, marker="s", color="black", label="Initial endowment")

        # c. create title
        plt.title('Pareto optimal points', color="white")

        # d. enable grid for primary axes 
        ax1.grid(True)
        plt.tight_layout()

        # e. limits
        ax1.plot([0,w1bar],[0,0],lw=2,color='black')
        ax1.plot([0,w1bar],[w2bar,w2bar],lw=2,color='black')
        ax1.plot([0,0],[0,w2bar],lw=2,color='black')
        ax1.plot([w1bar,w1bar],[0,w2bar],lw=2,color='black')

        ax1.set_xlim([-0.1, w1bar + 0.1])
        ax1.set_ylim([-0.1, w2bar + 0.1])    
        ax2.set_xlim([w1bar + 0.1, -0.1])
        ax2.set_ylim([w2bar + 0.1, -0.1])

        # f. add legend
        ax1.legend(frameon=True, loc='center left', bbox_to_anchor=(1.3, 0.5))

        # g. set tick parameter
        ax1.tick_params(axis='x', colors='white')
        ax1.tick_params(axis='y', colors='white')
        ax2.tick_params(axis='x', colors='white')
        temp.tick_params(axis='y', colors='white')

        plt.show()

    def market_maker(self):
        """
        Args: 
        Returns: solution for x1A, x2A and u_A when consumer A is the market maker
    
        """
        # a. consumer A
        utility_init_A = self.utility_A(self.par.w1A, self.par.w2A) 

        # b. consumer B
        utility_init_B = self.utility_B(self.par.w1B, self.par.w2B) 
        
        #create grid for x1a and x2a
        N = 75
        x1A_val = np.linspace(0,1,N)
        x2A_val = np.linspace(0,1,N)

        #create a set C
        C = []

        #loop through grid and append pareto optimal values to C
        for x1A in x1A_val:
            for x2A in x2A_val:
                # a. calculate the corresponding consume for B that follows from walras' law:
                x1B = 1-x1A
                x2B = 1-x2A
                # b. calculate utility at every point
                utility_A=self.utility_A(x1A,x2A)
                utility_B=self.utility_B(x1B,x2B)
                # c. check if value is pareto
                if utility_A >= utility_init_A and utility_B >= utility_init_B:
                    # d. append to C if condition is satisfied
                    C.append((x1A,x2A))
        C = np.array(C)

        # a. define initial utility and allocation for A
        max_utility_A = -np.inf 
        best_allocation_A = None

        # Loop through optimal allocations for good 1 and good 2
        for allocation in C:
            # b. find the utility
            utility_A = self.utility_A(allocation[0], allocation[1])
            # c. check if the utility is higher than the current max utility
            if utility_A > max_utility_A:
                max_utility_A = utility_A
                best_allocation_A = allocation
        a1 = best_allocation_A[0]
        a2 = best_allocation_A[1]
        print(f'Looping through the set C, the utility is maximised for agent A when x1 = {a1:.3f} and x2 = {a2:.3f} with u = {self.utility_A(a1,a2):.3f}')
        return a1, a2

    def plot_all(self):
        """"
        Args:
        Returns: plot of the Edgeworth box with all the different allocations
        """
        # Suppress print otput
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            # a. consumer A
            utility_init_A = self.utility_A(self.par.w1A, self.par.w2A) #calling utility function in py-file
            
            # b. consumer B
            utility_init_B = self.utility_B(self.par.w1B, self.par.w2B) #calling utility function in py-file
            
            #create grid for x1a and x2a
            N = 75
            x1A_val = np.linspace(0,1,N)
            x2A_val = np.linspace(0,1,N)

            #create a set C
            C = []

            #loop through grid and append pareto optimal values to C
            for x1A in x1A_val:
                for x2A in x2A_val:
                    # a. calculate the corresponding consume for B that follows from walras' law:
                    x1B = 1-x1A
                    x2B = 1-x2A
                    # b. calculate utility at every point
                    utility_A=self.utility_A(x1A,x2A)
                    utility_B=self.utility_B(x1B,x2B)
                    # c. check if value is pareto
                    if utility_A >= utility_init_A and utility_B >= utility_init_B:
                        # d. append to C if condition is satisfied
                        C.append((x1A,x2A))
            # Plotting the Edgeworth box
            C = np.array(C)

            # Show different allocations in the Edgeworth box
            # a. total endowment
            w1bar = 1.0
            w2bar = 1.0

            # b. figure set up
            fig = plt.figure(frameon=False,figsize=(6,6), dpi=100)
            ax_A = fig.add_subplot(1, 1, 1)

            ax_A.set_xlabel("$x_1^A$",color="white")
            ax_A.set_ylabel("$x_2^A$", color="white")

            temp = ax_A.twinx()
            temp.set_ylabel("$x_2^B$", color="white")
            ax_B = temp.twiny()
            ax_B.set_xlabel("$x_1^B$", color="white")

            ax_A.tick_params(axis='x', colors='white')
            ax_A.tick_params(axis='y', colors='white')
            ax_B.tick_params(axis='x', colors='white')
            temp.tick_params(axis='y', colors='white')

            # A
            # a. initial endowment
            ax_A.scatter(self.par.w1A,self.par.w2A,marker='s',color='black',label='initial endowment')

            # b. 3
            # Find market clearing prices
            # a. set initial guess
            init_guess = 2.5
            # b. call the walras equilibrium function in the py-file
            instance = self.walras(init_guess, eps=1e-8, maxiter=500)

            # c. demand functions
            # i. consumer A
            x1A, x2A = self.demand_A(instance)
            ax_A.scatter(x1A,x2A,marker='s',color='orange',label='3 - Walrasian equilibrium')

            # c. 4a
            optimal_price, optimal_allocation = self.solve_4a()
            ax_A.scatter(optimal_allocation[0], optimal_allocation[1], marker='s',color='navy',label='4a - Any price in P1')

            # d. 4b 
            # a. Define the utility function for A
            def uA(p1):
                x1B, x2B = self.demand_B(p1)
                return self.utility_A(1-x1B, 1-x2B)

            # b. Define the constraint
            new_price = [1]
            obj_func = lambda p1: -uA(p1)
            constraint = lambda p1: self.utility_B(1-x1B, 1-x2B)
            constraints = {'type': 'ineq', 'fun': constraint}

            #c. call optimizer 
            res = optimize.minimize(obj_func, new_price ,method='SLSQP', constraints=constraints)
            p1_best_scipy=res.x[0]
            ua_best_scipy=-res.fun
            opt_best_scipy=(1-x1B, 1-x2B)
            ax_A.scatter(opt_best_scipy[0], opt_best_scipy[1] ,marker='s',color='green',label='4b - Any positive price')

            # e. 5a 
            ax_A.scatter(self.market_maker()[0], self.market_maker()[1] ,marker='s',color='gray',label='5a - Choice set restricted to C')

            # f. 5b
            ax_A.scatter(self.solve_5b()[0],self.solve_5b()[1],marker='s',color='yellow',label='5b - Unrestricted choice set')

            # g. 6a
            ax_A.scatter(self.solve_socialplanner()[0], self.solve_socialplanner()[1] ,marker='s',color='red',label='6a - Social planner solution')

            # Possible allocations
            ax_A.scatter(self.set_C()[:, 0], self.set_C()[:, 1], s=15, color='blue', alpha=0.2, label='Pareto allocations')

            # limits
            ax_A.plot([0,w1bar],[0,0],lw=2,color='black')
            ax_A.plot([0,w1bar],[w2bar,w2bar],lw=2,color='black')
            ax_A.plot([0,0],[0,w2bar],lw=2,color='black')
            ax_A.plot([w1bar,w1bar],[0,w2bar],lw=2,color='black')

            ax_A.set_xlim([-0.1, w1bar + 0.1])
            ax_A.set_ylim([-0.1, w2bar + 0.1])    
            ax_B.set_xlim([w1bar + 0.1, -0.1])
            ax_B.set_ylim([w2bar + 0.1, -0.1])

            ax_A.legend(frameon=True,loc='center right',bbox_to_anchor=(2.0,0.5));

            ax_A.legend(frameon=True,loc='center right',bbox_to_anchor=(2.0,0.5));
    def set_C(self):
        """
        Args: 
        Returns: set C of pareto optimal points
        """
                # a. consumer A
        utility_init_A = self.utility_A(self.par.w1A, self.par.w2A) 

        # b. consumer B
        utility_init_B = self.utility_B(self.par.w1B, self.par.w2B) 
        
        #create grid for x1a and x2a
        N = 75
        x1A_val = np.linspace(0,1,N)
        x2A_val = np.linspace(0,1,N)

        #create a set C
        C = []

        #loop through grid and append pareto optimal values to C
        for x1A in x1A_val:
            for x2A in x2A_val:
                # a. calculate the corresponding consume for B that follows from walras' law:
                x1B = 1-x1A
                x2B = 1-x2A
                # b. calculate utility at every point
                utility_A= self.utility_A(x1A,x2A)
                utility_B= self.utility_B(x1B,x2B)
                # c. check if value is pareto
                if utility_A >= utility_init_A and utility_B >= utility_init_B:
                    # d. append to C if condition is satisfied
                    C.append((x1A,x2A))
        C = np.array(C)
        return C


        

