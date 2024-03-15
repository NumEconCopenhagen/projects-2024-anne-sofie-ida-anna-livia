from types import SimpleNamespace
import numpy as np

class ExchangeEconomyClass:

    def __init__(self):
        par = self.par = SimpleNamespace()

        # a. preferences
        par.alpha = 1/3
        par.beta = 2/3

        # b. endowments
        par.w1A = 0.8
        par.w2A = 0.3
        par.w1B = 1 - 0.8
        par.w2B = 1 - 0.3

    def utility_A(self,x1A,x2A):
        return  x1A**self.par.alpha*x2A**(1-self.par.alpha)

    def utility_B(self,x1B,x2B):
        return x1B**self.par.beta*x2B**(1-self.par.beta)

    def demand_A(self,p1):
        p2 = 1 #p2 is numeraire
        par = self.par
        x1A = par.alpha*(par.w1A*p1+par.w2A*p2)/p1
        x2A = (1-par.alpha)*(par.w1A*p1+par.w2A*p2)/p2
        return x1A, x2A

    def demand_B(self,p1):
        p2 = 1 #p2 is numeraire
        par = self.par
        x1B = par.beta*(par.w1A*p1+par.w2B*p2)/p1
        x2B = (1-par.beta)*(par.w1B*p1+par.w2B*p2)/p2
        return x1B, x2B

    def check_market_clearing(self,p1):

        par = self.par

        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)

        eps1 = x1A-par.w1A + x1B-(1-par.w1A)
        eps2 = x2A-par.w2A + x2B-(1-par.w2A)

        return eps1,eps2

    def walras(self, p1, eps=1e-8, maxiter=500):
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



        

