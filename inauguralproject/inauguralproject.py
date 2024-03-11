from types import SimpleNamespace

class ExchangeEconomyClass:

    def __init__(self):
        par = self.par = SimpleNamespace()

        # a. preferences
        par.alpha = 1/3
        par.beta = 2/3

        # b. endowments
        par.w1A = 0.8
        par.w2A = 0.3
        par.w1b = 1 - 0.8
        par.w2b = 1 - 0.3

    def utility_A(self,x1A,x2A):
        return  x1a**self.par.alpha*x2a**(1-self.par.alpha)

    def utility_B(self,x1B,x2B):
        return x1b**self.par.beta*x2b**(1-self.par.beta)

    def demand_A(self,p1):
        par = self.par
        x1a = par.alpha*(par.w1a*p1+par.w2a*p2)/p1
        x2a = (1-par.alpha)*(par.w1a*p1+par.w2a*p2)/p2
        return x1a, x2a

    def demand_B(self,p1):
        par = self.par
        x1b = par.beta*(par.w1b*p1+par.w2b*p2)/p1
        x2b = (1-par.beta)*(par.w1b*p1+par.w2b*p2)/p2
        return x1b, x2b


    def check_market_clearing(self,p1):

        par = self.par

        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)

        eps1 = x1A-par.w1A + x1B-(1-par.w1A)
        eps2 = x2A-par.w2A + x2B-(1-par.w2A)

        return eps1,eps2

