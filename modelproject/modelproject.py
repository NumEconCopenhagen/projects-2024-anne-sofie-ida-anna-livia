from scipy import optimize
from types import SimpleNamespace

def solve_ss(alpha, c):
    """ Example function. Solve for steady state k. 

    Args:
        c (float): costs
        alpha (float): parameter

    Returns:
        result (RootResults): the solution represented as a RootResults object.

    """ 
    
    # a. Objective function, depends on k (endogenous) and c (exogenous).
    f = lambda k: k**alpha - c
    obj = lambda kss: kss - f(kss)

    #. b. call root finder to find kss.
    result = optimize.root_scalar(obj,bracket=[0.1,100],method='bisect')
    
    return result

class Solowclass:
    def paramtervalues(self):
        """ Defining parameter values for both the baseline model and the extended model. 

        Args:
            ???

        Returns:
            ???
        """ 

    def __init__(self):
        self.par = SimpleNamespace()

    def paramtervalues(self):

        par=self.par

        # ISN: JEG HAR INDSAT VÆRDIERNE FRA EKSAMENSSÆTTET
        # Baseline Solow model
        par.alpha=0.2
        par.s=0.3
        par.n=0.01 
        par.g=0.027
        par.delta=0.05 

        # Extended Solow model
        par.beta=0.6
        par.eps=0.2
        par.sE=0.005
        par.phi=0.5

    # ISN: KOPIERET ASH'S LØSNINGER MEN INDSAT PARAMETERVÆRDIER 

    def solve_ss_z_par(self, zss):

        par=self.par

        obj_zss = lambda zss: zss-(1/(1-par.sE))**(par.eps+par.phi)*(1/((1+par.n)*(1+par.g)))**par.beta*(par.s+(1-par.delta)*zss)**(1-par.alpha)*zss**par.alpha
        result = optimize.root_scalar(obj_zss,bracket=[0.1,100],method='brentq')

        print('The steady state for z in the Solow model with an exhaustable resource and climate change is',result.root)

    def solve_ss_k_par(self, kss):
        par=self.par

        f = lambda k: k**par.alpha
        obj_kss = lambda kss: kss - (par.s*f(kss) + (1-par.delta)*kss)/((1+par.g)*(1+par.n))
        result = optimize.root_scalar(obj_kss,bracket=[0.1,100],method='brentq')

        print('The steady state for k in the standard Solow model is',result.root)   

