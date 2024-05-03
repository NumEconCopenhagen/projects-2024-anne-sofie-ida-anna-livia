from scipy import optimize

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

def solve_ss_z(sE, eps, phi, n, g, beta, s, delta, alpha):

obj_zss = lambda zss: zss-(1/(1-sE))**(eps+phi)*(1/((1+n)(1+g)))**beta*(s+(1-delta)*zss)**(1-alpha)*zss**alpha
result = optimize.root_scalar(obj_z,bracket=[0.1,100],method='brentq')

print('The steady state for z in the Solow model with an exhaustable resource and climate change is',result.root)

def solve_ss_k(alpha, s, delta, g, n):
f = lambda k: k**alpha
obj_kss = lambda kss: kss - (s*f(kss) + (1-delta)*kss)/((1+g)*(1+n))
result = optimize.root_scalar(obj_kss,bracket=[0.1,100],method='brentq')

print('The steady state for k in the standard Solow model is',result.root)   
