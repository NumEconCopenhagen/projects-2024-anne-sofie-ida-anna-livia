import numpy as np
from types import SimpleNamespace
import matplotlib.pyplot as plt

class Graduate:
    def __init__(self):
        """Initialize parameters"""
        self.par = SimpleNamespace()
        self.par.J = 3
        self.par.N = 10
        self.par.K = 10000

        self.par.F = np.arange(1, self.par.N + 1)
        self.par.sigma = 2

        self.par.v = np.array([1, 2, 3])
        self.par.c = 1

    def eps_sim(self, mu, sigma, shape):
        """Simulate epsilon values"""
        np.random.seed(2024)
        return np.random.normal(mu, sigma, shape)

    def avg_exp_util(self):
        par = self.par
        u = par.v + (1/par.K) * np.sum(self.eps_sim(0, par.sigma, (par.J, par.K)), axis=1)
        return u
    
    def avg_realised_util(self):
        par = self.par
        epsilon = self.eps_sim(0, par.sigma, (par.J, par.N))
        u_realised = self.par.v[:, None] + epsilon
        avg_u_realised = np.mean(u_realised, axis=1)
        return avg_u_realised

    def step1(self):
        par = self.par
        
        eps_friends = np.zeros((par.N, par.J, par.F[-1], par.K))
        
        # a. draw J*F epsilon values for each graduate friends
        for i in range(par.N):
            Fi = par.F[i]
            eps_friends[i, :, :Fi, :] = self.eps_sim(0, par.sigma, (par.J, Fi, par.K))

        # b. draw for individual him/herself
        eps_individual = self.eps_sim(0, par.sigma, (par.N, par.J, par.K))

        # c. calculate career choices and expected and actual utility
        career = np.zeros((par.N, par.K), dtype=int)
        exp_u = np.zeros((par.N, par.K))
        actual_u = np.zeros((par.N, par.K))

        # d. loop over individuals and friends
        for k in range(par.K):
            for i in range(par.N):
                Fi = par.F[i]
                prior = np.zeros(par.J)
                for j in range(par.J):
                    prior[j] = par.v[j] + np.mean(eps_friends[i, j, :Fi, k])
                chosen_career = np.argmax(prior)
                career[i, k] = chosen_career
                exp_u[i, k] = prior[chosen_career]
                actual_u[i, k] = par.v[chosen_career] + eps_individual[i, chosen_career, k]
        
        # Calculate the required metrics
        career_shares = np.zeros((par.N, par.J))
        avg_subjective_utilities = np.zeros(par.N)
        avg_realised_utilities = np.zeros(par.N)

        for i in range(par.N):
            for j in range(par.J):
                career_shares[i, j] = np.mean(career[i, :] == j)
            avg_subjective_utilities[i] = np.mean(exp_u[i, :])
            avg_realised_utilities[i] = np.mean(actual_u[i, :])

        return career_shares, avg_subjective_utilities, avg_realised_utilities
    
    def plot_share(self, career_shares):
        par = self.par
        fig, ax = plt.subplots()
        for j in range(par.J):
            ax.plot(par.F, career_shares[:, j], label=f"Career {j+1}")
        ax.set_xlabel("Individuals")
        ax.set_ylabel("Career share")
        ax.legend()
        plt.show()
    
    def plot_exp_util(self, avg_subjective_utilities):
        par = self.par
        fig, ax = plt.subplots()
        ax.plot(par.F, avg_subjective_utilities, marker='o', label="Average Subjective Utility")
        ax.set_xlabel("Individuals")
        ax.set_ylabel("Average expected utility")
        plt.show()

    def plot_realised_util(self, avg_realised_utilities):
        par = self.par
        fig, ax = plt.subplots()
        ax.plot(par.F, avg_realised_utilities, marker='o', label="Average Realised Utility")
        ax.set_xlabel("Individuals")
        ax.set_ylabel("Average realised utility")
        plt.show()
    
  

    



    