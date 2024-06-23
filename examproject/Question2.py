import numpy as np
from types import SimpleNamespace
import matplotlib.pyplot as plt

class Graduate:
    def __init__(self):
        """Initialize parameters"""
        np.random.seed(2024)
        self.par = SimpleNamespace()
        self.par.J = 3
        self.par.N = 10
        self.par.K = 10000

        self.par.F = np.arange(1, self.par.N + 1)
        self.par.sigma = 2

        self.par.v = np.array([1, 2, 3])
        self.par.c = 1

        # Create a matrix of friends in each career for each individual
        self.F = np.tile(np.arange(1, self.par.N + 1).reshape(self.par.N, 1), (1, self.par.J))

        # Create friends_in_career for each individual
        self.friends_in_career = [np.full(self.par.J, i + 1) for i in range(self.par.N)]

    def eps_sim(self, mu, sigma, shape):
        par = self.par
        """
        Args:
        mu (float): mean of the normal distribution
        sigma (float): standard deviation of the normal distribution
        shape (tuple): shape of the output array

        Returns:
        np.array: array of random numbers drawn from a normal distribution
    
        """
        return np.random.normal(mu, sigma, shape)

    def avg_exp_util(self):
        
        """
        Args:
        None

        Returns:
        np.array: average expected utility for each individual
        
        """
        np.random.seed(2024)
        par = self.par
        u = par.v + (1/par.K) * np.sum(self.eps_sim(0, par.sigma, (par.J, par.K)), axis=1)
        return u
    
    def avg_realised_util(self):
        """
        
        Args:
        None

        Returns:
        np.array: average realised utility for each individual

        """
        par = self.par
        np.random.seed(2024)
        epsilon = self.eps_sim(0, par.sigma, (par.J, par.K))
        u_realised = par.v[:, None] + epsilon
        avg_u_realised = np.mean(u_realised, axis=1)
        return avg_u_realised

    def algorithm_friends(self):
        """
        
        Args:
        None

        Returns:
        np.array: career shares for each individual
        np.array: average subjective utility for each individual
        np.array: average realised utility for each individual
        
        """
        par = self.par
        # Create a matrix of friends in each career for each individual
        self.F = np.tile(np.arange(1, par.N + 1).reshape(par.N, 1), (1, par.J))

        # Create friends_in_career for each individual
        self.friends_in_career = [np.full(par.J, i + 1) for i in range(par.N)]
        
        # draw epsilons
        # a. initiate empty arrays
        eps_friends = np.zeros((par.N, par.J, par.F[-1], par.K))
        
        # b. draw J*F epsilon values for each graduate friends
        for i in range(par.N):
            Fi = self.friends_in_career[i][0]  # Number of friends in each career for individual i
            eps_friends[i, :, :Fi, :] = self.eps_sim(0, par.sigma, (par.J, Fi, par.K))

        # c. draw for individual him/herself
        eps_individual = self.eps_sim(0, par.sigma, (par.N, par.J, par.K))

        # calculate averages
        # a. initiate empty arrays
        career = np.zeros((par.N, par.K), dtype=int)
        exp_u = np.zeros((par.N, par.K))
        actual_u = np.zeros((par.N, par.K))

        # b. simulate career choices and expected and actual utility
        for k in range(par.K):
            for i in range(par.N):
                # i. set number of friends in each career for individual i
                Fi = self.friends_in_career[i][0]  
                # ii. calculate expected utility
                prior = np.zeros(par.J)
                for j in range(par.J):
                    prior[j] = par.v[j] + np.mean(eps_friends[i, j, :Fi, k])
                # iii. choose career that maximizes expected utility 
                chosen_career = np.argmax(prior)
                # iv. store career choice and expected and actual utility
                career[i, k] = chosen_career
                exp_u[i, k] = prior[chosen_career]
                actual_u[i, k] = par.v[chosen_career] + eps_individual[i, chosen_career, k]
        
        # Initiate empty arrays
        career_shares = np.zeros((par.N, par.J))
        avg_subjective_utilities = np.zeros(par.N)
        avg_realised_utilities = np.zeros(par.N)

        # Calculate career shares and average subjective and realised utilities
        # i. over each graduate
        for i in range(par.N):
            # ii. over each career
            for j in range(par.J):
                # iii. calculate career shares
                career_shares[i, j] = np.mean(career[i, :] == j)
            # iv. calculate average subjective/realised utilities for each graduate
            avg_subjective_utilities[i] = np.mean(exp_u[i, :])
            avg_realised_utilities[i] = np.mean(actual_u[i, :])

        return career_shares, avg_subjective_utilities, avg_realised_utilities
    
    def plot_share(self, career_shares):
        par = self.par
        fig, ax = plt.subplots()
        for j in range(par.J):
            ax.plot(np.arange(1, par.N + 1), career_shares[:, j], label=f"Career {j+1}")
        ax.set_xlabel("Individuals")
        ax.set_ylabel("Career share")
        ax.legend()
        plt.show()
    
    def plot_exp_util(self, avg_subjective_utilities):
        par = self.par
        fig, ax = plt.subplots()
        ax.plot(np.arange(1, par.N + 1), avg_subjective_utilities, marker='o', label="Average Subjective Utility")
        ax.set_xlabel("Individuals")
        ax.set_ylabel("Average expected utility")
        plt.show()

    def plot_realised_util(self, avg_realised_utilities):
        par = self.par
        fig, ax = plt.subplots()
        ax.plot(np.arange(1, par.N + 1), avg_realised_utilities, marker='o', label="Average Realised Utility")
        ax.set_xlabel("Individuals")
        ax.set_ylabel("Average realised utility")
        plt.show()

    def modified_utility(self, u, c, chosen_career, j):
        """Calculate the modified utility"""
        if j == chosen_career:
            return u
        else:
            return u - c

    def new_prior(self):
        """
        
        Args:
        None

        Returns:
        np.array: new career choices for each individual
        np.array: average new expected utility for each individual
        np.array: average new actual utility for each individual
        np.array: switch share for each individual

        """
        #set parameters
        par = self.par

        # Repeat first-year simulation of initial career choices and utilities
        # a. initiate empty arrays
        initial_career = np.zeros((par.N, par.K), dtype=int)
        initial_exp_u = np.zeros((par.N, par.K))
        initial_actual_u = np.zeros((par.N, par.K))
        
        # b. draw epsilons for friends and individual
        eps_friends = np.zeros((par.N, par.J, par.F[-1], par.K))
        for i in range(par.N):
            Fi = self.friends_in_career[i][0]
            eps_friends[i, :, :Fi, :] = self.eps_sim(0, par.sigma, (par.J, Fi, par.K))
        eps_individual = self.eps_sim(0, par.sigma, (par.N, par.J, par.K))

        # c. simulate career choices and expected and actual utility
        for k in range(par.K):
            # i. for each graduate
            for i in range(par.N):
                # ii. set number of friends in each career for individual i
                Fi = self.friends_in_career[i][0]  
                prior = np.zeros(par.J)
                # iii. for each career, calculate expected utility
                for j in range(par.J):
                    prior[j] = par.v[j] + np.mean(eps_friends[i, j, :Fi, k])
                # iv. choose career that maximizes expected utility
                chosen_career = np.argmax(prior)
                initial_career[i, k] = chosen_career
                initial_exp_u[i, k] = prior[chosen_career]
                initial_actual_u[i, k] = par.v[chosen_career] + eps_individual[i, chosen_career, k]

        # add in new expected and realised utility functions where individuals know the utility
        # of their current career and the expected utility of other careers

        # a. initiate empty arrays
        new_career = np.zeros((par.N, par.K), dtype=int)
        new_exp_u = np.zeros((par.N, par.K))
        new_actual_u = np.zeros((par.N, par.K))
        switch_share = np.zeros(par.N)

        # b. simulate
        for k in range(par.K):
            # i. for each graduate
            for i in range(par.N):
                prior = np.zeros(par.J)
                # ii. if career is the same, individual knows the actual utility
                for j in range(par.J):
                    if j == initial_career[i, k]:
                        prior[j] = par.v[j]+ eps_individual[i, j, k]
                    # iii. if career is not same, individual knows the expected utiliy based on friends - c
                    else:
                        prior[j] = par.v[j] + np.mean(eps_friends[i, j, :Fi, k]) - par.c
                # iv. choose career that maximizes expected utility
                chosen_career = np.argmax(prior)

                new_career[i, k] = chosen_career
                new_exp_u[i, k] = prior[chosen_career]
                new_actual_u[i, k] = self.modified_utility(initial_actual_u[i, k], par.c, initial_career[i, k], chosen_career)
                
                # v. calculate share of individuals who switch career
                if chosen_career != initial_career[i, k]:
                    switch_share[i] += 1
        
        avg_new_exp_u = np.mean(new_exp_u, axis=1)
        avg_new_actual_u = np.mean(new_actual_u, axis=1)
        switch_share /= par.K

        return new_career, avg_new_exp_u, avg_new_actual_u, switch_share

    def plot_switch_share(self, switch_share):
        par = self.par
        fig, ax = plt.subplots()
        ax.plot(np.arange(1, par.N + 1), switch_share, marker='o', label="Switch share")
        ax.set_xlabel("Individuals")
        ax.set_ylabel("Switch share")
        plt.show()
    

    

    


  

    



    