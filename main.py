import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random
import operator
import copy

plt.ion()

# Problems
class Sphere:
  def __init__(self, D):
    self.D = D
  def evaluate(self, x):
    return (x**2).sum()

class Rastrigin:
  def __init__(self, D):
    self.D = D
  def evaluate(self, x):
    return 10 * self.D + sum(x**2 - (10 * np.cos(2 * np.pi * x)))

class ZDT1:
  def __init__(self, D=30):
    self.D = D
  def evaluate(self, x):
    f1 = x[0]
    g = 1.0 + (9/(self.D-1)) * x[1:].sum()
    h = 1.0 - np.sqrt(f1 / g)
    f2 = g * h
    return np.array([f1, f2])

# Mutations
class AdditiveGaussianMutation:
  def __init__(self, std=0.1):
    self.std = std
  def mutate(self, x):
    xp = x.copy()
    idx = np.random.randint(xp.shape[0])
    xp[idx] = np.inf
    while xp[idx] < 0 or xp[idx] > 1:
      z = np.random.randn() * self.std
      xp[idx] = x[idx] + z
    return xp

class Crossover:
  def __init__(self):
    pass
  def cross(self, parent1, parent2):
    D = parent1.shape[0]
    r = np.random.randint(D)
    soln = np.concatenate((parent1[:r], parent2[r:]))
    return soln

# Optimisers
class HillClimber:
  def __init__(self, mutation, D):
    self.mutation = mutation
    self.D = D
  def optimise(self, iterations, problem, popsize):
    x = np.random.uniform(-3, 3, self.D)
    y = problem.evaluate(x)
    history=[]
    for itr in range(iterations):
      xp = self.mutation.mutate(x)
      yp = problem.evaluate(xp)
      if yp <= y:
        x = xp
        y = yp
      history.append((x, y))
    return history

class GA:
  def __init__(self, crossover, mutation):
    self.crossover = crossover
    self.mutation = mutation
  def optimise(self, fevals, problem, popsize):
    x = [np.random.uniform(-3, 3, problem.D) for i in range(popsize)]
    y = [problem.evaluate(x[i]) for i in range(popsize)]
    history=[]
    history.append((x, y))
    for gen in range((fevals-popsize)//popsize):
      cx = []
      cy = []
      for i in range(popsize):
        cx.append(self.evolve(x))
        cy.append(problem.evaluate(cx[-1]))
      combX = x + cx
      combY = y + cy
      I = np.argsort(combY)
      x = [combX[i] for i in I[:popsize]]
      y = [combY[i] for i in I[:popsize]]
      history.append((x, y))
    return history
  def evolve(self, population):
    i, j = np.random.randint(0, population[0].shape[0], 2)
    child = self.crossover.cross(population[i], population[j])
    child = self.mutation.mutate(child)
    return child

class MOES:
  def __init__(self, mutation):
    self.mutation = mutation
  def optimise(self, problem, niter):
    x = np.random.rand(problem.D)
    y = problem.evaluate(x)
    archive = Archive()
    archive.update(y)
    for i in range(niter):
      xp = self.mutation.mutate(x)
      yp = problem.evaluate(xp)
      archive.update(yp)
      if not dominates(y, yp):
        x = xp
        y = yp
    return archive

# Checker
def dominates(u, v): 
  return (u<=v).all() and (u<v).any()

class Archive:
  def __init__(self):
    self.objective_vectors = []
  def update(self, y):
    to_remove = []
    for i in range(len(self.objective_vectors)):
      if dominates(self.objective_vectors[i], y):
        return
      if dominates(y, self.objective_vectors[i]):
        to_remove.append(i)
    self.objective_vectors.append(y)
    to_remove = sorted(to_remove, reverse=True)
    for i in to_remove:
      self.objective_vectors.pop(i)

# Repeater
def repeat_expt(problem, optimiser, Nrepeats=20, Niter=100, popsize=1, colour='k', title="Figure"):
    H = np.zeros((Nrepeats, Niter))
    
    for i in range(Nrepeats):
        history = optimiser.optimise(Niter*popsize, problem, popsize)
        H[i,:] = [np.mean(history[n][1]) for n in range(len(history))]
        
    plt.figure()
    plt.plot(np.arange(Niter, dtype=int)+1, H.mean(axis=0), c=colour)
    plt.plot(np.arange(Niter, dtype=int)+1, H.min(axis=0), c=colour, ls="--")
    plt.plot(np.arange(Niter, dtype=int)+1, H.max(axis=0), c=colour, ls="--")
    plt.xlabel("Iteration")
    plt.ylabel("Objective value")
    plt.title(title)
    plt.show()

    return H


D = 30
N = 3000
problem = ZDT1()
optimiser = MOES(AdditiveGaussianMutation())
archive = optimiser.optimise(problem, N)

Y = np.array([archive.objective_vectors[i] for i in range(len(archive.objective_vectors))])
plt.scatter(Y[:,0], Y[:,1])
plt.show()