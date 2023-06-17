import numpy as np
import matplotlib.pyplot as plt

# Define the details of the discrete optimization problem
nVar = 20
ub = np.ones(nVar)
lb = np.zeros(nVar)

# Define the objective function
def ObjectiveFunction(x):
    # Implement your objective function here
    return sum(x**2)

# Define the PSO's parameters
noP = 30
maxIter = 1000
wMax = 0.9
wMin = 0.2
c1 = 2
c2 = 2
vMax = (ub - lb) * 0.2
vMin = -vMax

# The PSO algorithm

# Define the Particle class
class Particle:
    def __init__(self):
        self.X = np.round((ub - lb) * np.random.rand(nVar) + lb)
        self.V = np.zeros(nVar)
        self.PBEST_X = np.zeros(nVar)
        self.PBEST_O = np.inf

Swarm = []
GBEST_X = np.zeros(nVar)
GBEST_O = np.inf

# Initialize the particles
for _ in range(noP):
    particle = Particle()
    Swarm.append(particle)
    if particle.PBEST_O < GBEST_O:
        GBEST_X = particle.X
        GBEST_O = particle.PBEST_O

# Main loop
cgCurve = np.zeros(maxIter)
for t in range(maxIter):

    # Calculate the objective value
    for k in range(noP):
        currentX = Swarm[k].X
        Swarm[k].O = ObjectiveFunction(currentX)

        # Update the PBEST
        if Swarm[k].O < Swarm[k].PBEST_O:
            Swarm[k].PBEST_X = currentX
            Swarm[k].PBEST_O = Swarm[k].O

        # Update the GBEST
        if Swarm[k].O < GBEST_O:
            GBEST_X = currentX
            GBEST_O = Swarm[k].O

    # Update the X and V vectors
    w = wMax - t * ((wMax - wMin) / maxIter)

    for k in range(noP):
        Swarm[k].V = w * Swarm[k].V + c1 * np.random.rand(nVar) * (Swarm[k].PBEST_X - Swarm[k].X) \
                                        + c2 * np.random.rand(nVar) * (GBEST_X - Swarm[k].X)

        # Check velocities
        index1 = np.where(Swarm[k].V > vMax)
        index2 = np.where(Swarm[k].V < vMin)

        Swarm[k].V[index1] = vMax[index1]
        Swarm[k].V[index2] = vMin[index2]

        # Sigmoid transfer function
        s = 1 / (1 + np.exp(-Swarm[k].V))

        # Update the position of k-th particle
        for d in range(nVar):
            r = np.random.rand()
            if r < s[d]:
                Swarm[k].X[d] = 1
            else:
                Swarm[k].X[d] = 0

    outmsg = 'Iteration# {} Swarm.GBEST.O = {}'.format(t, GBEST_O)
    print(outmsg)

    cgCurve[t] = GBEST_O

plt.semilogy(cgCurve)
plt.xlabel('Iteration#')
plt.ylabel('Weight')
plt.show()