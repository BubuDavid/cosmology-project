import numpy as np
import pandas as pd
from nbodykit.source.catalog import CSVCatalog, BinaryCatalog
from nbodykit.algorithms.paircount_tpcf.tpcf import SimulationBox2PCF
import matplotlib.pyplot as plt

from nbodykit.lab import *
from nbodykit import setup_logging, style
from nbodykit.binned_statistic import BinnedStatistic
plt.style.use(style.notebook)

def graphFunction(x,y, color=["purple","blue"], labels=["x","y"], titulo="", scatter=False, corte=[0,-1], limite=[]):
    '''
    Descripción: Función que grafica y con respecto a x
    IN = {
        x: Variable independiente.
        y: Variable dependiente.
        color: El color de la gráfica.
        labels: El nombre de las variables.
        titulo: El título
        scatter: Si es que se quieren resaltar los puntos
    }
    Out: Una bonita gráfica x-y
    '''
    # Creamos el objeto figura y nuestro ax.
    fig, ax = plt.subplots(figsize = (5, 3))
    # ponemos título
    ax.set_title(titulo)
    # Graficamos
    if scatter:
        ax.scatter(x[corte[0]:corte[1]],y[corte[0]:corte[1]], marker='.', c=color[1])
    ax.plot(x[corte[0]:corte[1]], y[corte[0]:corte[1]], color=color[0]) # Graficamos
    ax.set_xlabel(labels[0]) # Asignamos nombres a los labels
    ax.set_ylabel(labels[1]) # Asignamos nombres a los labels
    if limite != []:
        plt.ylim(top = max(limite), bottom= min(limite))
    ax.grid() # Con cuadrícula
    fig.savefig(f"./results/{titulo}.png")


# Reading the data
columns = ["x", "y", "z", "vx", "vy", "vz"]
# read the data
f = CSVCatalog('./data/example_128_z0p000.*', columns)

# combine x, y, z to Position
f['Position'] = f['x'][:, None] * [1, 0, 0] + f['y'][:, None] * [0, 1, 0] + f['z'][:, None] * [0, 0, 1]
f.attrs["BoxSize"] = [512., 512., 512.]

# Correlation function
L = 160
nbins = 15
bins = np.array([(Bin+1)*L/nbins for Bin in range(nbins+1)])
corr = SimulationBox2PCF(mode="1d", data1=f, show_progress=True, edges= bins, BoxSize=512.)

# Saving results
corr.save("./results/CorrelationFuncionOfExample_128_z.json")

graphFunction(corr.corr["r"], corr.corr["corr"], labels=["r", "ξ(r)"], scatter=True, titulo="correlation_function")


# Power Spectrum
# Converting to a mesh
mesh = f.to_mesh(window='tsc', Nmesh=512, compensated=True)

# Visuailzation
plt.imshow(mesh.preview(axes=[0,2]))
plt.savefig("./results/mesh.png")

# Computing the Power Spectrum
r = FFTPower(mesh, mode='1d')

# the result is stored at "power" attribute
Pk = r.power
print(Pk)

# Definition of things
k = Pk["k"][1:]
power = Pk["power"][1:]

# Saving Results
Pk.to_json("./results/PowerSpectrumOfExample_128_z.json")

# Ploting the Pecrum
fig, ax = plt.subplots(figsize = (7, 4))

plt.loglog(k, power.real)

# format the axes
plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
plt.ylabel(r"$P(k)$ [$h^{-3}\mathrm{Mpc}^3$]")

plt.savefig("./results/spectrum.png")