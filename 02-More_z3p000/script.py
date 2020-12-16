# This script will be executed on the server
# Importing things
import numpy as np
import pandas as pd
from nbodykit.source.catalog import Gadget1Catalog, BinaryCatalog
from nbodykit.algorithms.paircount_tpcf.tpcf import SimulationBox2PCF
from nbodykit.lab import UniformCatalog
import matplotlib.pyplot as plt
from nbodykit.lab import *
from nbodykit import setup_logging, style

plt.style.use(style.notebook)

# Graph function
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
        ax.scatter(x[corte[0]:corte[1]],y[corte[0]:corte[1]], marker='.', c=color[0])
    ax.plot(x[corte[0]:corte[1]], y[corte[0]:corte[1]], color=color[1]) # Graficamos
    ax.set_xlabel(labels[0]) # Asignamos nombres a los labels
    ax.set_ylabel(labels[1]) # Asignamos nombres a los labels
    if limite != []:
        plt.ylim(top = max(limite), bottom= min(limite))
    ax.grid() # Con cuadrícula
    fig.savefig(f"/results/{titulo}.png")

# Reaging data
path="./data/simulation_100_z3p000.*"
cat = Gadget1Catalog(path,
    columndefs=[('Position', ('auto', 3), 'all'),
    ('GadgetVelocity', ('auto', 3), 'all'),('Mass', 'auto', None)],ptype=1)
cat.attrs["BoxSize"] = np.array([512., 512., 512.]) * 2

# Checking the data
print(cat)
print(cat.columns)
print(cat.csize)

# Converting to a mesh
mesh = cat.to_mesh(Nmesh=1024, resampler = "tsc", compensated=True)

# Computing the Power Spectrum
r = FFTPower(mesh, mode='1d')

# the result is stored at "power" attribute
Pk = r.power
print()
print("Power Spectrum Done")
print(Pk)

# Saving Results
Pk.to_json("./results/PowerSpectrum.json")


# Correlation Function
L = 256
nbins = 30
bins = np.array([(Bin+1)*L/nbins for Bin in range(nbins+1)])
corr = SimulationBox2PCF(mode="1d", data1=cat, show_progress=True, edges=bins)

print()
print("Correlation function done")

# Saving results
corr.save("./results/CorrelationFunction.json")

print("Listo kalisto")

