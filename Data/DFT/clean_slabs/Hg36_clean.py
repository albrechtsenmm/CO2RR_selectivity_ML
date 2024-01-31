
from ase import Atoms
from ase.optimize import BFGS
import numpy as np
from ase.visualize import view
from ase.io import read
from ase.constraints import FixAtoms
from ase.db import connect
import math
from gpaw import GPAW
from ase.calculators.emt import EMT

db = connect('clean_isolated.db')

clean_metals = read('Metals_large.db@ads=Clean')

slab = clean_metals[18]

fixed = [i for i in range(0,33,4)] + [i for i in range(1,34,4)]
c = FixAtoms(fixed)
slab.set_constraint(c)

calc = GPAW(xc='BEEF-vdW',
    h=0.18,
    kpts={'size': (4, 4, 1)},
    basis='dzp',
    maxiter=3000)

slab.calc = calc
dyn = BFGS(slab, trajectory=f'{slab.symbols}_clean.traj')
dyn.run(fmax=0.05)

slab.get_potential_energy()

db.write(slab)
        