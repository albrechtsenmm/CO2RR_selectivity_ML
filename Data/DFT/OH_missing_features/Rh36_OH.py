
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


db = connect('OH_missing_features.db')

OH = Atoms('OH',positions=[(0, 0, 0), (0, 0, 0.96)])

clean_metals = read('Metals_large.db@ads=Clean')

slab = clean_metals[0]

fixed = [i for i in range(0,33,4)] + [i for i in range(1,34,4)]
c = FixAtoms(fixed)
slab.set_constraint(c)

pos = np.array(slab.get_positions()[19])
pos[2] += 2
OH.set_positions([(pos[0], pos[1], pos[2]), (pos[0], pos[1], pos[2] + 0.96)])

combined = slab + OH

calc = GPAW(xc='BEEF-vdW',
    h=0.18,
    kpts={'size': (4, 4, 1)},
    basis='dzp')

combined.calc = calc

dyn = BFGS(combined, trajectory=f'{slab.symbols}_OH.traj')
dyn.run(fmax=0.05)

combined.get_potential_energy()

db.write(combined)
        