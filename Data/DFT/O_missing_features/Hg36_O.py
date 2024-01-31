
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


db = connect('C_missing_features.db')

ads_O = Atoms('O',positions=[(0, 0, 0)])

clean_metals = read('Metals_large.db@ads=Clean')

slab = clean_metals[18]

fixed = [i for i in range(0,33,4)] + [i for i in range(1,34,4)]
c = FixAtoms(fixed)
slab.set_constraint(c)

pos = np.array(slab.get_positions()[19])
ads_dist = slab.get_distance(19,23) * math.sqrt(3) / 3
ads_pos = [(slab.get_positions()[19][0], slab.get_positions()[19][1] + ads_dist, slab.get_positions()[19][2] + ads_dist/2)]
ads_O.set_positions(ads_pos)

combined = slab + ads_O

calc = GPAW(xc='BEEF-vdW',
    h=0.18,
    kpts={'size': (4, 4, 1)},
    basis='dzp')

combined.calc = calc
dyn = BFGS(combined, trajectory=f'{slab.symbols}_O.traj')
dyn.run(fmax=0.05)

combined.get_potential_energy()

db.write(combined)
        