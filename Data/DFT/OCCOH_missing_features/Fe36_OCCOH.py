
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

db = connect('OCCOH_missing_features.db')

clean_metals = read('Metals_large.db@ads=Clean')

Cu111=read('Metals_large.db@Metal=%s,ads=Clean' %'Cu')[0]
Cu_pos = Cu111.get_positions()[19]

C_C = 1.54
O_H = 0.96
C_O = 1.43

OCCOH = Atoms('HOCCO',
              positions=[(0, 0 - (np.sqrt(2) * C_O)/2 + (np.sqrt(2) * O_H)/2 - C_C, 0 + (np.sqrt(2) * C_O)/2 + (np.sqrt(2) * O_H)/2),
                          (0, 0 -(np.sqrt(2) * C_O)/2 - C_C, 0 + (np.sqrt(2) * C_O)/2),
                         (0, 0 -C_C, 0),
                         (0, 0, 0),
                         (0, 0 + (np.sqrt(2) * C_O)/2, 0 + (np.sqrt(2) * C_O)/2)])

slab = clean_metals[11]

fixed = [i for i in range(0,33,4)] + [i for i in range(1,34,4)]
c = FixAtoms(fixed)
slab.set_constraint(c)

slab_pos = slab.get_positions()[19]

OCCOH.set_positions([(1.367702 - Cu_pos[0] + slab_pos[0], -0.305658 - Cu_pos[1] + slab_pos[1], 19.732750 - Cu_pos[2] + slab_pos[2]),
                    (1.345185 - Cu_pos[0] + slab_pos[0], -0.671294 - Cu_pos[1] + slab_pos[1], 18.940677 - Cu_pos[2] + slab_pos[2]),
                    (1.348938 - Cu_pos[0] + slab_pos[0], 0.195161 - Cu_pos[1] + slab_pos[1], 17.941584 - Cu_pos[2] + slab_pos[2]),
                    (1.275624 - Cu_pos[0] + slab_pos[0], 1.691129 - Cu_pos[1] + slab_pos[1], 18.183691 - Cu_pos[2] + slab_pos[2]),
                    (1.328177 - Cu_pos[0] + slab_pos[0], 2.319677 - Cu_pos[1] + slab_pos[1], 19.467071 - Cu_pos[2] + slab_pos[2])])

combined = slab + OCCOH

calc = GPAW(xc='BEEF-vdW',
    h=0.18,
    kpts={'size': (4, 4, 1)},
    basis='dzp')

combined.calc = calc
dyn = BFGS(combined, trajectory=f'{slab.symbols}_OCCOH.traj')
dyn.run(fmax=0.05)

combined.get_potential_energy()

db.write(combined)
        