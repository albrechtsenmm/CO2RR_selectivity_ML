from ase import Atoms
from ase.optimize import BFGS
from ase.calculators.emt import EMT
import numpy as np
from ase.visualize import view
from ase.io import read
from ase.constraints import FixAtoms, FixBondLength
from gpaw import GPAW, PW
from ase.dft.bee import BEEFEnsemble



C_C = 1.54
O_H = 0.96
C_O = 1.43
Cu_C = 2 #roughly

Cu111=read('Metals_large.db@Metal=%s,ads=Clean' %'Cu')[0]

fixed = [i for i in range(0,33,4)] + [i for i in range(1,34,4)]
c = FixAtoms(fixed)
#Cu111.set_constraint(c)

pos = Cu111.get_positions()[19]
pos[2] += Cu_C



OCCOH = Atoms('HOCCO',
              positions=[(pos[0], pos[1] - (np.sqrt(2) * C_O)/2 + (np.sqrt(2) * O_H)/2 - C_C, pos[2] + (np.sqrt(2) * C_O)/2 + (np.sqrt(2) * O_H)/2),
                          (pos[0], pos[1] -(np.sqrt(2) * C_O)/2 - C_C, pos[2] + (np.sqrt(2) * C_O)/2),
                         (pos[0], pos[1] -C_C, pos[2]),
                         (pos[0], pos[1], pos[2]),
                         (pos[0], pos[1] + (np.sqrt(2) * C_O)/2, pos[2] + (np.sqrt(2) * C_O)/2)])
                         

b1 = FixBondLength(36, 37)
b2 = FixBondLength(37, 38)
b3 = FixBondLength(38, 39)
b4 = FixBondLength(39, 40)

combined = Cu111 + OCCOH

combined.set_constraint([c, b1, b2, b3, b4])

calc = GPAW(xc='BEEF-vdW',
            h=0.18,
            kpts={'size': (4, 4, 1), 'gamma': True},
            basis='dzp')

combined.calc = calc
dyn = BFGS(combined, trajectory='OCCOH_all_fixedbondlength_correctHplacement.traj')
dyn.run(fmax=0.05)

#print(combined.get_positions())

