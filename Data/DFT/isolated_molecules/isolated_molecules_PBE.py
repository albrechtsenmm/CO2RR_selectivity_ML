from ase import Atoms
import numpy as np
from ase.visualize import view
from ase.build import molecule
from ase.collections import g2
from gpaw import GPAW, PW
from ase.optimize import BFGS
from ase.calculators.emt import EMT
from ase.db import connect


db = connect('isolated_molecules_PBE.db')

# Defining the molecules
#print(g2.names) # Names of the available molecules

# Estimated bond lengths (All in Ångstroms)
C_C = 1.54
O_H = 0.96
C_O = 1.43
O_C_O = 1.17
Cu_C = 2 #roughly
H_H = 0.74


a = 20
b = a / 2

CO2 = Atoms('OCO',
              positions=[(b,b,b),
                         (O_C_O + b, b, b),
                         (2 * O_C_O + b, b, b)],
                cell=[a, a, a])

H2 = Atoms('H2',
              positions=[(b,b,b),
                         (H_H + b, b, b)],
            cell=[a, a, a])

H2O = Atoms('HOH',
              positions=[(-(np.cos((38*np.pi)/180)*O_H) + b ,b,np.sin((38*np.pi)/180)*O_H + b),
                         (b, b, b),
                         (np.cos((38*np.pi)/180)*O_H + b, b, np.sin((38*np.pi)/180)*O_H + b)],
                cell=[a, a, a])

CO = Atoms('CO',
              positions=[(b ,b ,b),
                         (C_O + b, b, b)],
                cell=[a, a, a])

molecules = [CO2, H2, H2O, CO]

calc1 = GPAW(xc='PBE',
            h=0.18,
            basis='dzp',
            convergence={'eigenstates': 1e-6})



for molecule in molecules:

    molecule.calc = calc1
    dyn = BFGS(molecule, trajectory=f'{molecule.symbols}_isolated_PBE.traj')
    dyn.run(fmax=0.05)

    molecule.get_potential_energy()

    db.write(molecule, relaxed=True)
