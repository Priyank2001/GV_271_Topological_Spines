"""
alanine_dipeptide_md.py: the simulation behind alanine_dipeptide_md.npz (for add_alanine_dipeptide.py).

    pip install openmm
    python3 alanine_dipeptide_md.py 30        # 30 ns, about 40 minutes on 2 CPU cores

Alanine dipeptide, implicit solvent (amber14 + OBC2), 300 K, Langevin 1/ps,
HBond constraints, hydrogen mass 4 amu, 4 fs steps. Saves heavy-atom positions (nm) and
backbone phi/psi (degrees) every 2 ps, in 5 ns chunks."""
import sys, time, numpy as np, openmm as mm, openmm.app as app, openmm.unit as u
NS, SAVE_PS, SEED = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0, 2.0, int(sys.argv[2]) if len(sys.argv) > 2 else 1
import io
PDB = '''REMARK  ACE                                                         
ATOM      1 1HH3 ACE     1       2.000   1.000  -0.000
ATOM      2  CH3 ACE     1       2.000   2.090   0.000
ATOM      3 2HH3 ACE     1       1.486   2.454   0.890
ATOM      4 3HH3 ACE     1       1.486   2.454  -0.890
ATOM      5  C   ACE     1       3.427   2.641  -0.000
ATOM      6  O   ACE     1       4.391   1.877  -0.000
ATOM      7  N   ALA     2       3.555   3.970  -0.000
ATOM      8  H   ALA     2       2.733   4.556  -0.000
ATOM      9  CA  ALA     2       4.853   4.614  -0.000
ATOM     10  HA  ALA     2       5.408   4.316   0.890
ATOM     11  CB  ALA     2       5.661   4.221  -1.232
ATOM     12 1HB  ALA     2       5.123   4.521  -2.131
ATOM     13 2HB  ALA     2       6.630   4.719  -1.206
ATOM     14 3HB  ALA     2       5.809   3.141  -1.241
ATOM     15  C   ALA     2       4.713   6.129   0.000
ATOM     16  O   ALA     2       3.601   6.653   0.000
ATOM     17  N   NME     3       5.846   6.835   0.000
ATOM     18  H   NME     3       6.737   6.359  -0.000
ATOM     19  CH3 NME     3       5.846   8.284   0.000
ATOM     20 1HH3 NME     3       4.819   8.648   0.000
ATOM     21 2HH3 NME     3       6.360   8.648   0.890
ATOM     22 3HH3 NME     3       6.360   8.648  -0.890
TER   
END   
'''
pdb = app.PDBFile(io.StringIO(PDB))
ff = app.ForceField('amber14-all.xml', 'implicit/obc2.xml')
system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds,
                         hydrogenMass=4 * u.amu)
integ = mm.LangevinMiddleIntegrator(300 * u.kelvin, 1 / u.picosecond, 4 * u.femtoseconds)
integ.setRandomNumberSeed(SEED)
sim = app.Simulation(pdb.topology, system, integ, mm.Platform.getPlatformByName('CPU'))
sim.context.setPositions(pdb.positions); sim.minimizeEnergy()
sim.context.setVelocitiesToTemperature(300 * u.kelvin, SEED)
sim.step(25000)                                     # 100 ps equilibration
atoms = list(pdb.topology.atoms())
heavy = [a.index for a in atoms if a.element.symbol != 'H']
name = {(a.residue.name, a.name): a.index for a in atoms}
C0, N1, CA, C1, N2 = name[('ACE', 'C')], name[('ALA', 'N')], name[('ALA', 'CA')], name[('ALA', 'C')], name[('NME', 'N')]
def dih(p, a, b, c, d):
    b0, b1, b2 = p[a] - p[b], p[c] - p[b], p[d] - p[c]
    b1 /= np.linalg.norm(b1)
    v = b0 - (b0 @ b1) * b1; w = b2 - (b2 @ b1) * b1
    return np.degrees(np.arctan2(np.cross(b1, v) @ w, v @ w))
steps = int(SAVE_PS / 0.004); per_chunk = int(5000 / SAVE_PS); total = int(NS * 1000 / SAVE_PS)
X, PHI, PSI, k, t0 = [], [], [], 0, time.time()
for i in range(total):
    sim.step(steps)
    p = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(u.nanometer)
    X.append(p[heavy].astype(np.float32)); PHI.append(dih(p, C0, N1, CA, C1)); PSI.append(dih(p, N1, CA, C1, N2))
    if len(X) == per_chunk or i == total - 1:
        np.savez(f'chunk_s{SEED}_{k:03d}.npz', pos=np.array(X), phi=np.array(PHI), psi=np.array(PSI),
                 heavy=np.array([atoms[j].residue.name + ':' + atoms[j].name for j in heavy]))
        print(f'chunk {k} done, {(i + 1) * SAVE_PS / 1000:.1f} ns, {time.time() - t0:.0f} s', flush=True)
        X, PHI, PSI, k = [], [], [], k + 1

# one file for add_alanine_dipeptide.py
import glob
parts = [np.load(f) for f in sorted(glob.glob(f'chunk_s{SEED}_*.npz'))]
np.savez_compressed('alanine_dipeptide_md.npz', pos=np.concatenate([p['pos'] for p in parts]),
                    phi=np.concatenate([p['phi'] for p in parts]), psi=np.concatenate([p['psi'] for p in parts]),
                    heavy=parts[0]['heavy'])
print('wrote alanine_dipeptide_md.npz')
