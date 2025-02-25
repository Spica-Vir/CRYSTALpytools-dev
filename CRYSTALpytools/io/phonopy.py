#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes and methods to parse files used by `Phonopy <https://phonopy.github.io/phonopy/>`_.
"""
def read_structure(file):
    """
    Read phonopy structure ('primitive cell') from 'band.yaml', 'phonopy.yaml'
    or 'phonopy_disp.yaml' files.

    Args:
        file (str): Phonopy yaml file

    Returns:
        struc (CStructure): Extended pymatgen structure
    """
    import yaml
    import numpy as np
    from CRYSTALpytools.units import au_to_angstrom
    from CRYSTALpytools.geometry import CStructure

    struc_file = open(file, 'r')
    data = yaml.safe_load(struc_file)
    struc_file.close()

    # Get unit
    try: # band.yaml
        len_unit = data['length_unit']
    except KeyError: # phonopy.yaml
        try:
            len_unit = data['physical_unit']['length']
        except KeyError:
            raise Exception("Unknown file format. Only 'band.yaml', 'phonopy.yaml' or 'phonopy_disp.yaml' are allowed.")

    if len_unit == 'angstrom':
        unit_len = 1.0
    elif len_unit == 'au':
        unit_len = au_to_angstrom(1.0)
    else:
        raise Exception("Unknown length unit. Available options: au, angstrom.")

    # Get structure
    spec = []
    coord = []
    try: # band.yaml
        latt = np.array(data['lattice'], dtype=float) * unit_len
        for idx_a, atom in enumerate(data['points']):
            spec.append(atom['symbol'])
            coord.append(atom['coordinates'])
    except KeyError: # phonopy.yaml
        latt = np.array(data['unit_cell']['lattice'], dtype=float) * unit_len
        for idx_a, atom in enumerate(data['unit_cell']['points']):
            spec.append(atom['symbol'])
            coord.append(atom['coordinates'])
    struc = CStructure(latt, spec, coord, )
    return struc


def read_frequency(file):
    """
    Read phonon frequency from `Phonopy <https://phonopy.github.io/phonopy/>`_
    band.yaml, mesh.yaml or qpoints.yaml files. Frequency units must be THz
    (default of Phonopy).

    Args:
        file (str): Phonopy yaml file
        q_id (list[int]): Specify the id (from 0) of q points to be read.
            nqpoint\*1 list.
        q_coord (list[list]): Specify the coordinates of q points to be
            read. nqpoint\*3 list.

    ``q_id`` and ``q_coord`` should not be set simultaneously. If set, ``q_id``
    takes priority and ``q_coord`` is ignored. If both are none, all the points
    will be read.

    .. note::

        Setting ``q_id`` or ``q_coord`` change their weights, i.e., the sum of
        their weights is renormalized to 1.

    Returns:
        qpoint (list): natom\*2 list. 1st element: 3\*1 array. Fractional
            coordinates of q points; 2nd element: float. Weight
        frequency (array): nqpint\*nmode array. Phonon frequency in THz.
    """
    import yaml
    import numpy as np
    import warnings

    phono_file = open(file, 'r', errors='ignore')
    data = yaml.safe_load(phono_file)
    phono_file.close()

    if np.all(q_id==None) and np.all(q_coord==None):
        nqpoint = data['nqpoint']
        qinfo = np.array(range(nqpoint), dtype=int)
    elif np.all(q_id!=None):
        qinfo = np.array(q_id, dtype=int)
        nqpoint = len(qinfo)
    elif np.all(q_id==None) and np.all(q_coord!=None):
        qinfo = np.array(q_coord, dtype=float)
        nqpoint = len(qinfo)

    natom = int(len(data['phonon'][0]['band']) / 3)
    qpoint = [[np.zeros([3, 1]), 0] for i in range(nqpoint)]
    frequency = np.zeros([nqpoint, 3 * natom])
    # Read phonon
    real_q = 0
    for idx_p, phonon in enumerate(data['phonon']):
        if real_q == nqpoint: break
        if len(qinfo.shape) == 1: # q_id and all q points
            if idx_p != qinfo[real_q]: continue
        else: # q_coord
            if np.linalg.norm(qinfo[real_q]-phonon['q-position']) > 1e-4:
                continue
        qpoint[real_q][0] = np.array(phonon['q-position'])
        try:
            qpoint[real_q][1] = phonon['weight'] # mesh
        except KeyError:
            qpoint[real_q][1] = 1 # qpoint / band
        frequency[real_q, :] = np.array([i['frequency'] for i in phonon['band']])
        real_q += 1

    if real_q < nqpoint: raise Exception('Some q points are missing from the yaml file.')
    # Normalize the weight
    tweight = np.sum([q[1] for q in qpoint])
    qpoint = [[q[0], q[1]/tweight] for q in qpoint]
    return qpoint, frequency


class PhonopyWriter():
    """
    Write the essientials into phonopy output. `Phonopy python API <https://phonopy.github.io/phonopy/phonopy-module.html>`_
    is used. The ``Phonopy`` instance is saved in ``self._phonopy`` attribute,
    with basic geometry and calculator information.

    .. note::

        Phonopy default units (:math:`\\AA`, AMU, THz, eV) are used for all
        calculators, which leads to no error as far as developers are aware of.

    Args:
        struc (Structure|CStructure): Pymatgen Structure class, unit cell.
        dim (list[int]): 1\*3 or 1\*9 list of supercell expansion matrix, i.e.,
            the ``--dim`` option of phonopy.
        calculator (str): Name of calculator. Will be used to determine the
            conversion factors.
        primitive (str|array): 9\*1 primitive matrix in phonopy convention, or
            'auto' to automatically identify the primitive cell.
    """
    def __init__(self, struc, dim, calculator='crystal', primitive='auto'):
        try:
            from pymatgen.core.structure import Structure
            from phonopy.structure.atoms import PhonopyAtoms
            from phonopy import Phonopy
            from phonopy import units
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Phonopy is required for this module.")
        import numpy as np
        import spglib

        if not isinstance(struc, Structure):
            raise TypeError("A pymatgen Structure or CRYSTALpytools CStructure object must be used.")

        freqfactor = {
            'vasp'     : 'units.VaspToTHz',
            'wien2k'   : 'units.Wien2kToTHz',
            'qe'       : 'units.PwscfToTHz',
            'abinit'   : 'units.AbinitToTHz',
            'siesta'   : 'units.SiestaToTHz',
            'elk'      : 'units.ElkToTHz',
            'crystal'  : 'units.CrystalToTHz',
            'turbomole': 'units.TurbomoleToTHz',
            'cp2k'     : 'units.CP2KToTHz',
            'fhi-aims' : 'units.VaspToTHz',
            'fleur'    : 'units.FleurToTHz',
            'castep'   : 'units.CastepToTHz',
            'abacus'   : 'units.AbinitToTHz',
            'lammps'   : 'units.VaspToTHz'
        }
        if calculator.lower() not in freqfactor.keys():
            raise ValueError("Calculator not supported: '{}'.".format(calculator))
        freqfac = eval(freqfactor[calculator.lower()])

        # supercell
        dim = np.array(dim, dtype=int, ndmin=1)
        s_matrix = np.zeros([3,3], dtype=int) # Phonopy convention
        if dim.shape[0] == 3:
            for i in range(3): s_matrix[i,i] = dim[i]
        elif dim.shape[0] == 9:
            for i in range(9): s_matrix[i//3, i%3] = dim[i]
        elif dim.shape[0] == 1:
            for i in range(3): s_matrix[i,i] = dim[0]
        else:
            raise ValueError('Dimensionality must have the length of 1, 3, or 9.')

        # primitive cell
        if isinstance(primitive, str) and primitive.lower() == 'auto':
            cell = (struc.lattice.matrix, struc.frac_coords, [i.Z for i in struc.species])
            p_cell, p_coords, p_species = spglib.find_primitive(cell)
            p_matrix = np.linalg.inv(struc.lattice.matrix.T) @ p_cell.T # Phonopy convention
        else:
            p_matrix = np.array(primitive, ndmin=1, dtype=float)
            if p_matrix.shape[0] != 9: raise ValueError('Primitive axes must be a 1*9 1D list.')
            p_matrix = p_matrix.reshape([3,3])

        atom = PhonopyAtoms(symbols=[i.symbol for i in struc.species],
                            cell=struc.lattice.matrix,
                            scaled_positions=struc.frac_coords)
        self._phonopy = Phonopy(atom,
                                supercell_matrix=s_matrix,
                                primitive_matrix=p_matrix,
                                factor=freqfac,
                                calculator=calculator.lower())
        self._phonopy._build_primitive_cell()

    def write_phonopy(self, filename='phonopy.yaml'):
        """
        Save computational setups and structure into 'phonopy.yaml'.

        Args:
            filename (str): The YAML file name.
        Returns:
            None
        """
        self._phonopy.save(filename=filename, settings={'force_constants': False})
        return

    def write_mesh(self, ha, filename='mesh.yaml', write_eigenvector=False):
        """
        Write frequency data over q points into 'mesh.yaml'. The mesh
        size is inferred from coordinates of qpoints, so it is important to use
        data obtained from the regular mesh grid.

        Args:
            ha (Harmonic): The ``thermodynamics.Harmonic`` object.
            filename (str): File name
            write_eigenvector (bool): *In developing* Whether to write
                eigenvector.
        Returns:
            None
        """
        import numpy as np
        import yaml
        from pymatgen.core.lattice import Lattice

        qcoords = np.array([i[0] for i in ha.qpoint], dtype=float)
        qweight = np.array([i[1] for i in ha.qpoint], dtype=float)
        # infer mesh size
        idx = np.where(qcoords!=0)
        if len(idx) == 0:
            mesh = [1, 1, 1]
        else:
            inrc = np.min(np.abs(qcoords[idx[0], idx[1], idx[2]]), axis=0)
            mesh = np.array(np.round(0.5/inrc), dtype=int).tolist()

        # structure
        lattmx = self._phonopy.unitcell.cell.tolist()
        rlattmx = Lattice(lattmx).reciprocal_lattice_crystallographic.matrix.tolist()

        # Dump to file
        # define float representer
        def float_representer(dumper, value):
            text = '{0:.15f}'.format(value)
            return dumper.represent_scalar('tag:yaml.org,2002:float', text)
        yaml.add_representer(float, float_representer)

        file = open(filename, 'w')
        # header
        header = dict(
            mesh=mesh,
            nqpoint=qcoords.shape[0],
            reciprocal_lattice=rlattmx,
            natom=len(self.scaled_positions.shape[0]),
            lattice=lattmx
        )
        yaml.dump(header, file, sort_keys=False, default_flow_style=None)
        del header
        # atoms
        points = []
        for crd, ele, mas in zip(self._phonopy.unitcell.scaled_positions,
                                 self._phonopy.unitcell.symbols,
                                 self._phonopy.unitcell.masses):
            points.append(dict(symbol=ele, coordinates=crd, mass=mas))
        yaml.dump(points, file, sort_keys=False, default_flow_style=None)
        del points
        file.write('\n')
        # Frequency
        phonon = []
        for crd, wei, freq in zip(qcoords, qweight, ha.freqency):
            phonon.append({
                'q-position' : crd,
                'distance_from_gamma' : np.linalg.norm(crd@rlattmx),
                'weight' : int(wei),
                'band' : [dict(frequency=i) for i in freq]
            })
        yaml.dump(phonon, file, sort_keys=False, default_flow_style=None)
        file.write('\n')
        file.close()
        return


