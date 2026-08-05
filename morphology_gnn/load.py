import h5py
import sys

if sys.version_info < (3,):
    RANGE_TYPE = list
else:
    RANGE_TYPE = range


def load(chkfile, key):
    """Load array(s) from chkfile

    Args:
        chkfile : str
            Name of chkfile. The chkfile needs to be saved in HDF5 format.
        key : str
            HDF5.dataset name or group name.  If key is the name of a HDF5
            group, the group will be loaded into a Python dict, recursively.

    Returns:
        whatever read from chkfile

    Examples:

    >>> from pyscf import gto, scf, lib
    >>> mol = gto.M(atom='He 0 0 0')
    >>> mf = scf.RHF(mol)
    >>> mf.chkfile = 'He.chk'
    >>> mf.kernel()
    >>> mo_coeff = lib.chkfile.load('He.chk', 'scf/mo_coeff')
    >>> mo_coeff.shape
    (1, 1)
    >>> scfdat = lib.chkfile.load('He.chk', 'scf')
    >>> scfdat.keys()
    ['e_tot', 'mo_occ', 'mo_energy', 'mo_coeff']
    """

    def load_as_dic(key, group):
        if key in group:
            val = group[key]
        elif key + "__from_list__" in group:
            key = key + "__from_list__"
            val = group[key]
        else:
            return None

        if isinstance(val, h5py.Group):
            if key.endswith("__from_list__"):
                return [load_as_dic(k, val) for k in val]
            else:
                return {
                    k.replace("__from_list__", ""): load_as_dic(k, val) for k in val
                }
        else:
            return val[()]

    with h5py.File(chkfile, "r") as fh5:
        return load_as_dic(key, fh5)


load_chkfile_key = load
