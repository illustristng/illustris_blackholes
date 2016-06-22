"""
"""
from datetime import datetime
from glob import glob
import h5py
import numpy as np
import os
import shutil
import warnings

__version__ = '1.0'

_OUTPUT_DETAILS_DIR = "blackholes/details/"
_OUTPUT_DETAILS_ORGANIZED_DIR = os.path.join(_OUTPUT_DETAILS_DIR, "organized")
_OUTPUT_MERGERS_DIR = "blackholes/mergers/"

_OUTPUT_DETAILS_ORGANIZED_FILENAME = "{:s}_blackhole_details_snap-{:03d}.{:s}"
_PUBLIC_DETAILS_FILENAME = "{:s}_blackhole_details.hdf5"

_OUTPUT_MERGERS_COMBINED_FILENAME = "{:s}_blackhole_mergers_combined_{:s}.{:s}"
_OUTPUT_MERGERS_DETAILS_FILENAME = "{:s}_blackhole_merger_details.hdf5"

_PUBLIC_MERGERS_FILENAME = "{:s}_blackhole_mergers.hdf5"

_ILLUSTRIS_METADATA_FILENAME = "aux_data/{:s}_metadata.hdf5"

# Precision when comparing between scale-factors
_DEF_SCALE_PRECISION = -8


class DTYPE:
    ID     = np.uint64
    SCALAR = np.float64
    INDEX  = np.int64


class DETAILS:
    ID     = 'id'
    SCALE  = 'time'
    MASS   = 'mass'
    MDOT   = 'mdot'
    RHO    = 'rho'
    CS     = 'cs'

    UNIQUE_IDS = 'unique/id'
    UNIQUE_FIRST = 'unique/first_index'
    UNIQUE_NUM_PER = 'unique/num_entries'


class MERGERS:
    SCALE = 'time'
    ID_IN = 'id_in'
    ID_OUT = 'id_out'
    MASS_IN = 'mass_in'
    MASS_OUT = 'mass_out'
    SNAPSHOT = 'snapshot'

    _TREE = 'tree'
    NEXT = 'tree/next'
    PREV_IN = 'tree/prev_in'
    PREV_OUT = 'tree/prev_out'

    UNIQUE = 'Header/unique_ids'

    _DETAILS = 'details'
    DET_SCALE = 'details/' + DETAILS.SCALE
    DET_MASS = 'details/' + DETAILS.MASS
    DET_MDOT = 'details/' + DETAILS.MDOT
    DET_RHO = 'details/' + DETAILS.RHO
    DET_CS = 'details/' + DETAILS.CS


class META:
    NUM_SNAPS = 'num_snaps'
    SNAP_TIMES = 'snap_times'
    SUBB_NUM_SNAPS = 'subbox_subbox_snaps'
    SUBB_SNAP_TIMES = 'subbox_snap_times'
    MERGERS_FILENAMES = 'mergers_filenames'
    DETAILS_FILENAMES = 'details_filenames'
    NUM_MERGERS_FILES = 'num_mergers_files'
    NUM_DETAILS_FILES = 'num_details_files'


def _all_exist(files):
    retval = all([os.path.exists(fil) for fil in files])
    return retval


def _get_output_dir(run, output_dir, append=None):
    """
    """
    if output_dir is None:
        output_dir = os.path.append(os.path.abspath(os.path.curdir), run)
    if append is not None:
        output_dir = os.path.join(output_dir, append, '')
    return output_dir


def GET_DETAILS_ORGANIZED_FILENAME(run, snap, type='txt', output_dir=None):
    """
    /n/home00/lkelley/ghernquistfs1/illustris/data/%s/blackhole/details/organized/
        ill-%d_blackhole_details_temp_snap-%d.%s
    """
    # Append: "blackholes/details/organized/"
    output_dir = _get_output_dir(run, output_dir, append=_OUTPUT_DETAILS_ORGANIZED_DIR)
    use_fname = _OUTPUT_DETAILS_ORGANIZED_FILENAME.format(run, snap, type)
    fname = os.path.join(output_dir, use_fname)
    _check_path(fname)
    return fname


def GET_PUBLIC_DETAILS_FILENAME(run, output_dir=None):
    """
    /n/home00/lkelley/ghernquistfs1/illustris/data/%s/blackhole/details/organized/
        "ill-%d_blackhole_details.hdf5"
    """
    output_dir = _get_output_dir(run, output_dir, append=_OUTPUT_DETAILS_DIR)
    use_fname = _PUBLIC_DETAILS_FILENAME.format(run)
    fname = os.path.join(output_dir, use_fname)
    _check_path(fname)
    return fname


def GET_MERGERS_COMBINED_FILENAME(run, filtered=False, type='txt', output_dir=None):
    """
    /n/home00/lkelley/ghernquistfs1/illustris/data/%s/blackhole/mergers/
        ill-%d_blackhole_mergers_combined_{:s}.{:.s}
    """
    output_dir = _get_output_dir(run, output_dir, append=_OUTPUT_MERGERS_DIR)
    if filtered:
        ending = 'filtered'
    else:
        ending = 'raw'
    use_fname = _OUTPUT_MERGERS_COMBINED_FILENAME.format(run, ending, type)
    fname = os.path.join(output_dir, use_fname)
    _check_path(fname)
    return fname


def GET_MERGERS_DETAILS_FILENAME(run, output_dir=None):
    """
    """
    output_dir = _get_output_dir(run, output_dir, append=_OUTPUT_MERGERS_DIR)
    use_fname = _OUTPUT_MERGERS_DETAILS_FILENAME.format(run)
    fname = os.path.join(output_dir, use_fname)
    _check_path(fname)
    return fname


def GET_PUBLIC_MERGERS_FILENAME(run, output_dir=None):
    output_dir = _get_output_dir(run, output_dir, append=_OUTPUT_MERGERS_DIR)
    use_fname = _PUBLIC_MERGERS_FILENAME.format(run)
    fname = os.path.join(output_dir, use_fname)
    _check_path(fname)
    return fname


def get_illustris_metadata(run, keys, input_dir=None):
    """Load data corresponding to `keys` from the appropriate illustris metadata file.

    First make sure the metadata file exists, and is the current version.
    """
    keys = np.atleast_1d(keys)
    meta_fname = _ILLUSTRIS_METADATA_FILENAME.format(run)

    reload = False
    results = []
    if not os.path.isfile(meta_fname):
        warnings.warn("Metadata for '{:s}' does not exist.  Recreating.".format(run))
        reload = True
    else:
        with h5py.File(meta_fname, 'r') as meta_hdf5:
            meta_vers = meta_hdf5['Header'].attrs['script_version']
            # Make sure metafile is up to date
            if meta_vers != __version__:
                warnings.warn("Metadata file '{}' is out of date (v: '{}').  Recreating.".format(
                    meta_fname, meta_vers))
                reload = True

            # Load data
            else:
                for key in keys:
                    results.append(meta_hdf5[key].value)

    if reload:
        _load_illustris_metadata(run)
        with h5py.File(meta_fname, 'r') as meta_hdf5:
            for key in keys:
                results.append(meta_hdf5[key].value)

    if keys.size == 1:
        return results[0]

    return results


def _load_illustris_metadata(run, input_dir=None):
    beg_all = datetime.now()
    # /n/ghernquist/Illustris/Runs/{:s}/output/ ==> /n/ghernquist/Illustris/Runs/L75n1820FP/output/
    # ill_dir = _ILLUSTRIS_OUTPUT_DIR_BASE.format(_ILLUSTRIS_RUN_NAMES[run])
    if input_dir is None:
        input_dir = '/n/ghernquist/Illustris/Runs/L75n1820FP/'
    ill_dir = os.path.join(input_dir, 'output', '')

    meta_fname = _ILLUSTRIS_METADATA_FILENAME.format(run)
    # Make sure path to save meta-files exists
    _check_path(meta_fname)
    print("Illustris directory: '{}'".format(ill_dir))

    # Get number of Snapshots, and Snapshot Times
    # -------------------------------------------
    beg = datetime.now()
    snap_dirs = sorted(glob(os.path.join(ill_dir, 'snapdir_*')))
    num_snaps = len(snap_dirs)
    snap_scales = np.zeros(num_snaps, dtype=DTYPE.SCALAR)
    # Get scalefactors of each snapshot
    for ii, sdir in enumerate(snap_dirs):
        regex = os.path.join(sdir, 'snap_*.0.hdf5')
        fname = glob(regex)
        if len(fname) != 1:
            raise RuntimeError("Wrong number of matches for '{}'".format(regex))
        with h5py.File(fname[0], 'r') as snap_hdf5:
            snap_scales[ii] = snap_hdf5['Header'].attrs['Time']

    snap_scales = np.sort(snap_scales)
    print("Num Snaps: {}; after {}".format(num_snaps, datetime.now()-beg))

    # Get Subbox Snapshot Times
    # -------------------------
    beg = datetime.now()
    regex = os.path.join(ill_dir, 'subbox0/snapdir_subbox0_*')
    subb_snap_dirs = sorted(glob(regex))
    subb_num_snaps = len(subb_snap_dirs)
    # For Illustris-3, subbox snapshots are right in the root directory
    if subb_num_snaps == 0:
        regex = os.path.join(ill_dir, 'subbox0/snap_subbox0_*.hdf5')
        subb_snaps = sorted(glob(regex))
        subb_num_snaps = len(subb_snaps)
        subb_snap_scales = np.zeros(subb_num_snaps, dtype=DTYPE.SCALAR)
        # Get scalefactors of each snapshot
        for ii, fname in enumerate(subb_snaps):
            with h5py.File(fname, 'r') as snap_hdf5:
                subb_snap_scales[ii] = snap_hdf5['Header'].attrs['Time']

    else:
        subb_snap_scales = np.zeros(subb_num_snaps, dtype=DTYPE.SCALAR)
        # Get scalefactors of each snapshot
        for ii, sdir in enumerate(subb_snap_dirs):
            regex = os.path.join(sdir, 'snap_subbox0_*.0.hdf5')
            fname = glob(regex)
            if len(fname) != 1:
                raise RuntimeError("Wrong number of matches for '{}'".format(regex))
            with h5py.File(fname[0], 'r') as snap_hdf5:
                subb_snap_scales[ii] = snap_hdf5['Header'].attrs['Time']

    subb_snap_scales = np.sort(subb_snap_scales)
    print("Num subbox snaps: {}; after {}".format(subb_num_snaps, datetime.now()-beg))

    # Get all mergers and details file-names
    # --------------------------------------
    beg = datetime.now()
    details_fnames = []
    mergers_fnames = []
    dets_regex = "blackhole_details/blackhole_details_*.txt"
    mrgs_regex = "blackhole_mergers/blackhole_mergers_*.txt"
    exclude = ['setup', 'code', 'Arepo', 'trees']

    def test_dir(path):
        if path in exclude:
            return False
        if path.startswith('groups'):
            return False
        if path.startswith('snapdir'):
            return False
        if path.startswith('subbox'):
            return False
        return True

    start_walk_dir = str(ill_dir)
    # Avoid issues with Illustris-1 txt file structure...
    if start_walk_dir.startswith('/n/ghernquist/Illustris/Runs/L75n1820FP/'):
        start_walk_dir = "/n/ghernquist/Illustris/Runs/L75n1820FP/txt-files/txtfiles_new/"
        warnings.warn("Overriding walk directory to '{}'".format(start_walk_dir))
    for dir_name, subdirs, files in os.walk(start_walk_dir):
        # Modify `subdirs` *in-place* to ignore directories in walk
        subdirs[:] = [sd for sd in subdirs if test_dir(sd)]
        print(dir_name)
        if "blackhole_details" in subdirs:
            details_fnames += glob(os.path.join(dir_name, dets_regex))
            print("\t", details_fnames[-1])
            subdirs[:] = [sd for sd in subdirs if sd != "blackhole_details"]

        if "blackhole_mergers" in subdirs:
            mergers_fnames += glob(os.path.join(dir_name, mrgs_regex))
            print("\t", mergers_fnames[-1])
            subdirs[:] = [sd for sd in subdirs if sd != "blackhole_mergers"]

    mergers_fnames = np.array(sorted([mf.encode("ascii", "ignore") for mf in mergers_fnames]))
    num_mergers_files = len(mergers_fnames)
    details_fnames = np.array(sorted([df.encode("ascii", "ignore") for df in details_fnames]))
    num_details_files = len(details_fnames)
    print("Found {} mergers files, {} details files; after {}".format(
        num_mergers_files, num_details_files, datetime.now()-beg))

    print("Saving metadata to '{}'".format(meta_fname))
    with h5py.File(meta_fname, 'w') as meta_hdf5:
        head = meta_hdf5.create_group('Header')
        head.attrs['script'] = str(__file__)
        head.attrs['script_version'] = str(__version__)
        head.attrs['git_version'] = str(_get_git())
        head.attrs['created'] = str(datetime.now().ctime())
        head.attrs['simulation'] = '{}'.format(run)

        meta_hdf5[META.NUM_SNAPS] = num_snaps
        meta_hdf5[META.SNAP_TIMES] = snap_scales
        meta_hdf5[META.SUBB_NUM_SNAPS] = subb_num_snaps
        meta_hdf5[META.SUBB_SNAP_TIMES] = subb_snap_scales
        meta_hdf5[META.MERGERS_FILENAMES] = mergers_fnames
        meta_hdf5[META.DETAILS_FILENAMES] = details_fnames
        meta_hdf5[META.NUM_MERGERS_FILES] = num_mergers_files
        meta_hdf5[META.NUM_DETAILS_FILES] = num_details_files

    fsize = os.path.getsize(meta_fname)/1024/1024
    print("'{}' complete after {}, size: {} MB".format(meta_fname, datetime.now()-beg_all, fsize))
    return


def _backup_exists(fname, verbose=True, append='.bak'):
    if os.path.isfile(fname):
        back_fname = fname + '.bak'
        shutil.move(fname, back_fname)
        if verbose:
            print(" - Moved existing file\n\tFrom: '{}'\n\tTo: '{}'".format(fname, back_fname))
    return


def _zero_pad_end(arr, add_len):
    return np.pad(arr, (0, add_len), mode='constant', constant_values=0)


def _get_git():
    """Get a string representing the current git status --- i.e. tag and commit hash.
    """
    import subprocess
    git_vers = subprocess.getoutput(["git describe --always"]).strip()
    return git_vers


def _check_version(fname, vers):
    try:
        with h5py.File(fname, 'r') as h5file:
            file_vers = h5file['Header'].attrs['script_version']
        if file_vers == vers:
            return True
    except Exception as err:
        warnings.warn("`_check_version`:" + str(err))

    return False


def _check_path(fpath):
    head, tail = os.path.split(os.path.abspath(fpath))
    if not os.path.exists(head):
        os.makedirs(head)
    if not os.path.isdir(head):
        raise RuntimeError("Path '{}' (from '{}') is invalid!".format(head, fpath))
    return


def scale_to_age_flat(scales):
    """Calculate age of the universe [sec] at the given redshift (assuming flat cosmology).

    Analytical formula from Peebles, p.317, eq 13.2.
    """
    is_scalar = np.isscalar(scales)
    redshifts = (1.0/np.atleast_1d(scales)) - 1.0

    ages = np.zeros(redshifts.size, dtype='float32')

    H0_kmsMpc = 70.0
    omega_m = 0.2726

    arcsinh_arg = np.sqrt((1-omega_m)/omega_m) * (1+redshifts)**(-3.0/2.0)
    ages = 2 * np.arcsinh(arcsinh_arg) / (H0_kmsMpc * 3 * np.sqrt(1-omega_m))
    ages *= 3.085678e+19   # Seconds

    if is_scalar:
        ages = ages[0]

    return ages
