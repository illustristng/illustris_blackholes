"""
"""
# from collections import namedtuple
from glob import glob
import numpy as np
import os
import shutil

NUM_SNAPS = 136

_ILLUSTRIS_DETAILS_FILENAME_REGEX = "blackhole_details_*.txt"

_ILLUSTRIS_MERGERS_FILENAME_REGEX = "blackhole_mergers_*.txt"

_ILLUSTRIS_DETAILS_DIRS = {3: "/n/ghernquist/Illustris/Runs/L75n455FP/"
                              "output/blackhole_details/",

                           2: "/n/ghernquist/Illustris/Runs/L75n910FP/"
                              "combined_output/blackhole_details/",

                           1: ["/n/ghernquist/Illustris/Runs/L75n1820FP/"
                               "txt-files/txtfiles_new/txt-files-curie/blackhole_details/",

                               "/n/ghernquist/Illustris/Runs/L75n1820FP/"
                               "txt-files/txtfiles_new/txt-files-supermuc/blackhole_details/",

                               "/n/ghernquist/Illustris/Runs/L75n1820FP/"
                               "txt-files/txtfiles_new/txt-files-partial/Aug8/blackhole_details/",

                               "/n/ghernquist/Illustris/Runs/L75n1820FP/"
                               "txt-files/txtfiles_new/txt-files-partial/Aug14/blackhole_details/",

                               "/n/ghernquist/Illustris/Runs/L75n1820FP/"
                               "txt-files/txtfiles_new/txt-files-partial/Oct10/blackhole_details/",

                               "/n/ghernquist/Illustris/Runs/L75n1820FP/"
                               "txt-files/txtfiles_new/txt-files-partial/Sep25/blackhole_details/"]
                           }

_ILLUSTRIS_MERGERS_DIRS = {3: "/n/ghernquist/Illustris/Runs/L75n455FP/"
                              "output/blackhole_mergers/",

                           2: "/n/ghernquist/Illustris/Runs/L75n910FP/"
                              "combined_output/blackhole_mergers/",

                           1: ["/n/ghernquist/Illustris/Runs/L75n1820FP/"
                               "txt-files/txtfiles_new/txt-files-curie/blackhole_mergers/",

                               "/n/ghernquist/Illustris/Runs/L75n1820FP/"
                               "txt-files/txtfiles_new/txt-files-supermuc/blackhole_mergers/",

                               "/n/ghernquist/Illustris/Runs/L75n1820FP/"
                               "txt-files/txtfiles_new/txt-files-partial/Aug8/blackhole_mergers/",

                               "/n/ghernquist/Illustris/Runs/L75n1820FP/"
                               "txt-files/txtfiles_new/txt-files-partial/Aug14/blackhole_mergers/",

                               "/n/ghernquist/Illustris/Runs/L75n1820FP/"
                               "txt-files/txtfiles_new/txt-files-partial/Sep25/blackhole_mergers/",

                               "/n/ghernquist/Illustris/Runs/L75n1820FP/"
                               "txt-files/txtfiles_new/txt-files-partial/Oct10/blackhole_mergers/"]
                           }

_ILLUSTRIS_RUN_NAMES   = {1: "L75n1820FP",
                          2: "L75n910FP",
                          3: "L75n455FP"}

_ILLUSTRIS_SUBBOX_TIMES_FILENAMES = \
    "/n/ghernquist/Illustris/Runs/%s/postprocessing/subboxes_times.txt"

_SNAPSHOT_COSMOLOGY_DATA_FILENAME = "illustris-snapshot-cosmology-data.npz"

_OUTPUT_DIR = "/n/home00/lkelley/ghernquistfs1/illustris/data/%s/"
_OUTPUT_DETAILS_DIR = os.path.join(_OUTPUT_DIR, "blackholes/details/")
_OUTPUT_DETAILS_ORGANIZED_DIR = os.path.join(_OUTPUT_DETAILS_DIR, "organized")
_OUTPUT_DETAILS_ORGANIZED_FILENAME = "ill-%d_blackhole_details_snap-%03d"
_OUTPUT_DETAILS_FILENAME = "ill-%d_blackhole_details.hdf5"

_OUTPUT_MERGERS_DIR = os.path.join(_OUTPUT_DIR, "blackholes/mergers/")

_OUTPUT_MERGERS_COMBINED_FILENAME = "ill-{:d}_blackhole_mergers_combined_{:s}.{:s}"
# _OUTPUT_MERGERS_RAW_FILENAME_INTERMEDIATE = "ill-%d_blackhole_mergers_raw.txt"
_OUTPUT_MERGERS_DETAILS_FILENAME = "ill-%d_blackhole_merger_details.hdf5"

_PUBLIC_MERGERS_FILENAME = "ill-{:d}_blackhole_mergers.hdf5"


class DTYPE:
    ID     = np.uint64
    SCALAR = np.float64
    INDEX  = np.int64

# class BH_TYPE:
#     IN  = 0
#     OUT = 1


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


def _all_exist(files):
    retval = all([os.path.exists(fil) for fil in files])
    return retval

'''
def GET_PROCESSED_DIR(run):
    """
    _ILLUSTRIS_RUN_NAMES   = {1: "L75n1820FP",
                              2: "L75n910FP",
                              3: "L75n455FP"}
    _PROCESSED_DIR = "/n/home00/lkelley/ghernquistfs1/illustris/data/%s/output/postprocessing/"
    """
    return _PROCESSED_DIR % (_ILLUSTRIS_RUN_NAMES[run])
'''


def GET_DETAILS_ORGANIZED_FILENAME(run, snap, type='txt'):
    """
    /n/home00/lkelley/ghernquistfs1/illustris/data/%s/blackhole/details/organized/
        ill-%d_blackhole_details_temp_snap-%d.%s
    """
    use_dir = _OUTPUT_DETAILS_ORGANIZED_DIR % (_ILLUSTRIS_RUN_NAMES[run])
    use_fname = _OUTPUT_DETAILS_ORGANIZED_FILENAME % (run, snap) + '.%s' % format(type)
    fname = os.path.join(use_dir, use_fname)
    return fname


def GET_OUTPUT_DETAILS_FILENAME(run):
    """
    /n/home00/lkelley/ghernquistfs1/illustris/data/%s/blackhole/details/organized/
        "ill-%d_blackhole_details.hdf5"
    """
    use_dir = _OUTPUT_DETAILS_ORGANIZED_DIR % (_ILLUSTRIS_RUN_NAMES[run])
    use_fname = _OUTPUT_DETAILS_FILENAME % (run)
    fname = os.path.join(use_dir, use_fname)
    return fname


def GET_MERGERS_COMBINED_FILENAME(run, filtered=False, type='txt'):
    """
    /n/home00/lkelley/ghernquistfs1/illustris/data/%s/blackhole/mergers/
        ill-%d_blackhole_mergers_combined_{:s}.{:.s}
    """
    use_dir = _OUTPUT_MERGERS_DIR % (_ILLUSTRIS_RUN_NAMES[run])
    if filtered:
        ending = 'filtered'
    else:
        ending = 'raw'
    use_fname = _OUTPUT_MERGERS_COMBINED_FILENAME.format(run, ending, type)
    fname = os.path.join(use_dir, use_fname)
    return fname


def GET_MERGERS_DETAILS_FILENAME(run):
    """
    """
    use_dir = _OUTPUT_MERGERS_DIR % (_ILLUSTRIS_RUN_NAMES[run])
    use_fname = _OUTPUT_MERGERS_DETAILS_FILENAME % (run)
    fname = os.path.join(use_dir, use_fname)
    return fname


def GET_PUBLIC_MERGERS_FILENAME(run):
    use_dir = _OUTPUT_MERGERS_DIR % (_ILLUSTRIS_RUN_NAMES[run])
    use_fname = _PUBLIC_MERGERS_FILENAME.format(run)
    fname = os.path.join(use_dir, use_fname)
    return fname


def GET_ILLUSTRIS_BH_DETAILS_FILENAMES(run):
    file_paths = _ILLUSTRIS_DETAILS_DIRS[run]
    file_paths = np.atleast_1d(file_paths)

    det_files = []
    for fdir in file_paths:
        file_regex = os.path.join(fdir, _ILLUSTRIS_DETAILS_FILENAME_REGEX)
        match_files = sorted(glob(file_regex))
        det_files += match_files

    return det_files


def GET_ILLUSTRIS_BH_MERGERS_FILENAMES(run):
    files_dir = np.atleast_1d(_ILLUSTRIS_MERGERS_DIRS[run])
    files = []
    for fdir in files_dir:
        filesNames = os.path.join(fdir, _ILLUSTRIS_MERGERS_FILENAME_REGEX)
        someFiles = sorted(glob(filesNames))
        files += someFiles
    return files


def GET_SNAPSHOT_SCALES():
    fname = str(_SNAPSHOT_COSMOLOGY_DATA_FILENAME)
    if not os.path.isfile(fname):
        raise ValueError("Snapshot scales file '{}' is invalid".format(fname))
    data = np.load(fname)
    return data['scale']


def GET_SUBBOX_TIMES(run):
    # /n/ghernquist/Illustris/Runs/%s/postprocessing/subboxes_times.txt
    times_fname = _ILLUSTRIS_SUBBOX_TIMES_FILENAMES % _ILLUSTRIS_RUN_NAMES[run]
    if not os.path.isfile(times_fname):
        raise ValueError("Subbox times file '{}' does not exist.".format(times_fname))

    times = np.zeros(int(4e4))
    count = 0
    for line in open(times_fname, 'r'):
        num, scale = line.split(" ")
        times[int(num)] = float(scale)
        if int(num) != count:
            raise RuntimeError("Something went wrong.  count = {}, line = '{}'".format(count, line))
        count += 1

    times = times[:count]
    return times


def _backup_exists(fname, verbose=True, append='.bak'):
    if os.path.isfile(fname):
        back_fname = fname + '.bak'
        shutil.move(fname, back_fname)
        if verbose:
            print(" - Moved existing file\n\tFrom: '{}'\n\tTo: '{}'".format(fname, back_fname))
    return


def _zero_pad_end(arr, add_len):
    return np.pad(arr, (0, add_len), mode='constant', constant_values=0)
