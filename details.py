"""Handle Illustris blackhole details files.



API Functions
-------------


Details Dictionary
------------------


Notes
-----
  - The BH Details files from illustris, 'blackhole_details_<#>.txt' are organized by the processor
    on which each BH existed in the simulation.  The method `_reorganizeBHDetails()` sorts each
    detail entry instead by the time (scalefactor) of the entry --- organizing them into files
    grouped by which snapshot interval the detail entry corresponds to.  The reorganization is
    first done into 'temporary' ASCII files before being converted into numpy `npz` files by the
    method `_convertDetailsASCIItoNPZ()`.  The `npz` files are effectively dictionaries storing
    the select details parameters (i.e. mass, BH ID, mdot, rho, cs), along with some meta data
    about the `run` number, and creation time, etc.  Execution of the BHDetails ``main`` routine
    checks to see if the npz files exist, and if they do not, they are created.

  - There are also routines to obtain the details entries for a specific BH ID.  In particular,
    the method `detailsForBH()` will return the details entry/entries for a target BH ID and
    run/snapshot.

  - Illustris Blackhole Details Files 'blackhole_details_<#>.txt'
    - Each entry is given as
      0   1            2     3     4    5
      ID  scalefactor  mass  mdot  rho  cs


NOTE: 'details' entries at the same time as a 'merger' seem to always be 'after' merger based on
      the masses.

"""

from datetime import datetime
import h5py
import os
import numpy as np
import shutil
# import sys
import warnings

# import zcode.inout as zio

# from illpy.Constants import DTYPE, NUM_SNAPS
# from . import BHConstants
# from . BHConstants import DETAILS
from constants import _all_exist, DETAILS, DTYPE, GET_DETAILS_ORGANIZED_FILENAME, \
    GET_ILLUSTRIS_BH_DETAILS_FILENAMES, GET_MERGERS_DETAILS_FILENAME, \
    GET_MERGERS_COMBINED_FILENAME, \
    GET_OUTPUT_DETAILS_FILENAME, GET_SNAPSHOT_SCALES, GET_SUBBOX_TIMES, MERGERS, NUM_SNAPS

__version__ = '1.0'

# VERSION = 0.23                                    # Version of BHDetails

# Precision when comparing between scale-factors
_DEF_SCALE_PRECISION = -8

# Size of initial Details arrays allocation; and 'chunk' size that it is increased by as needed
_DETS_BUF_SIZE = int(1e7)


class _MTYPE:
    M_BEF_D = "Merger then detail entry"
    D_BEF_M = "Detail entry then merger entry"
    UNKNOWN = "Unknown merger vs. detail order"
    CONFLICT = "Details in conflict with merger"
    ASYNC = "No details at merger time"
    ASYNC_CONFLICT = "Surrounding details conflict with merger"
    PARTIAL = "Only partial matches"

    _KEYS = ["M_BEF_D", "D_BEF_M", "UNKNOWN", "CONFLICT", "ASYNC", "ASYNC_CONFLICT", "PARTIAL"]

    @classmethod
    def _VALS(cls):
        for key in cls._KEYS:
            yield getattr(cls, key)


def main(run, output_dir=None, verbose=True, reorganize=False, reconvert=False):
    """
    """

    '''
    # Raw ('txt') ==> Organized ('txt')
    # ---------------------------------
    organized_fnames_txt = [GET_DETAILS_ORGANIZED_FILENAME(run, snap, type='txt')
                            for snap in range(NUM_SNAPS)]
    org_txt_exists = _all_exist(organized_fnames_txt)
    if verbose and org_txt_exists:
        print("All 'organized' details txt files exist.")

    # Move all Details Entries from Illustris Raw Details files, into files with details organized
    #    by snapshot.  No processing of data.
    if reorganize or not org_txt_exists:
        if verbose: print("Organizing details files.")
        # Organize Details by Snapshot Time; create new, temporary ASCII Files
        beg = datetime.now()
        organize_txt_by_snapshot(run, verbose=verbose)
        if verbose: print("Organization complete after {}".format(datetime.now()-beg))

    # Organized ('txt') ==> Organized ('hdf5')
    # ----------------------------------------
    # Convert data from ASCII to hdf5, still no processing / trimming of data
    organized_fnames_hdf5 = [GET_DETAILS_ORGANIZED_FILENAME(run, snap, type='hdf5')
                             for snap in range(NUM_SNAPS)]
    org_hd5f_exists = _all_exist(organized_fnames_hdf5)
    if verbose and org_hd5f_exists:
        print("All 'organized' details hdf5 files exist.")

    if reconvert or not org_hd5f_exists:
        if verbose: print("Converting txt files to hdf5")
        convert_txt_to_hdf5(run, verbose)
        if verbose: print("Conversion complete after {}".format(datetime.now()-beg))

    # Combine and downsample from details snapshot files into general- and merger-details files.
    '''

    return


def organize_txt_by_snapshot(run, verbose=True, clean_overwrites=True):
    """Move Raw Illustris details entries (txt) into files with organized by snapshot (still txt).

    Notes
    -----
    -   Produces 136 (`NUM_SNAPS`) output files containing all details entries up to (and including)
        the time of the snapshot.
    -   Attempts to clean duplicate entries produced when Illustris Restarts (crashes?), if the
        argument `clean_overwrites == True`.
        -   In these cases, there are details entries output between the time at which illustris
            will restart, and the point at which it previously terminated.  This time span will be
            rerun after the restart, but the details entries (nor merger entries?) were not deleted,
            and thus duplicate entries occur.
        -   Method: if a non-monotonicity in scale-factor is detected, s_i > s_i+1, then we assume
            the new (earlier) entry (s_r = s_i+1) corresponds to a restart, and all previous entries
            with times (s_j > s_r) are deleted.
            -   This is not foolproof.  If, for example, the overwritten section crosses
                different files - this method won't catch it.

    Arguments
    ---------
    run : int
        Illustris simulation number {1, 3}
    verbose : bool
        Print basic output information during execution.
    clean_overwrites : bool
        Use a simple method to clean-up duplicate details entries (see 'Notes' above).

    """
    print(" - Organizing details entries by snapshot.")

    # Load scalefactor (time) for each snapshot
    snap_scales = GET_SNAPSHOT_SCALES()

    # Output filenames, one per snapshot (even if empty---i.e. no blackholes)
    output_fnames = [GET_DETAILS_ORGANIZED_FILENAME(run, snap, type='txt')
                     for snap in range(NUM_SNAPS)]
    # Input filenames, raw illustris output ('blackhole_details_*.txt') 10,000s for Illustris-1
    input_fnames = GET_ILLUSTRIS_BH_DETAILS_FILENAMES(run)

    # Make sure output path is okay
    head_dir, tail_path = os.path.split(output_fnames[0])
    if not os.path.exists(head_dir):
        os.makedirs(head_dir)
    if not os.path.isdir(head_dir):
        raise RuntimeError("Output '{}' is not a valid directory".format(head_dir))
    # Open each output ('org'anized) file
    org_files = [open(tfil, 'w') for tfil in output_fnames]

    num_org = len(output_fnames)
    num_raw  = len(input_fnames)
    num_iter = int(np.floor(num_raw/100))
    if verbose:
        print(" - - Organizing {} raw files into {} organized files".format(num_raw, num_org))
    beg = datetime.now()
    num_lines = 0
    num_deleted = 0
    # Go through each input file
    for ii, raw_fn in enumerate(input_fnames):
        raw_lines = []    # Stores each line from input file
        raw_scales = []   # Stores the scale-factor of each entry (each line)
        last_scale = 0.0  # Stores the previous scale-factor (time)
        # Load all lines and entry scale-factors from raw details file
        for dline in open(raw_fn, 'r'):
            # Extract scale-factor from line
            det_scale = DTYPE.SCALAR(dline.split()[1])

            # Remove Duplicated Entries
            # -------------------------
            # If the times go backwards, simulation was restarted.  erase (previous) overlap segment
            if clean_overwrites and det_scale < last_scale:
                # Find the beginning of the overlap
                time_mask = (det_scale < raw_scales) | np.isclose(raw_scales, det_scale)
                bads = np.where(time_mask)[0]
                if bads.size:
                    # Reverse through so that subsequent indices are still correct
                    for idx in reversed(bads):
                        del raw_lines[idx]
                        del raw_scales[idx]
                    num_deleted += bads.size

            # Store each line and scale-factor
            raw_lines.append(dline)
            raw_scales.append(det_scale)
            last_scale = det_scale

        # Convert to array
        raw_lines  = np.array(raw_lines)
        raw_scales = np.array(raw_scales)

        # If file is empty, continue
        if not raw_lines.size or not raw_scales.size:
            continue

        # Round snapshot scales to desired precision
        round_scales = np.around(snap_scales, -_DEF_SCALE_PRECISION)

        # Find snapshots following each entry (right-edge) or equal (include right: 'right=True')
        #    Each row of `snap_bins` corresponds to a details entry, and includes the snapshot in
        #    which it belongs.
        snap_bins = np.digitize(raw_scales, round_scales, right=True)

        # For each Snapshot, write appropriate lines
        for jj in range(num_org):
            inds = np.where(snap_bins == jj)[0]
            if inds.size:
                org_files[jj].writelines(raw_lines[inds])
                num_lines += inds.size

        if verbose and ii % num_iter == 0:
            dur = datetime.now() - beg
            print(" - - - {:5d}/{} = {:.4f} after {}; {:.3e} lines written, {:.3e} deleted".format(
                ii, num_raw, ii/num_raw, dur, num_lines, num_deleted))

    # Close out details files
    tot_file_sizes = 0.0
    for ii, orgf in enumerate(org_files):
        orgf.close()
        tot_file_sizes += os.path.getsize(orgf.name)

    if verbose:
        tot_file_sizes /= 1024/1024
        ave_size = tot_file_sizes/num_org
        print(" - - Total organized size = '{:.2f}' [MB], average = '{:.2f}' [MB]".format(
            tot_file_sizes, ave_size))

    return


def convert_txt_to_hdf5(run, verbose=True):
    """Convert the 'organized' (by snapshot) txt files into hdf5 files (still org. by snapshot).

    Notes
    -----
    -   All of the details entries should remain identical, but in the hdf5 file they are sorted by
        BH ID number and then by time (scale-factor).
    -   Additional meta-data is also added about the number of entries, this generation script, and
        information about each unique BH:
        -   `DETAILS.UNIQUE_IDS`: contains each unique ID number (in order)
        -   `DETAILS.UNIQUE_FIRST`: the details entry number each unique ID first appears (in order)
        -   `DETAILS.UNIQUE_NUM_PER`: the number of entries for each unique BH (in order).
    -   Stores all details data (no filtering, processing etc).

    Arguments
    ---------
    run : int
        Illustris simulation number {1, 3}
    verbose: bool
        Print basic information during execution.

    """
    print(" - Converting files from 'txt' to 'hdf5'")
    # Input filenames for organized 'txt' files
    input_fnames = [GET_DETAILS_ORGANIZED_FILENAME(run, snap, type='txt')
                    for snap in range(NUM_SNAPS)]
    # Make sure they all exist
    txt_exist = _all_exist(input_fnames)
    if not txt_exist:
        raise ValueError("'txt' files (e.g. '{}') do not all exist.".format(input_fnames[0]))

    # Output filenames for hdf5 data
    output_fnames = [GET_DETAILS_ORGANIZED_FILENAME(run, snap, type='hdf5')
                     for snap in range(NUM_SNAPS)]

    tot_num_entries = 0
    beg = datetime.now()
    # Iterate over all files (note some at early snapshots are empty)
    for snap in range(NUM_SNAPS):
        with open(input_fnames[snap], 'r') as tfile, h5py.File(output_fnames[snap], 'w') as h5file:
            # Load details data (in proper numerical forms) from ASCII File
            id, scale, mass, mdot, rho, cs = _load_details_txt(tfile)
            # Sort by BH ID, then by time (scale-factor)
            id, scale, mass, mdot, rho, cs = _sort_details(id, scale, mass, mdot, rho, cs)
            # Find unique IDs
            q_ids, q_locs, q_counts = np.unique(id, return_index=True, return_counts=True)

            # Create Header with meta/summary data
            head = h5file.create_group('Header')
            head.attrs['script'] = str(__file__)
            head.attrs['script_version'] = str(__version__)
            head.attrs['created'] = str(datetime.now().ctime())
            head.attrs['simulation'] = 'Illustris-{}'.format(run)
            head.attrs['num_entries'] = id.size
            head.attrs['num_blackholes'] = q_ids.size

            # If there are no Blackholes in this snapshot, only write meta-data, skip rest
            if not id.size:
                continue

            head.attrs['time_first'] = np.min(scale)
            head.attrs['time_last'] = np.max(scale)

            # Store data on the unique BHs
            h5file[DETAILS.UNIQUE_IDS] = q_ids
            h5file[DETAILS.UNIQUE_FIRST] = q_locs
            h5file[DETAILS.UNIQUE_NUM_PER] = q_counts

            # Store all 'details' data
            time_dset = h5file.create_dataset(DETAILS.SCALE, data=scale)
            time_dset.attrs['desc'] = 'Cosmological scale factor'
            h5file.create_dataset(DETAILS.ID, data=id)
            h5file.create_dataset(DETAILS.MASS, data=mass)
            h5file.create_dataset(DETAILS.MDOT, data=mdot)
            h5file.create_dataset(DETAILS.RHO, data=rho)
            h5file.create_dataset(DETAILS.CS, data=cs)

            tot_num_entries += id.size

        if verbose:
            dur = datetime.now() - beg
            print(" - {:3d} BH: {:5d}, entries: {:.2e}, median per: {:5.0f}."
                  " Total: {:.2e} entries after {}".format(
                      snap, q_ids.size, id.size, np.median(q_counts), tot_num_entries, dur))

    return


def combine_downsample_and_mergers_hdf5(run, verbose=True, error_in_aft=False):
    """Convert hdf5 details files into one general-details file, and mergers-details file.

    Notes
    -----
    -   NOTE: Runtime is ~7-8 hrs.
    -   Goes through all of the individual hdf5 details file, which include all (non-duplicate)
        details entries, and store entries desired for output files.  Two types of output files are
        produced: 'general-details' for all BH, and 'merger-details' for merger BH.
        -   General-Details: details entries for all BH, downsampled to times resembling subboxes.
            -   For each unique BH ID in each snapshot interval, the details entries closest
                to the target times (the output times of illustris subboxes) are stored.
            -   Also try to store the very first and very last entry for each BH.
            -   NOTE: details data is *not* resampled or interpolated, we just find the entries
                which are *nearest* to the target times, and use those.  Interplation can be done
                during usage.
        -   Merger-Details: details entries for BH involved in mergers, near the time of merger.
            -   For a given snapshot (interval), details entries are searched for BH IDs with merger
                events during (or near) the snapshot.  We attempt to find details entries for the:
                0: 'in' BH, just before merger
                1: 'out' BH, just before merger
                2: 'out' BH, just after merger (corresponding to merger remnant, in+out BH)
            -   The resulting arrays are thus [N, 3] in shape, for 'N' mergers.
            -   For the 'out' BH, a sanity check is done to make sure the change in mass from
                'bef' to 'aft' is consistent with gaining the mass of the 'in' BH.
            -   Matches are not always successful, and missing entries remain zero.

    Arguments
    ---------

    """
    print(" - Combining and down-sampling details and identifying merger-related entries.")
    # Input hdf5 detaisl filenames
    input_fnames = [GET_DETAILS_ORGANIZED_FILENAME(run, snap, type='hdf5')
                    for snap in range(NUM_SNAPS)]
    # Make sure they all, already exist
    if not _all_exist(input_fnames):
        raise RuntimeError("Organized details files dont exist (e.g. '{}')".format(input_fnames[0]))

    # Load times (scale-factors) corresponding to subbox output, these are higher resolution
    #    output times at which we'll store details entries.
    target_scales = GET_SUBBOX_TIMES(run)
    if verbose: print(" - {} Target times.".format(target_scales.size))

    # Load mergers data
    with h5py.File(GET_MERGERS_COMBINED_FILENAME(run, filtered=True, type='hdf5'), 'r') as mergs_in:
        mscales = mergs_in[MERGERS.SCALE][:]      # Scalefactors of mergers
        m_mass_in = mergs_in[MERGERS.MASS_IN][:]  # Mass of the 'in' BH ('out' is incorrect)
        m_ids_in = mergs_in[MERGERS.ID_IN][:]     # IDs of the 'in' BH
        m_ids_out = mergs_in[MERGERS.ID_OUT][:]   # IDs of the 'out' BH
        num_mergers = mscales.size
        if verbose:
            print(" - Loaded {} mergers from '{}'".format(num_mergers, mergs_in.filename))

    # Initialize Storage for merger-details ('md')
    md_id    = np.zeros([num_mergers, 3], dtype=DTYPE.ID)
    md_scale = np.zeros([num_mergers, 3], dtype=DTYPE.SCALAR)
    md_mass  = np.zeros([num_mergers, 3], dtype=DTYPE.SCALAR)
    md_mdot  = np.zeros([num_mergers, 3], dtype=DTYPE.SCALAR)
    md_rho   = np.zeros([num_mergers, 3], dtype=DTYPE.SCALAR)
    md_cs    = np.zeros([num_mergers, 3], dtype=DTYPE.SCALAR)

    # Storage for details entries
    #    Make initial guess for how many entries there will be (resized later as-needed)
    id    = np.zeros(_DETS_BUF_SIZE, dtype=DTYPE.ID)
    scale = np.zeros(_DETS_BUF_SIZE, dtype=DTYPE.SCALAR)
    mass  = np.zeros(_DETS_BUF_SIZE, dtype=DTYPE.SCALAR)
    mdot  = np.zeros(_DETS_BUF_SIZE, dtype=DTYPE.SCALAR)
    rho   = np.zeros(_DETS_BUF_SIZE, dtype=DTYPE.SCALAR)
    cs    = np.zeros(_DETS_BUF_SIZE, dtype=DTYPE.SCALAR)
    det_ids = []    # ID numbers for BHs with some entries already
    num_det_ids = 0

    # Load All Details Data, for details and merger-details output
    # ============================================================
    beg_all = datetime.now()
    count = 0
    match_count = {val: 0 for val in _MTYPE._VALS()}
    for snap in range(NUM_SNAPS):
        # for snap in range(128, 134):
        with h5py.File(input_fnames[snap], 'r') as h5file_in:
            s_num_entries = h5file_in['Header'].attrs['num_entries']
            # If there are no details in this snapshot, skip
            if not s_num_entries:
                continue

            # Find BH IDs in next snapshot ('None' if this is the last snapshot); used for:
            #    a) Make sure we have the last detail entry for each BH,
            #    b) Handle merger events near snapshots (and thus at file boundaries)
            if snap < NUM_SNAPS-1:
                with h5py.File(input_fnames[snap+1], 'r') as next_h5file:
                    next_q_ids = next_h5file[DETAILS.UNIQUE_IDS][:]
            else:
                next_q_ids = None

            # Find which of the target times are covered in this snapshot
            first = h5file_in['Header'].attrs['time_first']
            last = h5file_in['Header'].attrs['time_last']
            targets = target_scales[(target_scales >= first) & (target_scales <= last)]

            # Load unique BH information
            q_ids = h5file_in[DETAILS.UNIQUE_IDS]
            q_first = h5file_in[DETAILS.UNIQUE_FIRST]
            q_num = h5file_in[DETAILS.UNIQUE_NUM_PER]

            # Merger-Details : Find entries for merger-BH in this snapshot
            # ------------------------------------------------------------

            # Find the indices of 'unique' details IDs which match BH in mergers
            #    `q_locs_` are unique details IDs (corresponding to `q_ids`...)
            #    `m_locs_` are merger numbers
            #    'in' are matches for the 'in' BH, and 'out' are matches for the 'out' BH
            q_locs_in, m_locs_in, snap_mergers_in = _unique_locs_for_mergers(
                mscales, m_ids_in, q_ids, first, last)
            q_locs_out, m_locs_out, snap_mergers_out = _unique_locs_for_mergers(
                mscales, m_ids_out, q_ids, first, last)
            # Mergers matching w/ 'in' *or* 'out' BH
            snap_mergers_one = set(snap_mergers_in).union(snap_mergers_out)
            # Mergers matching w/ 'in' *and* 'out' BH
            snap_mergers_both = set(snap_mergers_in).intersection(snap_mergers_out)

            # Using the unique BH IDs which match merger IDs, find the entries closest to merger.
            # --

            # 'in' BH: Iterate over pairs of unique BH ID indices `qq`, and merger indices `mm`
            for qq, mm in zip(q_locs_in, m_locs_in):
                mtime = mscales[mm]   # Time of this merger
                # Find the details entry numbers closest (before and after) to merger time
                #    Because this is an 'in' BH, there shouldnt be an `aft` value (should be 'None')
                bef, aft = _bef_aft_scales_in(qq, mtime, h5file_in)
                # Store this 'before' entry if its the first one, or its better
                #    `mm` is merger number, '0' is for 'in'-'bef'
                md_loc = (mm, 0)
                if ((bef is not None and
                     (np.isclose(md_scale[md_loc], 0.0) or
                      h5file_in[DETAILS.SCALE][bef] > md_scale[md_loc]))):
                    _store_dets_from_hdf5(bef, h5file_in, md_loc,
                                          md_id, md_scale, md_mass, md_mdot, md_rho, md_cs)
                # If `error_in_aft` is True, Raise error if 'in' BH has an 'after' entry
                #    In general it shouldnt, unless the merger or detail entry is in error.
                #    General procedure is to try to catch these errors in a later processing state.
                if aft is not None and error_in_aft:
                    print("Snap {}, Merger {}, time = {}".format(snap, mm, mtime))
                    raise RuntimeError("'in' has `aft` = {}, scale = {}".format(
                        aft, h5file_in[DETAILS.SCALE][aft]))

            # 'out' BH: Iterate over pairs of unique BH ID indices `qq`, and merger indices `mm`
            for qq, mm in zip(q_locs_out, m_locs_out):
                mtime = mscales[mm]   # Time of this merger
                # Find the details entry numbers closest (before and after) to merger time
                #    Use 'in' BH mass as a way of checking details entries' consistency w/ merger
                #    `match_type` describes the result of the mass comparison
                bef, aft, match_type = _bef_aft_scales_out(qq, m_mass_in[mm], mtime, h5file_in)
                match_count[match_type] += 1

                # Store this 'before' entry if its the first one, or its better (later than prev)
                #    `mm` is merger number, '1' is the for 'out'-'bef'
                md_loc = (mm, 1)
                if ((bef is not None and
                     (np.isclose(md_scale[md_loc], 0.0) or
                      h5file_in[DETAILS.SCALE][bef] > md_scale[md_loc]))):
                    _store_dets_from_hdf5(bef, h5file_in, md_loc,
                                          md_id, md_scale, md_mass, md_mdot, md_rho, md_cs)

                # Store this 'after' entry if its the first one, or its better (earlier than prev)
                #    `mm` is merger number, '2' is for 'out'-'aft'
                md_loc = (mm, 2)
                if ((aft is not None and
                     (np.isclose(md_scale[md_loc], 0.0) or
                      h5file_in[DETAILS.SCALE][aft] > md_scale[md_loc]))):
                    _store_dets_from_hdf5(aft, h5file_in, md_loc,
                                          md_id, md_scale, md_mass, md_mdot, md_rho, md_cs)

            # Details: for each unique BH, store downsampled details
            # ------------------------------------------------------
            # Iterate over info for each unique BH
            for qid, qf, qn in zip(q_ids, q_first, q_num):
                lo = qf     # Index of first entry for this BH
                hi = qf+qn  # Index of last (+1) entry for this BH
                # Record the index numbers nearest in time to the target times
                src = lo + _args_nearest(h5file_in[DETAILS.SCALE][lo:hi], targets)
                # Select only unique entries
                src = np.unique(src)
                # Make sure the very *first* entry for this BH is stored
                #    If this is the first time seeing thie BH ID
                if src[0] != lo:
                    if ((num_det_ids == 0 or
                         det_ids[np.searchsorted(det_ids, qid).clip(max=num_det_ids-1)] != qid)):
                        src = np.append(lo, src)
                # Make sure the very *last* entry for this BH is stored
                #    If this BH not in the next snapshot (or this is last snap)
                if src[-1] != hi-1:
                    if (next_q_ids is None or len(det_ids) == 0 or
                        det_ids[np.searchsorted(next_q_ids, qid).clip(
                            max=num_det_ids-1)] != qid):
                        src = np.append(src, hi-1)

                # Resize arrays if we reach edge
                if count + src.size >= id.size:
                    id = np.pad(id, (0, _DETS_BUF_SIZE), mode='constant', constant_values=0)
                    scale = np.pad(scale, (0, _DETS_BUF_SIZE), mode='constant', constant_values=0)
                    mass = np.pad(mass, (0, _DETS_BUF_SIZE), mode='constant', constant_values=0)
                    mdot = np.pad(mdot, (0, _DETS_BUF_SIZE), mode='constant', constant_values=0)
                    rho = np.pad(rho, (0, _DETS_BUF_SIZE), mode='constant', constant_values=0)
                    cs = np.pad(cs, (0, _DETS_BUF_SIZE), mode='constant', constant_values=0)

                # Store select entries into final details arrays
                des = slice(count, count+src.size)
                _store_dets_from_hdf5(src, h5file_in, des, id, scale, mass, mdot, rho, cs)
                count += src.size

            # Track the unique ID numbers
            det_ids = np.unique(np.append(det_ids, q_ids))
            num_det_ids = det_ids.size

        if verbose and count:
            n_one = len(snap_mergers_one)
            n_both = len(snap_mergers_both)
            print(" - - {:3d} after {}: Lines = {:.3e}, Blackholes = {:.3e}"
                  ", Mergers = {:3d}, {:3d} (one, both)".format(snap, datetime.now()-beg_all,
                                                                count, num_det_ids,
                                                                n_one, n_both))

    # Process and Save all data
    # =========================

    # Process Details Data
    # --------------------
    if verbose: print("\nDetails")
    beg = datetime.now()
    num_cut = np.count_nonzero(scale)
    if count != num_cut:
        warnings.warn("Nonzero elements '{}' does not match `count` '{}', using larger.".format(
            num_cut, count))
        num_cut = np.max([num_cut, count])

    # Trim empty elements
    id, scale, mass, mdot, rho, cs = _trim_details_arrays(num_cut, id, scale, mass, mdot, rho, cs)
    # Sort by ID, then scalefactor
    id, scale, mass, mdot, rho, cs = _sort_details(id, scale, mass, mdot, rho, cs)
    # Find unique IDs, what index they first occur at, and how many entries there are
    q_ids, q_locs, q_counts = np.unique(id, return_index=True, return_counts=True)

    num_uniq = q_ids.size
    if num_uniq != num_det_ids:
        warnings.warn("Unique ids `q_ids` size = {}, does not match `num_det_ids` = {}".format(
            num_uniq, num_det_ids))

    if verbose:
        print(" - Completed Details analysis, {} entries for {} BH, median-per {}".format(
            num_cut, num_uniq, np.median(q_counts)))

    # Write details data to HDF5
    # --------------------------
    dets_fname = GET_OUTPUT_DETAILS_FILENAME(run)
    if verbose: print(" - Saving Details to '{}'".format(dets_fname))
    # Backup previous file if it exists
    if os.path.exists(dets_fname):
        backup_fname = dets_fname + '.bak'
        shutil.move(dets_fname, backup_fname)
        warnings.warn("Moved existing file:\n\tfrom: '{}'\n\tto: '{}'".format(
            dets_fname, backup_fname))

    with h5py.File(dets_fname, 'w') as dets_h5file:
        # Create Header with meta/summary data
        dets_head = dets_h5file.create_group('Header')
        dets_head.attrs['script'] = str(__file__)
        dets_head.attrs['script_version'] = str(__version__)
        dets_head.attrs['created'] = str(datetime.now().ctime())
        dets_head.attrs['simulation'] = 'Illustris-{}'.format(run)
        dets_head.attrs['target_times'] = target_scales
        dets_head.attrs['num_entries'] = num_cut
        dets_head.attrs['num_blackholes'] = num_uniq
        dets_head.attrs['description'] = (
            "Illustris blackhole details data, combined from all of the "
            "individual blackhole (BH) details text files.  The content of the "
            "data is completely unchanged, but is downsampled.  The first and last entry "
            "for each BH is recorded, in addition to the entries closest to the target times "
            "given in the `Header/target_times` attribute.  These times are the scale factors "
            "at which subbox outputs are generated, at significantly higher temporal resolution "
            "than the normal illustris snapshots.  The details data, each in its own dataset, "
            "are the BH 'ID' number, 'time' of the entry (scale-factor), particle 'mass', "
            "accretion rate ('mdot'), local gas density ('rho'), and local sound-speed ('cs') "
            "of the gas."
        )

        # Unique BH info
        dets_h5file[DETAILS.UNIQUE_IDS] = q_ids
        dets_h5file[DETAILS.UNIQUE_FIRST] = q_locs
        dets_h5file[DETAILS.UNIQUE_NUM_PER] = q_counts

        # General Details entries
        time_dset = dets_h5file.create_dataset(DETAILS.SCALE, data=scale)
        time_dset.attrs['desc'] = 'Cosmological scale factor'
        dets_h5file.create_dataset(DETAILS.ID, data=id)
        dets_h5file.create_dataset(DETAILS.MASS, data=mass)
        dets_h5file.create_dataset(DETAILS.MDOT, data=mdot)
        dets_h5file.create_dataset(DETAILS.RHO, data=rho)
        dets_h5file.create_dataset(DETAILS.CS, data=cs)

    if verbose:
        det_fsize = os.path.getsize(dets_fname)/1024/1024
        print(" - Saved {} entries for {} unique BH after {}, filesize = {:.3e} MB".format(
            num_cut, num_uniq, datetime.now()-beg, det_fsize))

    # Process Merger-Details Data
    # ---------------------------
    if verbose: print("\nMerger-Details")
    beg = datetime.now()

    # Wrap this in a try so it doesn't break everything... just in case
    try:
        if verbose:
            # Count the nonzero elements in each column (i.e. in-bef, out-bef, out-aft)
            num_found = np.apply_along_axis(np.count_nonzero, 0, md_scale)
            tot_num_found = np.sum(num_found)
            col_names = ['In-Bef', 'Out-Bef', 'Out-Aft']
            print(" - {} Total merger-details entries found".format(tot_num_found))
            print(" - - " +
                  ", ".join("{}: {:5d}".format(cc, nn) for cc, nn in zip(col_names, num_found)))

            # Count the nonzero elements in each column (i.e. each merger)
            num_found = np.apply_along_axis(np.count_nonzero, 1, md_scale)
            print(" - Matches:")
            for ii in range(4):
                numf = np.sum(num_found == ii)
                print(" - - {}: {}/{} = {:.4f}".format(
                    ii, numf, num_mergers, numf/num_mergers))

    except Exception as err:
        warnings.warn("CAUGHT EXCEPTION (num_found) '{}'".format(str(err)))
        pass

    num_mdet_ids = 0
    try:
        all_merger_ids = set(m_ids_in).union(m_ids_out)
        all_mdet_ids = set(md_id[:, 0]).union(md_id[:, 1]).union(md_id[:, 2]) - 0
        num_merger_ids = len(all_merger_ids)
        num_mdet_ids = len(all_mdet_ids)
        extra_ids = all_mdet_ids - all_merger_ids
        missing_ids = all_merger_ids - all_mdet_ids
        if len(extra_ids):
            warnings.warn("Some ({}) merger-details IDs are not Merger IDs\n{}".format(
                len(extra_ids), extra_ids))
        if verbose:
            print("Found details for {}/{} = {:.4f} merger IDs.  {} missing.".format(
                num_mdet_ids, num_merger_ids, num_mdet_ids/num_merger_ids, len(missing_ids)))
    except Exception as err:
        warnings.warn("CAUGHT EXCEPTION (num_mdet_ids) '{}'".format(str(err)))
        pass

    # Write merger-details data to HDF5
    # ---------------------------------
    mdet_fname = GET_MERGERS_DETAILS_FILENAME(run)
    if verbose: print(" - Saving Merger-Details to '{}'".format(mdet_fname))
    # Backup previous file if it exists
    if os.path.exists(mdet_fname):
        backup_fname = mdet_fname + '.bak'
        shutil.move(mdet_fname, backup_fname)
        warnings.warn("Moved existing file:\n\tfrom: '{}'\n\tto: '{}'".format(
            mdet_fname, backup_fname))

    with h5py.File(mdet_fname, 'w') as mdet_h5file:
        # Create Header with meta/summary data
        mdet_head = mdet_h5file.create_group('Header')
        mdet_head.attrs['script'] = str(__file__)
        mdet_head.attrs['script_version'] = str(__version__)
        mdet_head.attrs['created'] = str(datetime.now().ctime())
        mdet_head.attrs['simulation'] = 'Illustris-{}'.format(run)
        mdet_head.attrs['description'] = (
            "Illustris blackhole details data specifically associated with blackhole mergers.  "
            "Each merger involves an 'in' BH and an 'out' BH, distinguished by which one "
            "dissappears after the merger ('in'), and which continues as the remaining sum of both "
            "BH ('out').  For each merger event, we try to find details entries matching both BH.  "
            "For the 'in' BH, we try to find the details entry as close to the merger time as "
            "possible (i.e. just before it).  For the 'out' BH, we try to find the details entry "
            "immediately before the merger (i.e. information about the second parent BH), and "
            "also immediately after the merger (i.e. information about the combined, child BH).  "
            "Each 'row' in the merger-details ('mdets') datasets correspond to a merger event, and "
            "then contains three 'columns' for the 'in' BH before merger, and the 'out' BH before "
            "and after merger; i.e., [in-bef, out-bef, out-aft].  Entries which could not be found "
            "are all zero.  For example, `['mass'][123, 2]` should be the mass of the 'out' BH, "
            "soon after merger, of merger event 123.  The 'time' and 'id' datasets are the "
            "time and ID number of the details entry retrieved, which can be compared to the "
            "merger data to assure that it is valid (cross-check the ID), and near the merger time."
        )

        # Details entries
        mdet_h5file.create_dataset(DETAILS.ID, data=md_id)
        time_dset = mdet_h5file.create_dataset(DETAILS.SCALE, data=md_scale)
        time_dset.attrs['desc'] = 'Cosmological scale factor'
        mdet_h5file.create_dataset(DETAILS.MASS, data=md_mass)
        mdet_h5file.create_dataset(DETAILS.MDOT, data=md_mdot)
        mdet_h5file.create_dataset(DETAILS.RHO, data=md_rho)
        mdet_h5file.create_dataset(DETAILS.CS, data=md_cs)

    if verbose:
        mdet_fsize = os.path.getsize(mdet_fname)/1024/1024
        print(" - Saved {} entries for {} unique BH after {}, filesize = {:.3e} MB".format(
            tot_num_found, num_mdet_ids, datetime.now()-beg, mdet_fsize))

        print("Done after {}".format(datetime.now()-beg_all))

    return


def _load_details_txt(txt_file):
    """Load each ASCII line from a details text file, convert to numerical arrays.
    """
    lines = txt_file.readlines()
    num_lines = len(lines)

    # Allocate storage
    id    = np.zeros(num_lines, dtype=DTYPE.ID)
    scale = np.zeros(num_lines, dtype=DTYPE.SCALAR)
    mass  = np.zeros(num_lines, dtype=DTYPE.SCALAR)
    mdot  = np.zeros(num_lines, dtype=DTYPE.SCALAR)
    rho   = np.zeros(num_lines, dtype=DTYPE.SCALAR)
    cs    = np.zeros(num_lines, dtype=DTYPE.SCALAR)

    count = 0
    # Iterate over lines, storing only those with content (should be all)
    for lin in lines:
        lin = lin.strip()
        if len(lin):
            tid, tim, mas, dot, den, tcs = _parse_illustris_details_line(lin)
            id[count] = tid
            scale[count] = tim
            mass[count] = mas
            mdot[count] = dot
            rho[count] = den
            cs[count] = tcs
            count += 1

    # Trim excess (shouldn't be needed)
    if count < num_lines:
        id, scale, mass, mdot, rho, cs = _trim_details_arrays(count, id, scale, mass, mdot, rho, cs)

    return id, scale, mass, mdot, rho, cs


def _parse_illustris_details_line(instr):
    """Parse a line from an Illustris 'blachole_details_#.txt' file (or copy of that data).

    The line is formatted (in C) as:
        "BH=%llu %g %g %g %g %g\n",
        (long long) P[n].ID, All.Time, BPP(n).BH_Mass, mdot, rho, soundspeed

    Arguments
    ---------
    instr : str

    Returns
    -------
    ID, time, mass, mdot, rho, cs

    """
    args = instr.split()
    # First element is 'BH=########', trim to just the id number
    args[0] = args[0].split("BH=")[-1]
    idn  = DTYPE.ID(args[0])
    time = DTYPE.SCALAR(args[1])
    mass = DTYPE.SCALAR(args[2])
    mdot = DTYPE.SCALAR(args[3])
    rho  = DTYPE.SCALAR(args[4])
    cs   = DTYPE.SCALAR(args[5])
    return idn, time, mass, mdot, rho, cs


def _store_dets_from_hdf5(h5_loc, h5dat, save_loc, id, scale, mass, mdot, rho, cs):
    """Copy data from hdf5 file to given arrays.
    """
    if np.iterable(h5_loc):
        _loc = list(h5_loc)
    else:
        _loc = h5_loc
    id[save_loc] = h5dat[DETAILS.ID][_loc]
    scale[save_loc] = h5dat[DETAILS.SCALE][_loc]
    mass[save_loc] = h5dat[DETAILS.MASS][_loc]
    mdot[save_loc] = h5dat[DETAILS.MDOT][_loc]
    rho[save_loc] = h5dat[DETAILS.RHO][_loc]
    cs[save_loc] = h5dat[DETAILS.CS][_loc]
    return


def _args_nearest(options, targets):
    """For a (sorted) array of `options`, find the indices of elements nearest to `targets`.

    Arguments
    ---------
    options : (N,) array of scalar
        Values from which to select.
    targets : (M,) array of scalar
        Locations to find values nearest to.

    Returns
    -------
    idx : (M,) array of int
        The index number in array `options` corresponding to the value nearest each of `targets`.
        NOTE: there may be duplicate entries in `idx`, but every `targets` will have some match.

    """
    # Find the locations in the `options` array where `targets` fit
    #    Normally `searchsorted` may return N for N elements, thus clip to N-1 for comparisons
    idx = np.searchsorted(options, targets, side="left").clip(max=options.size-1)
    # Find the distance to the nearest element below
    dist_lo = np.fabs(targets - options[idx-1])
    # Find the distance to the nearest element above
    dist_hi = np.fabs(targets - options[idx])
    # By default, `idx` chooses the upper match, if the distance to lower is better, shift down 1
    mask = (idx > 0) & ((idx == len(options)) | (dist_lo < dist_hi))
    idx = idx - mask
    return idx


def _bef_med_aft(vals, targ):
    """Find indices of array values before, at, and after that target value.
    """
    idx = np.searchsorted(vals, targ, side="left")
    lo = idx-1
    hi = idx
    if lo < 0: lo = 0
    if idx < vals.size and np.isclose(vals[idx], targ):
        me = idx
        if lo == me:
            lo = None
            hi = me + 1
        if hi == me:
            hi = me + 1
    else:
        me = None

    if hi > vals.size-1: hi = vals.size-1

    return lo, me, hi


def _bef_aft_scales_in(id_loc, mtime, h5file):
    """Given a unique details BH, find the details entries nearest the given merger time.

    Arguments
    ---------
    id_loc : int
        Index in the `unique` BH arrays corresponding to the BH of interest.
    mtime : scalar
        Time (scale-factor) of the target merger.
    h5file : `h5py.File` handle containing details data for one snapshot interval.

    Returns
    -------
    bef : int or 'None'
        Index in the BH details arrays corresponding to the latest entry for the target BH, still
        occuring before the time of merger.  'None' if no match found.
    aft : int of 'None'
        Index in the BH details arrays corresponding to the first entry for the target BH,
        occuring after the time of merger.  'None' if no match found.
    """
    # Index of the first details entry for this BH
    i0 = h5file[DETAILS.UNIQUE_FIRST][id_loc]
    # Index of the last details entry (+1) for this BH
    i1 = i0 + h5file[DETAILS.UNIQUE_NUM_PER][id_loc]
    # All scale-factors or details entries for this BH
    dscales = h5file[DETAILS.SCALE][i0:i1]
    # Find scales nearest before ('lo'), exatly matching ('me'), and nearest after merger ('hi')
    #     each `lo` and `med` will be 'None' if no such match is found
    #     `hi` is always a number, but may be invalid (i.e. not actually after merger)
    lo, me, hi = _bef_med_aft(dscales, mtime)
    # For the 'in' BH, if there is an entry at time of merger, must be before
    if me is not None:
        bef = me
        aft = hi
    else:
        bef = lo
        aft = hi

    # Make sure results are consistent
    #    `bef` and `aft` can be before/after `mtime` if `mtime` ourside range of details entries
    if bef is not None:
        if dscales[bef] > mtime:
            bef = None
        else:
            bef += i0
    if aft is not None:
        if dscales[aft] < mtime:
            aft = None
        else:
            aft += i0

    return bef, aft


def _bef_aft_scales_out(id_loc, m_mass, mtime, h5file):
    """Given a unique details BH, find the details entries nearest the given merger time.

    Arguments
    ---------
    id_loc : int
        Index in the `unique` BH arrays corresponding to the BH of interest.
    mtime : scalar
        Time (scale-factor) of the target merger.
    h5file : `h5py.File` handle containing details data for one snapshot interval.

    Returns
    -------
    bef : int or 'None'
        Index in the BH details arrays corresponding to the latest entry for the target BH, still
        occuring before the time of merger.  'None' if no match found.
    aft : int of 'None'
        Index in the BH details arrays corresponding to the first entry for the target BH,
        occuring after the time of merger.  'None' if no match found.
    """
    # Index of the first details entry for this BH
    i0 = h5file[DETAILS.UNIQUE_FIRST][id_loc]
    # Index of the last details entry (+1) for this BH
    i1 = i0 + h5file[DETAILS.UNIQUE_NUM_PER][id_loc]
    # All scale-factors or details entries for this BH
    dscales = h5file[DETAILS.SCALE][i0:i1]
    d_mass = h5file[DETAILS.MASS][i0:i1]
    # Find scales nearest before ('lo'), exatly matching ('me'), and nearest after merger ('hi')
    #     each `lo` and `med` will be 'None' if no such match is found
    #     `hi` is always a number, but may be invalid (i.e. not actually after merger)
    lo, me, hi = _bef_med_aft(dscales, mtime)
    # If all values are returned, use masses to try to determine best 'bef' and 'aft'
    if lo is not None and me is not None:
        # Find the change in mass between entries, compared to the 'in' BH mass
        delta_mass = np.diff(d_mass[[lo, me, hi]]) - m_mass
        # If only the *first* change is larger than merger mass, thats when merger occurs
        if delta_mass[0] >= 0.0 and delta_mass[1] < 0.0:
            bef = lo
            aft = me
            match_type = _MTYPE.M_BEF_D
        # If only the *second* change is larger than merger mass, thats when merger occurs
        elif delta_mass[0] < 0.0 and delta_mass[1] >= 0.0:
            bef = me
            aft = hi
            match_type = _MTYPE.D_BEF_M
        # If either change is enough, guess that `me` is after merger (tends to be the case)
        elif delta_mass[0] >= 0.0 and delta_mass[1] >= 0.0:
            bef = lo
            aft = me
            match_type = _MTYPE.UNKNOWN
        # If neither change is enough... something is wrong, set both `bef` and `aft` to None.
        elif delta_mass[0] < 0.0 and delta_mass[1] < 0.0:
            match_type = _MTYPE.CONFLICT
            return None, None, match_type
        else:
            raise RuntimeError("WHATS HAPPENING?!?!  delta_mass = {}".format(delta_mass))
    # If only `lo` and `hi` are found
    elif lo is not None:
        # Still make sure that mass change is consistent with 'in' BH mass
        delta_mass = np.diff(d_mass[[lo, hi]]) - m_mass
        if delta_mass > 0.0:
            bef = lo
            aft = hi
            match_type = _MTYPE.ASYNC
        # Set both to 'None' if its in conflict
        else:
            match_type = _MTYPE.ASYNC_CONFLICT
            return None, None, match_type
    # If only `hi` is found, use that
    else:
        bef = None
        aft = hi
        match_type = _MTYPE.PARTIAL

    # Both `bef` and `aft` can be before/after `mtime` if `mtime` ourside range
    if bef is not None:
        if dscales[bef] > mtime:
            bef = None
        else:
            bef += i0
    if aft is not None:
        if dscales[aft] < mtime:
            aft = None
        else:
            aft += i0

    return bef, aft, match_type


def _unique_locs_for_mergers(mscales, mids, unique_ids, first, last, pad=0.05):
    """Find which merger numbers match which unique details ID numbers for this snapshot.

    If a merger occurs near a snapshot, the corresponding details entries may stagger two files.
    the `pad` parameter tries to accommodate this by searching slightly beyond the target snapshot.

    Arguments
    ---------
    mscales : (N,) array of scalar
        Times (scale-factors) of 'N' merger events.
    mids : (N,) array of uint
        BH ID numbers for *either* the 'in' or the 'out' BH for 'N' merger events.
    unique_ids : (M,) array of uint
        Unique BH ID numbers for details entries in this snapshot, for 'M' details entries.
    first : scalar
        Time (scale-factor) of first detail entry for this file.
    last : scalar
        Time (scale-factor) of last detail entry for this file.
    pad : scalar {0.0, 1.0}
        The amount of time (scale-factor) outside of [first, last] for which to match with merger
        events.  Fraction of (last-first).

    Returns
    -------
    match_unique_locs : (N,) array of int
        Index numbers for the `unique_ids` array, for details ID numbers matching merger ID numbers.
    match_mergers : (N,) array of int
        Index numbers for the `mids` array, for merger ID numbers matching details-unique-IDs.
    m_inds_snap : (M,) array of int
        Index numbers of mergers (`mscales`) which occur during this snapshot interval, specifically
        between [`first`, `last`] scales (with `pad` additional padding).

    """
    # Find mergers occuring in (or near: `pad`) this snapshot
    _pad = pad * (last - first)
    m_inds_snap = np.where((mscales + _pad >= first) & (mscales - _pad <= last))[0]
    # Return empty if no matches
    if m_inds_snap.size == 0:
        return [], [], m_inds_snap

    # Get the IDs for mergers within this time interval
    m_ids = mids[list(m_inds_snap)]
    # Use sorted ID arrays
    sort_m_ids = np.argsort(m_ids)

    # Find details entries which match merger IDs
    # Find the locations in details unique IDs to place 'in' merger IDs
    d_locs_m_ids = np.searchsorted(unique_ids, m_ids[sort_m_ids]).clip(max=unique_ids.size-1)
    # Find which of those locations correspond to actual matches
    match = np.where(unique_ids[:][list(d_locs_m_ids)] == m_ids[sort_m_ids])[0]

    # Retrieve merger numbers with IDs matching details-unique-ID entries
    match_mergers = m_inds_snap[sort_m_ids[match]]
    # Retrieve details numbers for Details-unique-ID entries matching merger IDs
    match_unique_locs = d_locs_m_ids[list(match)]
    return match_unique_locs, match_mergers, m_inds_snap


def _sort_details(id, scale, mass, mdot, rho, cs):
    """Sort details data first by BH ID number, then by time (scale-factor) of detail entry.
    """
    inds = np.lexsort((scale, id))
    id = id[inds]
    scale = scale[inds]
    mass = mass[inds]
    mdot = mdot[inds]
    rho = rho[inds]
    cs = cs[inds]
    return id, scale, mass, mdot, rho, cs


def _trim_details_arrays(trim, id, scale, mass, mdot, rho, cs):
    """Remove excess elements in the given arrrays.
    """
    if trim >= id.size:
        return id, scale, mass, mdot, rho, cs
    id    = id[:trim]
    scale = scale[:trim]
    mass  = mass[:trim]
    mdot  = mdot[:trim]
    rho   = rho[:trim]
    cs    = cs[:trim]
    return id, scale, mass, mdot, rho, cs


'''
def loadBHDetails(run, snap, loadsave=True, verbose=True):
    """Load Blackhole Details dictionary for the given snapshot.

    If the file does not already exist, it is recreated from the temporary ASCII files, or directly
    from the raw illustris ASCII files as needed.

    Arguments
    ---------
        run     : <int>, illustris simulation number {1, 3}
        snap    : <int>, illustris snapshot number {0, 135}
        loadsave <bool> :
        verbose  <bool> : print verbose output

    Returns
    -------
        dets    : <dict>, BHDetails dictionary object for target snapshot

    """
    if verbose: print(" - - BHDetails.loadBHDetails")

    detsName = BHConstants.GET_DETAILS_SAVE_FILENAME(run, snap, VERSION)

    # Load Existing Save File
    if(loadsave):
        if verbose: print((" - - - Loading details from '%s'" % (detsName)))
        if(os.path.exists(detsName)):
            dets = zio.npzToDict(detsName)
        else:
            loadsave = False
            warnStr = "%s does not exist!" % (detsName)
            warnings.warn(warnStr, RuntimeWarning)

    # If file does not exist, or is wrong version, recreate it
    if(not loadsave):
        if verbose: print(" - - - Re-converting details")
        # Convert ASCII to NPZ
        saveFile = _convertDetailsASCIItoNPZ_snapshot(run, snap, loadsave=True, verbose=verbose)
        # Load details from newly created save file
        dets = zio.npzToDict(saveFile)

    return dets
'''

'''
def _getPrecision(args):
    """Estimate the precision needed to differenciate between elements of an array
    """
    diffs = np.fabs(np.diff(sorted(args)))
    inds  = np.nonzero(diffs)
    if len(inds) > 0:
        minDiff = np.min(diffs[inds])
    else:
        minDiff = np.power(10.0, _DEF_SCALE_PRECISION)
    order = int(np.log10(0.49*minDiff))
    return order
'''

'''
def get_out_dir(run, output_dir=None):
    """
    /n/home00/lkelley/ghernquistfs1/illustris/data/%s/blackhole/details/
    """
    if output_dir is None:
        output_dir = os.path.join(_PROCESSED_DIR % (_ILLUSTRIS_RUN_NAMES[run]), _DEF_OUTPUT_DIR, '')
    return output_dir
'''
