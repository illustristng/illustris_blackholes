"""Module to handle Illustris BH Merger Files.

This module is an interface to the 'blackhole_mergers_<#>.txt' files produced
by Illustris.  Raw Illustris files are only used to initially load data, then
an intermediate numpy npz-file is produced to store a dictionary of merger data
for easier access in all future calls.  Executing the `main()` routine will
prepare the intermediate file, as will calls to the `loadMergers()` function -
if the intermediate file hasn't already been loaded.

The `mergers` are represented as a dictionary object with keys given by the
variables `MERGERS_*`, e.g. `MERGERS_NUM` is the key for the number of mergers.

Internal Parameters
-------------------

Functions
---------


Mergers Dictionary
------------------

Examples
--------

Notes
-----
 - 'Raw Mergers' : these are mergers directly from the illustris files with NO modifications or
                   filtering of any kind.



   The underlying data is in the illustris bh merger files, 'blackhole_mergers_<#>.txt', which are
   processed by `_loadMergersFromIllustris()`.  Each line of the input files is processed by
   `_parse_line_merger()` which returns the redshift ('time') of the merger, and the IDs and masses
   of each BH.  Each `merger` is sorted by time (redshift) in `_importMergers()` and placed in a
   `dict` of all results.  This merger dictionary is saved to a 'raw' savefile whose name is given
   by `savedMergers_rawFilename()`.
   The method `processMergers()` not only loads the merger objects, but also creates mappings of
   mergers to the snapshots nearest where they occur (``mapM2S`) and visa-versa (``mapS2M``); as
   well as mergers which take place exactly during a snapshot iteration (``ontop``).  These three
   maps are included in the merger dictionary.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

from datetime import datetime
import h5py
import numpy as np
import os
import shutil
import warnings

from illbh import constants
from illbh.constants import DETAILS, DTYPE, MERGERS, META

# Physical Constants in CGS
SPLC = 2.99792458e+10
NWTG = 6.67384000e-08
MPRT = 1.67262178e-24
SIGMA_T = 6.65245873e-25
MSOL = 1.98910000e+33
KPC = 3.08567758e+21
HPAR = 6.93200000e-01
YR = 3.15576000e+07

__version__ = '1.0.1'

# Size of initial Details arrays allocation; and 'chunk' size that it is increased by as needed
_MERGS_BUF_SIZE = int(1e5)


def combine_raw_merger_files(run, verbose=True, output_dir=None, input_dir=None):
    """Concatenate the contents of a set of input files into a single output file.

    Arguments
    ---------
    in_fnames : iterable<str>, list of input file names
    out_fname : <str>, output file name
    verbose : <bool> (optional=_VERBOSE), print verbose output

    Returns

    """
    beg = datetime.now()
    git_vers = constants._get_git()
    # Load scalefactor (time) for each snapshot (rounded)
    # Raw illustris merger filenames
    # in_filenames = GET_ILLUSTRIS_BH_MERGERS_FILENAMES(run)
    num_snaps, in_filenames = constants.get_illustris_metadata(
        run, [META.NUM_SNAPS, META.MERGERS_FILENAMES])

    # Filename for combined mergers file, raw = unfiltered
    out_raw_fname = constants.GET_MERGERS_COMBINED_FILENAME(
        run, filtered=False, type='txt', output_dir=output_dir)
    if verbose:
        print(" - - Writing   raw    combined mergers to '{}'".format(out_raw_fname))
    # Make backups of existing output files
    constants._backup_exists(out_raw_fname, verbose=verbose)

    # Storage for details entries
    #    Make initial guess for how many entries there will be (resized later as-needed)
    scale = np.zeros(_MERGS_BUF_SIZE, dtype=DTYPE.SCALAR)
    id_in = np.zeros(_MERGS_BUF_SIZE, dtype=DTYPE.ID)
    id_out = np.zeros(_MERGS_BUF_SIZE, dtype=DTYPE.ID)
    mass_in = np.zeros(_MERGS_BUF_SIZE, dtype=DTYPE.SCALAR)
    mass_out = np.zeros(_MERGS_BUF_SIZE, dtype=DTYPE.SCALAR)

    nums_in = len(in_filenames)
    interv = int(np.floor(nums_in/10))
    count_raw = 0
    count = 0
    num_deleted = 0
    # Open output file for writing
    with open(out_raw_fname, 'w') as out_raw:
        # Iterate over input files
        if verbose: print(" - - Iterating over {} input merger files".format(nums_in))
        for ii, in_name in enumerate(in_filenames):
            last_time = 0.0  # Stores the previous scale-factor (time)

            file_count = count  # Store number of lines `count` before this file
            # Iterate over lines in input file
            for mline in open(in_name, 'r'):
                # Write *all* lines to 'raw' output file
                out_raw.write(mline)
                count_raw += 1

                # Parse values, convert to numerical data types
                time, out_id, out_mass, in_id, in_mass = _parse_line_merger(mline)

                # If times go backwards, simulation was restarted.  Erase (previous) overlap segment
                if time < last_time:
                    # Find lines matching or following the newest read scale-factor
                    #    Only search elements from *this* file
                    time_mask = (time < scale[file_count:]) | np.isclose(scale[file_count:], time)
                    bads = np.where(time_mask)[0] + file_count
                    # warnings.warn("Merger overwrite detected.  Time: {}, last: {}.\n\t"
                    #               "file_count: {}, count: {}, bads: {}".format(
                    #                   time, last_time, file_count, count, bads))
                    # Delete these lines
                    if bads.size:
                        scale = np.delete(scale, bads)
                        id_in = np.delete(id_in, bads)
                        id_out = np.delete(id_out, bads)
                        mass_in = np.delete(mass_in, bads)
                        mass_out = np.delete(mass_out, bads)
                        num_deleted += bads.size
                        count -= bads.size

                # Store filtered values
                scale[count] = time
                id_in[count] = in_id
                id_out[count] = out_id
                mass_in[count] = in_mass
                mass_out[count] = out_mass
                last_time = time
                count += 1

                # If we reach edge of storage arrays, increase size
                if count >= scale.size:
                    scale = constants._zero_pad_end(scale, _MERGS_BUF_SIZE)
                    id_in = constants._zero_pad_end(id_in, _MERGS_BUF_SIZE)
                    id_out = constants._zero_pad_end(id_out, _MERGS_BUF_SIZE)
                    mass_in = constants._zero_pad_end(mass_in, _MERGS_BUF_SIZE)
                    mass_out = constants._zero_pad_end(mass_out, _MERGS_BUF_SIZE)

            if verbose and ii % interv == 0:
                dur = datetime.now()-beg
                print("\t{:5d}/{} = {:.4f} after {}.  Raw: {:5d}; Filtered: {:5d}, Deleted: {:5d}"
                      "".format(ii, nums_in, ii/nums_in, dur, count_raw, count, num_deleted))

    if verbose:
        print(" - - Finished Raw: {}, Filtered: {} lines; after {}".format(
            count_raw, count, datetime.now()-beg))

    # Process and Save Filtered Merger Data to HDF5
    # ---------------------------------------------

    # Trim excess in filtered storage arrays
    scale = scale[:count]
    id_in = id_in[:count]
    id_out = id_out[:count]
    mass_in = mass_in[:count]
    mass_out = mass_out[:count]

    # Find indices which sort by time
    inds = np.argsort(scale)
    # Use indices to reorder arrays
    scale   = scale[inds]
    id_in    = id_in[inds]
    id_out   = id_out[inds]
    mass_in  = mass_in[inds]
    mass_out = mass_out[inds]

    # Find all unique BH IDs
    all_ids = np.append(id_in, id_out)
    all_ids = np.unique(all_ids)
    if verbose:
        print(" - - Mergers: {}, Unique BH: {}".format(inds.size, all_ids.size))

    # Write Raw data to hdf5 file
    # Filename for combined mergers file, filtered
    out_filtered_fname = constants.GET_MERGERS_COMBINED_FILENAME(
        run, filtered=True, type='hdf5', output_dir=output_dir)
    print(" - - Writing filtered combined mergers to '{}'".format(out_filtered_fname))
    constants._backup_exists(out_filtered_fname, verbose=verbose)
    with h5py.File(out_filtered_fname, 'w') as h5file:
        # Add metadata in "Header" dataset
        head = h5file.create_group('Header')
        head.attrs['script'] = str(__file__)
        head.attrs['script_version'] = str(__version__)
        head.attrs['git_version'] = str(git_vers)
        head.attrs['created'] = str(datetime.now().ctime())
        head.attrs['simulation'] = '{}'.format(run)
        head.attrs['description'] = (
            "Illustris blackhole merger data, combined from all of the "
            "individual blackhole (BH) merger text files.  The content of the "
            "data is completely unchanged.  Each merger involves two BH, the 'out' "
            "BH which persists after the merger, and the 'in' BH which is consumed.  "
            "NOTE: the mass of the 'out' BH is incorrect in this data.  The values "
            "given correspond to the total cell (dynamical) mass, instead of the BH "
            "mass itself."
        )

        # Save unique ID numbers
        h5file[MERGERS.UNIQUE] = all_ids

        # Add merger data
        time_dset = h5file.create_dataset(MERGERS.SCALE, data=scale)
        time_dset.attrs['units'] = 'Cosmological scale factor'
        h5file.create_dataset(MERGERS.ID_IN, data=id_in)
        h5file.create_dataset(MERGERS.ID_OUT, data=id_out)
        h5file.create_dataset(MERGERS.MASS_IN, data=mass_in)
        h5file.create_dataset(MERGERS.MASS_OUT, data=mass_out)

    if verbose:
        fsize = os.path.getsize(out_filtered_fname)/1024/1024
        print(" - - Saved to '{}', Size: '{:.3e}' MB".format(out_filtered_fname, fsize))

    return count_raw, count


def combine_mergers_with_details(run, verbose=True, output_dir=None):
    """
    """
    print(" - Combining mergers with details.")
    # Load scalefactor (time) for each snapshot (rounded)
    num_snaps, snap_scales = constants.get_illustris_metadata(
        run, [META.NUM_SNAPS, META.SNAP_TIMES])
    snap_scales = np.around(snap_scales, -constants._DEF_SCALE_PRECISION)

    git_vers = constants._get_git()
    beg_all = datetime.now()
    mergers_in_fname = constants.GET_MERGERS_COMBINED_FILENAME(
        run, filtered=True, type='hdf5', output_dir=output_dir)
    mdets_in_fname = constants.GET_MERGERS_DETAILS_FILENAME(run, output_dir=output_dir)
    details_in_fname = constants.GET_PUBLIC_DETAILS_FILENAME(run, output_dir=output_dir)

    # Load scalefactor (time) for each snapshot (rounded)
    # snap_scales = constants.GET_SNAPSHOT_SCALES()

    # Load Everything
    # ---------------

    # All details data
    if verbose:
        print(" - Loading details data from '{}'".format(details_in_fname))
    beg = datetime.now()
    with h5py.File(details_in_fname, 'r') as details_in:
        det_scale = details_in[DETAILS.SCALE][:]
        det_id = details_in[DETAILS.ID][:]
        det_q_id = details_in[DETAILS.UNIQUE_IDS][:]
        det_q_first = details_in[DETAILS.UNIQUE_FIRST][:]

    num_det_unique = det_q_id.size
    if verbose:
        print(" - - Loaded {} entries, from {} unique BH; after {}".format(
            det_scale.size, num_det_unique, datetime.now()-beg))

    # Mergers Data
    if verbose:
        print(" - Loading basic mergers data from '{}'".format(mergers_in_fname))
    beg = datetime.now()
    with h5py.File(mergers_in_fname, 'r') as mergers_in:
        m_scale = mergers_in[MERGERS.SCALE][:]
        m_id_in = mergers_in[MERGERS.ID_IN][:]
        m_id_out = mergers_in[MERGERS.ID_OUT][:]
        m_mass_in = mergers_in[MERGERS.MASS_IN][:]
        # m_mass_out = mergers_in[MERGERS.MASS_OUT][:]
        m_uniq = mergers_in[MERGERS.UNIQUE][:]

    if verbose:
        print(" - - Loaded {} mergers, {} unique IDs; after {}".format(
            m_scale.size, m_uniq.size, datetime.now()-beg))

    # Mergers Details Data
    if verbose:
        print(" - Loading mergers-details data from '{}'".format(mdets_in_fname))
    beg = datetime.now()
    with h5py.File(mdets_in_fname, 'r') as mdets_in:
        d_id = mdets_in[DETAILS.ID][:]
        d_scale = mdets_in[DETAILS.SCALE][:]
        d_mass = mdets_in[DETAILS.MASS][:]
        d_mdot = mdets_in[DETAILS.MDOT][:]
        d_rho = mdets_in[DETAILS.RHO][:]
        d_cs = mdets_in[DETAILS.CS][:]

    if verbose:
        num_in_bef = np.count_nonzero(d_scale[:, 0])
        num_out_bef = np.count_nonzero(d_scale[:, 1])
        num_out_aft = np.count_nonzero(d_scale[:, 2])
        print(" - - Loaded {} in-bef, {} out-bef, {} out-aft details; after {}".format(
            num_in_bef, num_out_bef, num_out_aft, datetime.now()-beg))

    # Analyze Data
    # ------------
    mdet_size = d_scale.shape[0]
    num_mergers = m_scale.size
    if mdet_size != num_mergers:
        raise ValueError("Mergers size {} does not match merger-details {}".format(
            num_mergers, mdet_size))

    # Correct the merger 'out' mass
    mass_out = np.zeros(num_mergers)

    # Get lookback times
    d_lbt = np.zeros([num_mergers, 3])
    m_lbt = constants.scale_to_age_flat(m_scale)
    for ii in range(3):
        inds = np.where(d_scale[:, ii] > 0.0)[0]
        d_lbt[inds, ii] = constants.scale_to_age_flat(d_scale[inds, ii])

    valid = _check_mergers_for_repeats(
        m_scale, m_id_in, m_id_out, det_q_id, det_q_first, det_scale, det_id)

    # Merger-Tree Storage
    m_next = -1*np.ones(num_mergers, dtype=int)
    m_prev_in = -1*np.ones(num_mergers, dtype=int)
    m_prev_out = -1*np.ones(num_mergers, dtype=int)
    # Out-Mass Fixing Stats
    num_bef_mass = 0
    num_aft_mass = 0
    num_nxt_mass = [0, 0, 0]
    num_corr_succ = 0
    num_corr_fail = 0
    for mm in range(num_mergers):
        if not valid[mm]:
            continue
        mtime = m_lbt[mm]
        dtime = d_lbt[mm]
        dmass = d_mass[mm]

        # Build Merger Tree
        # -----------------
        this_out = m_id_out[mm]
        # See if the 'out' BH of this merger becomes the 'in' BH of another merger
        next_in = np.where((this_out == m_id_in[mm+1:]) & valid[mm+1:])[0] + mm+1
        # See if the 'out' BH from this merger become the 'out' BH from another one
        next_out = np.where((this_out == m_id_out[mm+1:]) & valid[mm+1:])[0] + mm+1

        if next_in.size:
            next_in = next_in[0]
        else:
            next_in = 0

        if next_out.size:
            next_out = next_out[0]
        else:
            next_out = 0

        # If both next 'in' and 'out' found, choose earlier
        if next_in > 0 and next_out > 0:
            if m_scale[next_in] > m_scale[next_out]:
                next_in = 0
            else:
                next_out = 0

        if next_in > 0:
            # Make sure IDs match
            if m_id_in[next_in] != this_out:
                raise ValueError("Merger {}: next_in = {}, this_out = {}".format(
                    mm, m_id_in[next_in], this_out))
            # The next merger for this ('out') BH is `next_in`
            #    Shouldnt have been set before
            if m_next[mm] >= 0:
                raise ValueError("`mm` = {}, `next_in` = {}, m_next = `{}`".format(
                    mm, next_in, m_next[mm]))
            m_next[mm] = next_in
            # The previous merger for the `next_in`--'in' BH, is this one
            #    Shouldnt have been set before
            if m_prev_in[next_in] >= 0:
                raise ValueError("`mm` = {}, `next_in` = {}, m_prev_in = `{}`".format(
                    mm, next_in, m_prev_in[next_in]))

            m_prev_in[next_in] = mm

        if next_out > 0:
            # Make sure IDs match
            if m_id_out[next_out] != this_out:
                raise ValueError("Merger {}: next_out = {}, this_out = {}".format(
                    mm, m_id_out[next_out], this_out))
            # The next merger for this ('out') BH is `next_in`
            #    Shouldnt have been set before
            if m_next[mm] >= 0:
                raise ValueError("`mm` = {}, `next_out` = {}, m_next = `{}`".format(
                    mm, next_out, m_next[mm]))
            m_next[mm] = next_out
            # The previous merger for the `next_out`--'out' BH, is this one
            #    Shouldnt have been set before
            if m_prev_out[next_out] >= 0:
                raise ValueError("`mm` = {}, `next_out` = {}, m_prev_out = `{}`".format(
                    mm, next_out, m_prev_out[next_out]))
            m_prev_out[next_out] = mm

        # Fix incorrect Merger 'out' Masses
        # ---------------------------------

        # Use 'out-bef' mass
        bef_mass = 0.0
        if dmass[1] > 0.0:
            dm = _mass_change_frac(dtime[1], mtime)
            # Make sure difference in time is only enough for a doubling of mass (~2x uncertainty)
            if dm <= 1.0:
                bef_mass = dmass[1]
                num_bef_mass += 1

        # Use 'out-aft' mass
        aft_mass = 0.0
        if dmass[2] > 0.0:
            dm = _mass_change_frac(dtime[2], mtime)
            # Make sure difference in time is only enough for a doubling of mass (~2x uncertainty)
            if dm <= 1.0:
                aft_mass = dmass[2] - m_mass_in[mm]
                num_aft_mass += 1

        # If neither details entry exists, look for a subsequent merger that's soon
        nxt_mass = 0.0
        # if np.isclose(aft_mass, 0.0) and np.isclose(bef_mass, 0.0):
        # Look for a next merger (merger number)
        nxt = m_next[mm]
        # If next merger and out ==> in BH
        if nxt >= 0 and m_id_in[nxt] == this_out:
            dm = _mass_change_frac(m_lbt[nxt], mtime)
            if dm <= 1.0:
                nxt_mass = m_mass_in[nxt] - m_mass_in[mm]
                num_nxt_mass[0] += 1

        # If next merger and out ==> out BH
        elif nxt >= 0 and m_id_out[nxt] == this_out:
            # If there is an 'in' Detail entry
            if d_id[nxt, 1] == this_out:
                dm = _mass_change_frac(d_lbt[nxt, 1], mtime)
                if dm <= 1.0:
                    nxt_mass = d_mass[nxt, 1] - m_mass_in[mm]
                    num_nxt_mass[1] += 1
            # If there is an 'out' Detail entry
            if d_id[nxt, 2] == this_out:
                dm = _mass_change_frac(d_lbt[nxt, 2], mtime)
                if dm <= 1.0:
                    nxt_mass = d_mass[nxt, 2] - m_mass_in[mm]
                    num_nxt_mass[2] += 1

        # if np.any(np.array([bef_mass, aft_mass, nxt_mass]) > 0.0):
        mass_out[mm] = _compare_select_correction_mass([bef_mass, aft_mass, nxt_mass])
        if mass_out[mm] > 0.0:
            num_corr_succ += 1
        else:
            num_corr_fail += 1

    if verbose:
        print(" - Num next: {}, Prev: {} (in: {}, out: {})".format(
            np.count_nonzero(m_next > 0), np.count_nonzero((m_prev_in > 0) | (m_prev_out > 0)),
            np.count_nonzero(m_prev_in > 0), np.count_nonzero(m_prev_out > 0)))
        print(" - Out masses - Yes: {}, No: {}".format(
            np.count_nonzero(mass_out), np.sum(np.isclose(mass_out, 0.0))))
        print(" - - Bef: {}, Aft: {}, Nxt: {} ({}, {}, {})".format(
            num_bef_mass, num_aft_mass, np.sum(num_nxt_mass), *num_nxt_mass))

        print(" - - Correction - Success: {}, Failure: {}".format(num_corr_succ, num_corr_fail))

    val = np.where(valid & (mass_out > 0.0))[0]

    # Prepare final output data
    # Merger Data
    scale = m_scale[val]
    id_in = m_id_in[val]
    id_out = m_id_out[val]
    mass_in = m_mass_in[val]
    mass_out = mass_out[val]
    # Merger-Details Data
    md_scale = d_scale[val, :]
    md_mass = d_mass[val, :]
    md_mdot = d_mdot[val, :]
    md_rho = d_rho[val, :]
    md_cs = d_cs[val, :]
    # Merger Tree
    m_next = m_next[val]
    m_prev_in = m_prev_in[val]
    m_prev_out = m_prev_out[val]
    # Unique
    mq_ids = np.array(list(set(id_in).union(id_out)))
    num_mergers = scale.size
    num_unique = mq_ids.size

    # Determine which snapshot each merger 'belongs' in
    m_snaps = np.digitize(scale, snap_scales, right=True)

    # Produce Public Output File
    mergers_out_fname = constants.GET_PUBLIC_MERGERS_FILENAME(run, output_dir=output_dir)
    if verbose:
        print(" - Saving public merger output data to '{}'".format(mergers_out_fname))
    # Backup previous file if it exists
    if os.path.exists(mergers_out_fname):
        backup_fname = mergers_out_fname + '.bak'
        shutil.move(mergers_out_fname, backup_fname)
        warnings.warn("Moved existing file:\n\tfrom: '{}'\n\tto: '{}'".format(
            mergers_out_fname, backup_fname))
    beg = datetime.now()
    with h5py.File(mergers_out_fname, 'w') as mergers_out:
        # Add metadata in "Header" dataset
        head = mergers_out.create_group('Header')
        head.attrs['script'] = str(__file__)
        head.attrs['script_version'] = str(__version__)
        head.attrs['git_version'] = str(git_vers)
        head.attrs['created'] = str(datetime.now().ctime())
        head.attrs['simulation'] = '{}'.format(run)
        head.attrs['num_mergers'] = num_mergers
        head.attrs['num_blackholes'] = num_unique
        head.attrs['description'] = (
            "Illustris blackhole merger data, combined from all of the "
            "individual blackhole (BH) merger text files.  The content of the "
            "data is completely unchanged.  Each merger involves two BH, the 'out' "
            "BH which persists after the merger, and the 'in' BH which is consumed.  "
            "NOTE: the mass of the 'out' BH is incorrect in this data.  The values "
            "given correspond to the total cell (dynamical) mass, instead of the BH "
            "mass itself."
        )

        # Add merger data
        time_dset = mergers_out.create_dataset(MERGERS.SCALE, data=scale)
        time_dset.attrs['units'] = 'Cosmological scale factor'
        mergers_out[MERGERS.ID_IN] = id_in
        mergers_out[MERGERS.ID_OUT] = id_out
        mergers_out[MERGERS.MASS_IN] = mass_in
        mergers_out[MERGERS.MASS_OUT] = mass_out

        # Save mapping to snapshots
        mergers_out[MERGERS.SNAPSHOT] = m_snaps

        # Save unique ID numbers
        mergers_out[MERGERS.UNIQUE] = mq_ids

        # Save Merger Tree Data
        tree = mergers_out.create_group(MERGERS._TREE)
        mergers_out[MERGERS.NEXT] = m_next
        mergers_out[MERGERS.PREV_IN] = m_prev_in
        mergers_out[MERGERS.PREV_OUT] = m_prev_out
        tree.attrs['description'] = (
            """Illustris black hole merger-tree data.  Three datasets are included, each an array of
            integers with shape (N,) where `N` is the number of mergers (stored in
            `Header.attrs['num_mergers']`).
            For each merger:
            -   'tree/next'    : the index of the next merger that this remnant is involved in.
            -   'tree/prev_in' : the index of the previous merger that the 'in' BH took part in.
            -   'tree/prev_out': the index of the previous merger that the 'out' BH took part in.
            If any of these values doesnt exist, the array element is `-1` --- meaning there is no
            next (or prev) merger.
            """
        )

        # Save Merger-Details Data
        details = mergers_out.create_group(MERGERS._DETAILS)
        mergers_out[MERGERS.DET_SCALE] = md_scale
        mergers_out[MERGERS.DET_MASS] = md_mass
        mergers_out[MERGERS.DET_MDOT] = md_mdot
        mergers_out[MERGERS.DET_RHO] = md_rho
        mergers_out[MERGERS.DET_CS] = md_cs
        details.attrs['description'] = (
            """Illustris black hole details for merger black holes.
            """
        )

    mrg_fsize = os.path.getsize(mergers_out_fname)/1024/1024
    print(" - Saved {} mergers for {} unique BH after {}, filesize = {:.3e} MB\n{}".format(
        num_mergers, num_unique, datetime.now()-beg_all, mrg_fsize, mergers_out_fname))

    return


def _parse_line_merger(line):
    """Process quantities from each line of the illustris merger files.

    See 'http://www.illustris-project.org/w/index.php/Blackhole_Files' for
    details regarding the illustris BH file structure.

    The format of each line is:
        "PROC-NUM  TIME  ID1  MASS1  ID0  MASS0"
        where
            '1' corresponds to the 'out'/'accretor'/surviving BH
            '0' corresponds to the 'in' /'accreted'/eliminated BH
        NOTE: that `MASS1` (`out_mass`) is INCORRECT in illustris (dynamical mass, instead of BH)

    Returns
    -------
    time     : scalar, redshift of merger
    out_id   : long, id number of `out` BH
    out_mass : scalar, mass of `out` BH in simulation units (INCORRECT VALUE)
    in_id    : long, id number of `in` BH
    in_mass  : scalar, mass of `in` BH in simulation units

    """
    strs     = line.split()
    # Convert to proper types
    time     = DTYPE.SCALAR(strs[1])
    out_id   = DTYPE.ID(strs[2])
    out_mass = DTYPE.SCALAR(strs[3])
    in_id    = DTYPE.ID(strs[4])
    in_mass  = DTYPE.SCALAR(strs[5])
    return time, out_id, out_mass, in_id, in_mass


def _eddington_mdot(mass, eps=0.1):
    """Eddington Accretion rate, $\dot{M}_{Edd} = L_{Edd}/\epsilon c^2$.^

    Arguments
    ---------
    mass : array_like of scalar
        BH Mass.
    eps : array_like of scalar
        Efficiency parameter.

    Returns
    -------
    mdot : array_like of scalar
        Eddington accretion rate.

    """
    edd_lum = 4.0*np.pi*SPLC*NWTG*mass*MPRT/SIGMA_T
    mdot = edd_lum/(eps*np.square(SPLC))
    return mdot


def _mass_change_frac(t1, t2):
    """Calculate the possible change in mass accreting at eddington over the given time interval.
    """
    # Get eddington accretion rate per unit mass (hence 1.0)
    mdot_edd = _eddington_mdot(1.0)
    # Get time-interval
    dt = np.fabs(t1 - t2)
    # 1/edd is accretion timescale, so dt*edd is fraction
    frac = dt*mdot_edd
    return frac


def _compare_select_correction_mass(masses):

    def _close(m1, m2):
        return np.isclose(m1, m2, atol=0.0, rtol=1.0)

    masses = np.asarray(masses)
    inds = np.where(masses > 0.0)[0]
    # If no options for mass, return zero
    if inds.size == 0:
        return 0.0
    # If single option for mass, return that
    elif inds.size == 1:
        return masses[inds[0]]

    # If there are two possibilities
    elif inds.size == 2:
        masses = masses[inds]
        # If they are comparable, return the smaller one
        if _close(masses[0], masses[1]):
            return np.min(masses)
        # Otherwise, we dont know.
        else:
            print("FAILED OPTIONS: {}, {}".format(masses[0], masses[1]))
            return 0.0

    # If there are three possibilities
    elif inds.size == 3:
        # If they are all comparable, return the smallest one
        if _close(masses[0], masses[1]) and _close(masses[1], masses[2]):
            return np.min(masses)
        # If a pair are comparable, return minimum of those
        else:
            for ii in range(3):
                jj = (ii + 1) % 2
                if _close(masses[ii], masses[jj]):
                    return np.min([masses[ii], masses[jj]])

            print("FAILED OPTIONS: {}, {}, {}".format(masses[0], masses[1], masses[2]))

    return 0.0


def _check_mergers_for_repeats(m_scale, m_id_in, m_id_out, det_q_id, det_q_first, det_scale, det_id,
                               verbose=True):
    num_again_in = 0
    num_again_out = 0
    num_cont = 0
    num_det_unique = det_q_id.size
    num_mergers = m_scale.size
    valid = np.ones(num_mergers, dtype=bool)
    beg = datetime.now()
    for mm in range(num_mergers):
        inid = m_id_in[mm]
        # Make sure the `in` BH dissappears
        in_is_in = np.where(inid == m_id_in[mm+1:])[0] + mm+1
        if in_is_in.size:
            valid[mm] = False
            num_again_in += 1
            continue
        in_is_out = np.where(inid == m_id_out[mm+1:])[0] + mm+1
        if in_is_out.size:
            valid[mm] = False
            num_again_out += 1
            continue
        in_cont = np.searchsorted(det_q_id, inid).clip(max=num_det_unique)
        # If this 'in' BH has any details entries
        if det_q_id[in_cont] == inid:
            if in_cont+1 < num_det_unique:
                i1 = det_q_first[in_cont+1] - 1
            else:
                i1 = num_det_unique - 1

            if det_id[i1] != inid:
                raise ValueError("DETAILS ID MISMATCH!")

            # See if any of them are *after* the time of merger
            if det_scale[i1] > m_scale[mm]:
                valid[mm] = False
                num_cont += 1
                continue

    if verbose:
        print("{} Bad Mergers after {}.".format(np.sum(valid == False), datetime.now()-beg))
        print("\tin-again: {}, out-again: {}, continue: {}".format(
            num_again_in, num_again_out, num_cont))

    return valid


def _check_version(fname):
    return constants._check_version(fname, __version__)
