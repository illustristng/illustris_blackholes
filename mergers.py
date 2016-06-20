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

import astropy
import astropy.constants
from astropy.cosmology import WMAP9 as cosmo
from datetime import datetime
import h5py
import numpy as np
import os
import shutil
import warnings

from constants import DETAILS, DTYPE, \
    GET_ILLUSTRIS_BH_MERGERS_FILENAMES, GET_MERGERS_COMBINED_FILENAME, \
    GET_MERGERS_DETAILS_FILENAME, GET_OUTPUT_DETAILS_FILENAME, \
    GET_PUBLIC_MERGERS_FILENAME, MERGERS, \
    _backup_exists, _zero_pad_end

SPLC = astropy.constants.c.cgs.value
NWTG = astropy.constants.G.cgs.value
MPRT = astropy.constants.m_p.cgs.value
SIGMA_T = astropy.constants.sigma_T.cgs.value
MSOL = astropy.constants.M_sun.cgs.value
KPC = astropy.constants.kpc.cgs.value
HPAR = cosmo.h
YR = astropy.units.yr.to(astropy.units.second)


class CONV_ILL_TO_CGS:
    """Convert from illustris units to physical [cgs] units (multiply).
    """
    MASS = 1.0e10*MSOL/HPAR               # Convert from e10 Msol to [g]
    MDOT = 10.22*MSOL/YR                  # Multiply by this to get [g/s]
    DENS = 6.77025e-22                    # (1e10 Msol/h)/(ckpc/h)^3 to g/cm^3 *COMOVING*
    DIST = KPC/HPAR                       # Convert from [ckpc/h] to [comoving cm]
    VEL  = 1.0e5                          # [km/s] to [cm/s]
    CS   = 1.0                            # ??????? FIX

__version__ = '1.0.1'

# Size of initial Details arrays allocation; and 'chunk' size that it is increased by as needed
_MERGS_BUF_SIZE = int(1e5)


def main(run=1, output_dir=None, verbose=True, ):
    # Load Mapped Mergers ###
    # re-creates them if needed
    # mergersMapped = loadMappedMergers(run, verbose=verbose)

    # Load Fixed Mergers ###
    # mergersFixed = loadFixedMergers(run, verbose=verbose)

    return


def combine_raw_merger_files(run, verbose=True):
    """Concatenate the contents of a set of input files into a single output file.

    Arguments
    ---------
    in_fnames : iterable<str>, list of input file names
    out_fname : <str>, output file name
    verbose : <bool> (optional=_VERBOSE), print verbose output

    Returns

    """
    beg = datetime.now()
    print(" - Combining merger files")
    # Raw illustris merger filenames
    in_filenames = GET_ILLUSTRIS_BH_MERGERS_FILENAMES(run)
    # Filename for combined mergers file, raw = unfiltered
    out_raw_fname = GET_MERGERS_COMBINED_FILENAME(run, filtered=False, type='txt')
    if verbose:
        print(" - - Writing   raw    combined mergers to '{}'".format(out_raw_fname))
    # Make backups of existing output files
    _backup_exists(out_raw_fname, verbose=verbose)

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
        if verbose: print(" - Iterating over {} input merger files".format(nums_in))
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
                    warnings.warn("Merger overwrite detected.  Time: {}, last: {}.\n\t"
                                  "file_count: {}, count: {}, bads: {}".format(
                                      time, last_time, file_count, count, bads))
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
                    scale = _zero_pad_end(scale, _MERGS_BUF_SIZE)
                    id_in = _zero_pad_end(id_in, _MERGS_BUF_SIZE)
                    id_out = _zero_pad_end(id_out, _MERGS_BUF_SIZE)
                    mass_in = _zero_pad_end(mass_in, _MERGS_BUF_SIZE)
                    mass_out = _zero_pad_end(mass_out, _MERGS_BUF_SIZE)

            if verbose and ii % interv == 0:
                dur = datetime.now()-beg
                print("\t{:5d}/{} = {:.4f} after {}.  Raw: {:5d}; Filtered: {:5d}, Deleted: {:5d}"
                      "".format(ii, nums_in, ii/nums_in, dur, count_raw, count, num_deleted))

    if verbose:
        print("Finished Raw: {}, Filtered: {} lines after {}".format(
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
        print(" - Mergers: {}, Unique BH: {}".format(inds.size, all_ids.size))

    # Write Raw data to hdf5 file
    # Filename for combined mergers file, filtered
    out_filtered_fname = GET_MERGERS_COMBINED_FILENAME(run, filtered=True, type='hdf5')
    print(" - Writing filtered combined mergers to '{}'".format(out_filtered_fname))
    _backup_exists(out_filtered_fname, verbose=verbose)
    with h5py.File(out_filtered_fname, 'w') as h5file:
        # Add metadata in "Header" dataset
        head = h5file.create_group('Header')
        head.attrs['script'] = str(__file__)
        head.attrs['script_version'] = str(__version__)
        head.attrs['created'] = str(datetime.now().ctime())
        head.attrs['simulation'] = 'Illustris-{}'.format(run)
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
        print(" - Saved to '{}', Size: '{}' MB".format(out_filtered_fname, fsize))

    return count_raw, count


def combine_mergers_with_details(run, verbose=True):
    """
    """

    from astropy.cosmology import WMAP9 as cosmo

    print(" - Combining mergers with details.")
    beg_all = datetime.now()
    mergers_in_fname = GET_MERGERS_COMBINED_FILENAME(run, filtered=True, type='hdf5')
    mdets_in_fname = GET_MERGERS_DETAILS_FILENAME(run)
    details_in_fname = GET_OUTPUT_DETAILS_FILENAME(run)

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
    m_lbt = cosmo.lookback_time((1/m_scale) - 1).cgs.value
    for ii in range(3):
        inds = np.where(d_scale[:, ii] > 0.0)[0]
        d_lbt[inds, ii] = cosmo.lookback_time((1/d_scale[inds, ii]) - 1).cgs.value

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
        # dscale = d_scale[mm]
        # did = d_id[mm]

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
    # md_id = d_id[val, :]
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

    # Produce Public Output File
    mergers_out_fname = GET_PUBLIC_MERGERS_FILENAME(run)
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
        head.attrs['created'] = str(datetime.now().ctime())
        head.attrs['simulation'] = 'Illustris-{}'.format(run)
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


def convert_txt_to_hdf5(run, verbose=True, recombine=False):
    """

    Raw mergers are the data directly from illustris without modification.
    """
    # Intermediate filename to store all mergers in single text file
    #    Mergers will be in effectively random order
    combined_txt_fname = GET_MERGERS_RAW_FILENAME(run, type='txt')
    if recombine or not os.path.exists(combined_txt_fname):
        merger_fnames = GET_ILLUSTRIS_BH_MERGERS_FILENAMES(run)
        if verbose:
            print(" - Combining Merger Data from {} files".format(len(merger_fnames)))
        num_lines = combine_raw_merger_files(merger_fnames, combined_txt_fname, verbose)
    else:
        num_lines = sum(1 for line in open(combined_txt_fname))

    scale = np.zeros(num_lines, dtype=DTYPE.SCALAR)
    id_in = np.zeros(num_lines, dtype=DTYPE.ID)
    id_out = np.zeros(num_lines, dtype=DTYPE.ID)
    mass_in = np.zeros(num_lines, dtype=DTYPE.SCALAR)
    mass_out = np.zeros(num_lines, dtype=DTYPE.SCALAR)

    count = 0
    # Go through each merger (each line) in combined file, store parsed values
    if verbose:
        print(" - Loading combined merger data from '{}'".format(combined_txt_fname))
    beg = datetime.now()
    for line in open(combined_txt_fname, 'r'):
        # Get target elements, as appropriate data types, from each line of file
        #    NOTE: `out_mass` is incorrect in illustris
        time, out_id, out_mass, in_id, in_mass = _parse_line_merger(line)
        # Store values
        scale[count] = time
        id_in[count] = in_id
        id_out[count] = out_id
        mass_in[count] = in_mass
        mass_out[count] = out_mass
        count += 1

    # Resize arrays for the number of values actually read
    if verbose:
        print(" - Read {} lines after {}".format(count, datetime.now()-beg))
    if count != num_lines:
        raise ValueError("`count` = {} != `num_lines` = {}".format(count, num_lines))

    # Find indices which sort by time
    inds = np.argsort(scale)
    # Use indices to reorder arrays
    scale   = scale[inds]
    id_in    = id_in[inds]
    id_out   = id_out[inds]
    mass_in  = mass_in[inds]
    mass_out = mass_out[inds]

    # Calculate Meta-Data
    # -------------------
    # Find all unique BH IDs
    all_ids = np.append(id_in, id_out)
    all_ids = np.unique(all_ids)
    if verbose: print(" - Mergers: {}, Unique BH: {}".format(inds.size, all_ids.size))

    # Write Raw data to hdf5 file
    hdf5_fname = GET_MERGERS_RAW_FILENAME(run, type='hdf5')
    if verbose: print("Saving merger data to '{}'".format(hdf5_fname))
    with h5py.File(hdf5_fname, 'w') as h5file:
        # Add metadata in "Header" dataset
        head = h5file.create_group('Header')
        head.attrs['script'] = str(__file__)
        head.attrs['script_version'] = str(__version__)
        head.attrs['created'] = str(datetime.now().ctime())
        head.attrs['simulation'] = 'Illustris-{}'.format(run)
        head.attrs['description'] = (
            "Illustris blackhole merger data, combined from all of the "
            "individual blackhole (BH) merger text files.  The content of the "
            "data is completely unchanged.  Each merger involves two BH, the 'out' "
            "BH which persists after the merger, and the 'in' BH which is consumed.  "
            "NOTE: the mass of the 'out' BH is incorrect in this data.  The values "
            "given correspond to the total cell (dynamical) mass, instead of the BH "
            "mass itself."
        )
        head['unique_ids'] = all_ids

        # Add merger data
        time_dset = h5file.create_dataset(MERGERS.SCALE, data=scale)
        time_dset.attrs['units'] = 'Cosmological scale factor'
        h5file.create_dataset(MERGERS.ID_IN, data=id_in)
        h5file.create_dataset(MERGERS.ID_OUT, data=id_out)
        h5file.create_dataset(MERGERS.MASS_IN, data=mass_in)
        h5file.create_dataset(MERGERS.MASS_OUT, data=mass_out)

        '''
        # Merger tree data
        h5file.create_dataset(MERGERS.NEXT, data=m_next)
        h5file.create_dataset(MERGERS.PREV_IN, data=m_prev_in)
        h5file.create_dataset(MERGERS.PREV_OUT, data=m_prev_out)
        '''

    if verbose:
        fsize = os.path.getsize(hdf5_fname)/1024/1024
        print(" - Saved to '{}', Size: '{}' MB".format(hdf5_fname, fsize))

    return  # scale, id_in, id_out, mass_in, mass_out, hdf5_fname

'''
def loadMappedMergers(run, verbose=True, loadsave=True):
    """Load or create Mapped Mergers Dictionary as needed.
    """

    if verbose: print(" - - BHMergers.loadMappedMergers")

    mappedFilename = GET_MERGERS_RAW_MAPPED_FILENAME(run, VERSION_MAP)

    ## Load Existing Mapped Mergers
    #  ----------------------------
    if(loadsave):
        if verbose: print(" - - - Loading saved data from '%s'" % (mappedFilename))
        # If file exists, load data
        if(os.path.exists(mappedFilename)):
            mergersMapped = zio.npzToDict(mappedFilename)
        else:
            print(" - - - - '%s' does not exist.  Recreating" % (mappedFilename))
            loadsave = False


    ## Recreate Mappings
    #  -----------------
    if(not loadsave):
        if verbose: print(" - - - Recreating mapped mergers")

        # Load Raw Mergers
        scales, ids, masses, filename = load_raw_mergers(run, verbose=verbose)

        ### Create Mapping Between Mergers and Snapshots ###
        mapM2S, mapS2M, ontop = _mapToSnapshots(scales)

        # Store in dictionary
        mergersMapped = { MERGERS.FILE      : mappedFilename,
                          MERGERS.RUN       : run,
                          MERGERS.NUM       : len(scales),
                          MERGERS.CREATED   : datetime.now().ctime(),
                          MERGERS.VERSION   : VERSION_MAP,

                          MERGERS.SCALES    : scales,
                          MERGERS.IDS       : ids,
                          MERGERS.MASSES    : masses,

                          MERGERS.MAP_MTOS  : mapM2S,
                          MERGERS.MAP_STOM  : mapS2M,
                          MERGERS.MAP_ONTOP : ontop,
                          }

        zio.dictToNPZ(mergersMapped, mappedFilename, verbose=verbose)


    return mergersMapped


def loadFixedMergers(run, verbose=True, loadsave=True):
    """
    Load BH Merger data with duplicats removes, and masses corrected.

    Arguments
    ---------
       run      <int>  : illustris simulation run number {1, 3}
       verbose  <bool> : optional, print verbose output
       loadsave <bool> : optional, load existing save file (recreate if `False`)

    Returns
    -------
       mergersFixed <dict> : dictionary of 'fixed' mergers, most entries shaped [N, 2] for `N`
                             mergers, and an entry for each {``BH_TYPE.IN``, ``BH_TYPE.OUT``}

    """

    if verbose: print(" - - BHMergers.loadFixedMergers")

    fixedFilename = GET_MERGERS_FIXED_FILENAME(run, VERSION_FIX)

    ## Try to Load Existing Mapped Mergers
    if(loadsave):
        if verbose: print(" - - - Loading from save '%s'" % (fixedFilename))
        if(os.path.exists(fixedFilename)):
            mergersFixed = zio.npzToDict(fixedFilename)
        else:
            print(" - - - - '%s' does not exist.  Recreating." % (fixedFilename))
            loadsave = False


    ## Recreate Fixed Mergers
    if(not loadsave):
        if verbose: print(" - - - Creating Fixed Mergers")
        # Load Mapped Mergers
        mergersMapped = loadMappedMergers(run, verbose=verbose)
        # Fix Mergers
        mergersFixed = _fixMergers(run, mergersMapped, verbose=verbose)
        # Save
        zio.dictToNPZ(mergersFixed, fixedFilename, verbose=verbose)


    return mergersFixed


def _fixMergers(run, mergers, verbose=True):
    """
    Filter and 'fix' input merger catalog.

    This includes:
     - Remove duplicate entries (Note-1)
     - Load 'fixed' out-BH masses from ``BHMatcher`` (which uses ``BHDetails`` entries)

    Arguments
    ---------
       run     <int>  : illustris simulation number {1, 3}
       mergers <dict> : input dictionary of unfiltered merger events
       verbose <bool> : optional, print verbose output

    Returns
    -------
       fixedMergers <dict> : filtered merger dictionary

    Notes
    -----
       1 : There are 'duplicate' entries which have different occurence times (scale-factors)
           suggesting that there is a problem with the actual merger, not just the logging.
           This is not confirmed.  Currently, whether the times match or not, the *later*
           merger entry is the only one that is preserved in ``fixedMergers``

    """
    from illpy.illbh import BHMatcher

    if verbose: print(" - - BHMergers._fixMergers")

    # Make copy to modify
    fixedMergers = dict(mergers)

    # Remove Repeated Entries
    # =======================
    # Remove entries where IDs match a second time (IS THIS ENOUGH?!)

    ids    = fixedMergers[MERGERS.IDS]
    scales = fixedMergers[MERGERS.SCALES]

    # First sort by ``BH_TYPE.IN`` then ``BH_TYPE.OUT`` (reverse of given order)
    sort = np.lexsort((ids[:, BH_TYPE.OUT], ids[:, BH_TYPE.IN]))

    badInds = []
    numMismatch = 0

    if verbose: print(" - - - Examining %d merger entries" % (len(sort)))

    # Iterate over all entries
    for ii in range(len(sort)-1):

        this = ids[sort[ii]]
        jj = ii+1

        # Look through all examples of same BH_TYPE.IN
        while(ids[sort[jj], BH_TYPE.IN] == this[BH_TYPE.IN]):
            # If BH_TYPE.OUT also matches, this is a duplicate -- store first entry as bad |NOTE-1|
            if(ids[sort[jj], BH_TYPE.OUT] == this[BH_TYPE.OUT]):

                # Double check that time also matches
                if(scales[sort[ii]] != scales[sort[jj]]): numMismatch += 1
                badInds.append(sort[ii])
                break

            jj += 1

        # } while
    # ii

    if verbose: print(" - - - Total number of duplicates = %d" % (len(badInds)))
    if verbose: print(" - - - Number with mismatched times = %d" % (numMismatch))

    # Remove Duplicate Entries
    for key in MERGERS_PHYSICAL_KEYS:
        fixedMergers[key] = np.delete(fixedMergers[key], badInds, axis=0)

    # Recalculate maps
    mapM2S, mapS2M, ontop = _mapToSnapshots(fixedMergers[MERGERS.SCALES])
    fixedMergers[MERGERS.MAP_MTOS] = mapM2S
    fixedMergers[MERGERS.MAP_STOM] = mapS2M
    fixedMergers[MERGERS.MAP_ONTOP] = ontop

    # Change number, creation date, and version
    oldNum = len(mergers[MERGERS.SCALES])
    newNum = len(fixedMergers[MERGERS.SCALES])
    fixedMergers[MERGERS.NUM] = newNum
    fixedMergers[MERGERS.CREATED] = datetime.now().ctime()
    fixedMergers[MERGERS.VERSION] = VERSION_FIX

    if verbose: print(" - - - Number of Mergers %d ==> %d" % (oldNum, newNum))

    # Fix Merger 'Out' Masses
    #  =======================
    if verbose: print(" - - - Loading reconstructed 'out' BH masses")
    masses = fixedMergers[MERGERS.MASSES]
    aveBef = np.average(masses[:, BH_TYPE.OUT])
    massOut = BHMatcher.inferMergerOutMasses(run, mergers=fixedMergers, verbose=verbose)
    masses[:, BH_TYPE.OUT] = massOut
    aveAft = np.average(masses[:, BH_TYPE.OUT])
    if verbose: print(" - - - - Ave mass:  %.4e ===> %.4e" % (aveBef, aveAft))

    return fixedMergers
'''


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

'''
def _mapToSnapshots(scales, verbose=True):
    """Find the snapshot during which, or following each merger
    """

    if verbose: print(" - - BHMergers._mapToSnapshots")

    numMergers = len(scales)

    # Load Cosmology
    import illpy.illcosmo
    cosmo = illpy.illcosmo.cosmology.Cosmology()
    snapScales = cosmo.scales()

    # Map Mergers-2-Snapshots: snapshot before (or ontop) of each merger
    mapM2S = np.zeros(numMergers, dtype=DTYPE.INDEX)
    # Map Snapshots-2-Mergers: list of mergers just-after (or ontop) of each snapshot
    mapS2M = [[] for ii in range(cosmo.num)]
    # Flags if merger happens exactly on a snapshot (init to False=0)
    ontop  = np.zeros(numMergers, dtype=bool)

    # Find snapshots on each side of merger time ###

    # Find the snapshot just below and above each merger.
    #     each entry (returned) is [low, high, dist-low, dist-high]
    #     low==high if the times match (within function's default uncertainty)
    snapBins = [_findBoundingBins(sc, snapScales) for sc in scales]

    # Create Mappings
    # ---------------

    if verbose:
        print(" - - - Creating mappings")
        pbar = zio.getProgressBar(numMergers)

    for ii, bins in enumerate(snapBins):
        tsnap = bins[1]                                                                             # Set snapshot to upper bin
        mapM2S[ii] = tsnap                                                                          # Set snapshot for this merger
        mapS2M[tsnap].append(ii)                                                                    # Add merger to this snapshot
        # If this merger takes place ontop of snapshot, set flag
        if(bins[0] == bins[1]): ontop[ii] = True

        # Print Progress
        if verbose: pbar.update(ii)

    # ii

    if verbose: pbar.finish()

    # Find the most mergers in a snapshot
    numPerSnap = np.array([len(s2m) for s2m in mapS2M])
    mostMergers = np.max(numPerSnap)
    mostIndex = np.where(mostMergers == numPerSnap)[0]
    # Find the number of ontop mergers
    numOntop = np.count_nonzero(ontop)
    if verbose: print(" - - - Snapshot %d with the most (%d) mergers" % (mostIndex, mostMergers))
    if verbose: print(" - - - %d (%.2f) ontop mergers" % (numOntop, 1.0*numOntop/nums))

    return mapM2S, mapS2M, ontop
'''

'''
def _findBoundingBins(target, bins, thresh=1.0e-5):
    """
    Find the array indices (of "bins") bounding the "target"

    If target is outside bins, the missing bound will be 'None'
    low and high will be the same, if the target is almost exactly[*1] equal to a bin

    [*1] : How close counds as effectively the same is set by 'DEL_TIME_THRESH' below

    arguments
    ---------
        target  : [] value to be compared
        bins    : [] list of values to compare to the 'target'

    output
    ------
        low  : [int] index below target (or None if none)
        high : [int] index above target (or None if none)

    """

    # deltat  : test whether the fractional difference between two values is less than threshold
    #           This function allows the later conditions to accomodate smaller numerical
    #           differences, between effectively the same value  (e.g.   1.0 vs. 0.9999999999989)
    #
    if(thresh == 0.0): deltat = lambda x, y : False
    else               : deltat = lambda x, y : np.abs(x-y)/np.abs(x) <= thresh

    nums   = len(bins)
    # Find bin above (or equal to) target
    high = np.where((target <= bins) | deltat(target, bins))[0]
    if(len(high) == 0): high = None
    # Select first bin above target
    else:
        high = high[0]
        dhi  = bins[high] - target

    # Find bin below (or equal to) target
    low  = np.where((target >= bins) | deltat(target, bins))[0]
    if(len(low)  == 0): low  = None
    # Select  last bin below target
    else:
        low  = low[-1]
        dlo  = bins[low] - target

    # Print warning on error
    if(low == None or high == None):
        print("BHMergers._findBoundingBins: target = %e, bins = {%e, %e}; low, high = %s, %s !" % \
            (target, bins[0], bins[-1], str(low), str(high)))
        raise RuntimeError("Could not find bins!")

    return [low, high, dlo, dhi]
'''

'''
if __name__ == "__main__":
    main()
'''


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
                jj == (ii + 1) % 2
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
