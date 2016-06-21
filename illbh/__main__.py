"""Primary, command-line entry point for Illustris Black holes module.

"""
from datetime import datetime
import numpy as np
import os
import warnings
from illbh import constants
from . import details
from . import mergers

_DEF_OUTPUT_DIR = './'
_DEF_VERBOSE = True


def main():
    beg_time = datetime.now()
    args = load_args()

    # Setup Output Directory
    output_dir = os.path.abspath(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.isdir(output_dir):
        raise ValueError("`output_dir` '{}' is invalid.".format(output_dir))

    # Say Hello
    if args.verbose:
        git_vers = constants._get_git()
        title_str = "Illustris Black Holes Module, version: {}".format(git_vers)
        print("\n\n{}\n{}\n{}\n".format(title_str, '='*len(title_str), beg_time.ctime()))
    print("Output directory: '{}'".format(output_dir))

    # Make 'raw' 'organized' BH Merger Files
    # --------------------------------------
    if args.verbose:
        print("\nCombining merger files")
    mergers_combined_good = _check_mergers_combined(args, output_dir)
    # Run `combine_raw_merger_files` as needed
    if args.refresh_all or not mergers_combined_good:
        mergers.combine_raw_merger_files(args.run, verbose=args.verbose, output_dir=output_dir)
    if not _check_mergers_combined(args, output_dir):
        raise RuntimeError("Combined merger files are not okay!")

    # Process Details
    # ---------------
    if args.verbose:
        print("\nProcessing Details")
    # Organize details entries by snapshot, in txt files
    details_organized_txt_good = _check_details_organized_txt
    if args.refresh_all or not details_organized_txt_good:
        if args.verbose:
            print(" - Organizing Details by snapshot, as 'txt'")
        details.organize_txt_by_snapshot(args.run, verbose=args.verbose, output_dir=output_dir)
    if not _check_details_organized_txt(args, output_dir):
        raise RuntimeError("Details organized by snapshot ('txt') are not okay!")

    # Convert txt files to hdf5 files
    details_organized_hdf5_good = _check_details_organized_hdf5(args, output_dir)
    if args.refresh_all or not details_organized_hdf5_good:
        if args.verbose:
            print("\n - Converting Details by snapshot to 'hdf5'")
        details.convert_txt_to_hdf5(args.run, verbose=args.verbose, output_dir=output_dir)
    if not _check_details_organized_hdf5(args, output_dir):
        raise RuntimeError("Details organized by snapshot ('hdf5') are not okay!")

    # Downsample details and find merger-details
    details_good = _check_final_details(args, output_dir)
    if args.refresh_all or not details_good:
        if args.verbose:
            print("\n - Downsampling details and finding merger-details")
        details.combine_downsample_and_mergers_hdf5(
            args.run, verbose=args.verbose, output_dir=output_dir)
    if not _check_final_details(args, output_dir):
        raise RuntimeError("Details and merger-details are not okay!")

    # Make Final Mergers File
    # -----------------------
    mergers_good = _check_final_mergers(args, output_dir)
    if args.refresh_all or not mergers_good:
        if args.verbose:
            print("\nCombining mergers and merger-details")
        mergers.combine_mergers_with_details(
            args.run, verbose=args.verbose, output_dir=output_dir)
    if not _check_final_mergers(args, output_dir):
        raise RuntimeError("Final mergers are not okay!")

    end_time = datetime.now()
    print("All complete at {}, After {}".format(end_time, end_time-beg_time))
    return


def load_args(args=None):
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--verbose', '-v', dest='verbose', default=_DEF_VERBOSE, action='store_true',
                        help='Print more messages to the screen.')
    parser.add_argument('--update', '-u', dest='refresh_all', default=True, action='store_false',
                        help="Only recreate files that are outdated.  Otherwise, recreate all.")
    parser.add_argument('--output', dest='output_dir', default=_DEF_OUTPUT_DIR,
                        help='Destination directory for output.')
    parser.add_argument('--run', '-r', dest='run', default=1,
                        help='Illustris simulation run number {1,3}')

    args = parser.parse_args(args=args)
    return args


def _check_mergers_combined(args, output_dir):
    # Make sure files exist and are up to date
    mergers_raw_fname = constants.GET_MERGERS_COMBINED_FILENAME(
        args.run, filtered=False, type='txt', output_dir=output_dir)
    mergers_filt_fname = constants.GET_MERGERS_COMBINED_FILENAME(
        args.run, filtered=True, type='hdf5', output_dir=output_dir)
    raw_exists = os.path.isfile(mergers_raw_fname)
    filt_exists = os.path.isfile(mergers_filt_fname)
    if filt_exists:
        filt_current = mergers._check_version(mergers_filt_fname)
    else:
        filt_current = False
    is_good = (raw_exists and filt_exists and filt_current)
    if args.verbose:
        print(" - File '{}' exists: {}".format(mergers_raw_fname, raw_exists))
        print(" - File '{}' exists: {}, Current: {}".format(
            mergers_filt_fname, filt_exists, filt_current))
    return is_good


def _check_details_organized_txt(args, output_dir):
    output_fnames = \
        [constants.GET_DETAILS_ORGANIZED_FILENAME(args.run, snap, type='txt', output_dir=output_dir)
         for snap in range(constants.NUM_SNAPS)]
    is_good = np.all([os.path.exists(ofn) for ofn in output_fnames])
    if args.verbose:
        print(" - Files '{}'... all exist: {}".format(output_fnames[0], is_good))
    return is_good


def _check_details_organized_hdf5(args, output_dir):
    output_fnames = \
        [constants.GET_DETAILS_ORGANIZED_FILENAME(args.run, snap, type='hdf5', output_dir=output_dir)
         for snap in range(constants.NUM_SNAPS)]
    is_good = np.all([os.path.exists(ofn) for ofn in output_fnames])
    if is_good:
        is_current = np.all([details._check_version(ofn) for ofn in output_fnames])
    else:
        is_current = False
    if args.verbose:
        print(" - Files '{}'... all exist: {}, all current: {}".format(
            output_fnames[0], is_good, is_current))
    is_good = (is_good and is_current)
    return is_good


def _check_final_details(args, output_dir):
    mdet_fname = constants.GET_MERGERS_DETAILS_FILENAME(args.run, output_dir=output_dir)
    dets_fname = constants.GET_PUBLIC_DETAILS_FILENAME(args.run, output_dir=output_dir)
    mdet_exists = os.path.isfile(mdet_fname)
    dets_exists = os.path.isfile(dets_fname)
    mdet_current = dets_current = False
    if mdet_exists:
        mdet_current = details._check_version(mdet_fname)
    if dets_exists:
        dets_current = details._check_version(dets_fname)

    if args.verbose:
        print(" - File '{}' exists: {}, Current: {}".format(
            mdet_fname, mdet_exists, mdet_current))
        print(" - File '{}' exists: {}, Current: {}".format(
            dets_fname, dets_exists, dets_current))

    is_good = (mdet_exists and mdet_current and dets_exists and dets_current)
    return is_good


def _check_final_mergers(args, output_dir):
    mergers_fname = constants.GET_PUBLIC_MERGERS_FILENAME(args.run, output_dir=output_dir)
    exists = os.path.isfile(mergers_fname)
    if exists:
        current = mergers._check_version(mergers_fname)
    else:
        current = False
    if args.verbose:
        print(" - File '{}' exists: {}, Current: {}".format(
            mergers_fname, exists, current))
    is_good = exists and current
    return is_good

if __name__ == "__main__":
    main()
