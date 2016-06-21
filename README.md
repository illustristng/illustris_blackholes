# Illustris Black Holes Post-Processing Module #

This module compiles and processes the raw black hole (BH) output files from the Illustris simulations, and converts them into publicly usable HDF5 files.  The two primary types of BH output from the illustris simulation are 'details' and 'mergers' files:
-   'details' : handled by `illbh.details.py`
    -    These files contain the output from each BH particle active in the simulation at each integration time-step.
    -    The parameters included are 'ID, time, mass, mdot, rho, cs' which are the BH's unique ID number, the time the entry was recorded (in cosmological scale-factor), the mass of the BH particles, the mass-accretion rate of the particle, the local gas density ('rho'), and the local gas sound-speed ('cs').
-   'mergers' : handled by `illbh.mergers.py`
    -    These files contain a record of each BH-BH merger in the Illustris simulations.
    -    The parameters included are 'time, id_out, mass_out*, id_in, mass_in'.  We distinguish the two BH which participate in the merger as 'in' and 'out' BH.  The difference, determined randomly in the code, is which BH ID number persists along with the remnant after the merger.  The 'out' BH survives after the merger, increased in mass by that of the 'in' BH---which no longer exists.

### Public Output Files ###

Two primary output files are produced for each the details and mergers respectively.
-   Details: `ill-[N]_blackhole_details.hdf5` (for illustris simulation number `N`)
    ```
    'Header'
        'script' : Filename of the generating script (`illbh.details.py`)
        'script_version' : Internal version str of the script.
        'git_version' : Git commit SHA for the repository at time of file creation.
        'created' : Time of file creation.
        'simulation' : Illustris simulation number.
        'target_times' : The times (scale-factors) which were the targets for details entries
        'num_entries' : The total number of details entries stored.
        'num_blackholes' : The total number of unique BH with details entries.
        'description' : A description of the data.

    'unique'
        'id' : ID numbers of each unique BH
        'first_index' : The index number (i.e. row) of the first entry for each unique BH
        'num_entries' : The total number of entries for each unique BH

    'id' : ID number of the BH for each details entry
    'time' : Time (cosmological scale-factor) for each entry.
    'mass' : BH mass
    'mdot' : BH mass accretion rate
    'rho' : local gas density
    'cs' : local gas sound-speed
    ```

-   Mergers: `ill-[N]_blackhole_mergers.hdf5` (for illustris simulation number `N`)
