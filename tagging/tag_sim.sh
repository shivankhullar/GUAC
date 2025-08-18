#!/usr/bin/env bash

PATH_TO_SIMS="/fs/lustre/scratch/vpustovoit/SHIVAN/CCA_DATA/"
SIM_NAME="$1"
SIM_NAME="m12f"
SPHERE_SIZE="0.1"
OUTPUT_DIR=0 # Set to 1 if directory structure is like PATH/m12f/output/snapshot_000.hdf5

#python FIRE-3_CloudSelection.py "$PATH_TO_SIMS" "$SIM_NAME" $OUTPUT_DIR $SPHERE_SIZE
SNAP_NUM=$(cat snapnum.txt)
LOAD_FILE_NAME="cloud_tracked_pIDs.npy"
echo "Snapshot number: $SNAP_NUM"

python ../scripts/create_ics/create_fire_ic_refine_tags.py --path="$PATH_TO_SIMS" \
	                                                   --snap_num="$SNAP_NUM" \
	                                                   --load_file_name="$LOAD_FILE_NAME" \
							   --load_file_path="$PATH_TO_SIMS"\
							   --sim="$SIM_NAME" \
&& echo "Succeeded in creating ICs file"
