#!/usr/bin/env bash

PATH_TO_SIMS="/fs/lustre/scratch/vpustovoit/SHIVAN/CCA_DATA/"
SIM_NAME="$1"
SIM_NAME="m12f"
SPHERE_SIZE="0.1"
OUTPUT_DIR=0 # Set to 1 if directory structure is like PATH/m12f/output/snapshot_000.hdf5

#python FIRE-3_CloudSelection.py "$PATH_TO_SIMS" "$SIM_NAME" $OUTPUT_DIR $SPHERE_SIZE
SNAP_NUM=$(cat snapnum.txt)
LOAD_FILE_NAME="cloud_tracked_pIDs.npy"
#PATH_TO_SIMS="$PATH_TO_SIMS/$SIM_NAME"
FULL_LOAD_FILE_PATH="$PATH_TO_SIMS/$SIM_NAME"

if [[ "$OUTPUT" == 1 ]]; then
  FULL_LOAD_FILE_PATH="$PATH_TO_SIMS/$SIM_NAME/output/$LOAD_FILE_NAME"
else
  FULL_LOAD_FILE_PATH="$PATH_TO_SIMS/$SIM_NAME/$LOAD_FILE_NAME"
fi
echo "Snapshot number: $SNAP_NUM"

python ../scripts/create_ics/create_fire_ic_refine_tags.py --path="$PATH_TO_SIMS" \
	                                                   --snap_num="$SNAP_NUM" \
	                                                   --load_file_name="$LOAD_FILE_NAME" \
							   --load_file_path="$PATH_TO_SIMS"\
							   --sim="$SIM_NAME" \
							   --snapdir="0" \
	                                                   --full_load_file_path="$FULL_LOAD_FILE_PATH" \
&& echo "Succeeded in creating ICs file"
