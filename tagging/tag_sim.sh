#!/usr/bin/env bash

PATH_TO_SIMS="/fs/lustre/scratch/vpustovoit/SHIVAN/CCA_DATA/"
SPHERE_SIZE="0.1" # Size of the refinement sphere, in kpc 
OUTPUT_DIR=0 # Set to 1 if directory structure is like PATH/m12f/output/snapshot_000.hdf5
SIM_NAMES="m12f m12i m12m m12q" # List of sims for which to run the code
#SIM_NAMES="m12i" # List of sims for which to run the code


###################
#### MAIN LOOP ####
###################

create_refinetag_ic_for_sim() {

  SIM_NAME="$1"
  python FIRE-3_CloudSelection.py "$PATH_TO_SIMS" "$SIM_NAME" $OUTPUT_DIR $SPHERE_SIZE || exit
  SNAP_NUM=$(cat snapnum.txt)
  LOAD_FILE_NAME="cloud_tracked_pIDs.npy"
  #PATH_TO_SIMS="$PATH_TO_SIMS/$SIM_NAME"
  FULL_LOAD_FILE_PATH="$PATH_TO_SIMS/$SIM_NAME"
  IC_FILENAME="ic_${SIM_NAME}_refine_tags.hdf5"
  
  if [[ "$OUTPUT_DIR" == 1 ]]; then
    FULL_LOAD_FILE_PATH="$PATH_TO_SIMS/$SIM_NAME/output/$LOAD_FILE_NAME"
  else
    FULL_LOAD_FILE_PATH="$PATH_TO_SIMS/$SIM_NAME/$LOAD_FILE_NAME"
  fi
  echo "Snapshot number: $SNAP_NUM"
  
  python ../scripts/create_ics/create_fire_ic_refine_tags.py --path="$PATH_TO_SIMS" \
  	                                                   --snap_num="$SNAP_NUM" \
  	                                                   --load_file_name="$LOAD_FILE_NAME" \
  							   --load_file_path="$PATH_TO_SIMS"\
  							   --ic_file_name="$IC_FILENAME" \
  							   --sim="$SIM_NAME" \
  							   --snapdir="0" \
  	                                                   --full_load_file_path="$FULL_LOAD_FILE_PATH" \
  && echo "Succeeded in creating ICs file"
}

for SIM in $SIM_NAMES; do
  create_refinetag_ic_for_sim $SIM
done
