# how to run the NeuronCounting pipeline
# allows you to run this code from anywhere and the internal relative paths still work fine
#cd $(dirname ${BASH_SOURCE[0]}) || exit;
cd $WORKSTATION_CODE/shared/img_processing/NeuronCounting/examples || exit;
# first, run the environment setup script. This makes sure that the shell and python all know where imaris etc are
source ../environment_setup.bash



# input arguments
directory='B:/20.5xfad.01/BXD77/'
folder='B:/20.5xfad.01/BXD77/220114-1_1/' #<- change here when change specimens
folder_name=$(basename "$folder")

echo "Processing folder: $folder_name"

# Find paths
label_imaris_path=$(find "$folder/ims/" -type f -name "*labels.ims" -print -quit)
label_nhdr_path=$(find "$folder/Aligned-Data/labels/RCCF/" -type f -name "*labels.nhdr" -print -quit)
image_imaris_path=$(find "$folder/ims/" -type f -name "*Amyloid.ims" -print -quit)

# Directory for region dictionaries
regions_dict_dir="S:/yt133/To_delete_Statistics/history/region_dicts/Amyloid/"

# Iterate over each JSON file in the region_dicts directory
for regions_dict_path in "$regions_dict_dir"/5xFAD_Amyloid-*.json; do
    # Extract the specific part of the JSON filename to use as the folder name
    json_basename=$(basename "$regions_dict_path" .json)
    json_suffix=${json_basename#5xFAD_Amyloid-}

    # Define the working directory for this iteration
    work_dir="S:/yt133/To_delete_Statistics/history/data_5xFAD_Abeta_NewRegions/${folder_name}/${json_suffix}/"

    # Ensure the working directory exists
    mkdir -p "$work_dir"

    echo "Processing JSON: $json_basename"
    echo "Using work directory: $work_dir"

    # Run the Python script with the appropriate arguments
    python "K:/workstation/code/shared/img_processing/NeuronCounting/MainAlgor-multiRegions-Copy.py" \
        "$label_imaris_path" "$label_nhdr_path" "$image_imaris_path" "$regions_dict_path" "$work_dir"
done
