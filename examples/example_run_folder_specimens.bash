# how to run the NeuronCounting pipeline
# allows you to run this code from anywhere and the internal relative paths still work fine
#cd $(dirname ${BASH_SOURCE[0]}) || exit;
cd $WORKSTATION_CODE/shared/img_processing/NeuronCounting/examples || exit;
# first, run the environment setup script. This makes sure that the shell and python all know where imaris etc are
source ../environment_setup.bash



# input arguments
directory='B:/20.5xfad.01/BXD77/'

for folder in "$directory"/*/; do
    folder_name=$(basename "$folder")

    if [[ $folder_name =~ ^[0-9] ]]; then
        echo "Processing folder: $folder_name"
        label_imaris_path=$(find "$folder/ims/" -type f -name "*labels.ims" -print -quit);
        label_nhdr_path=$(find "$folder/Aligned-Data/labels/RCCF/" -type f -name "*labels.nhdr" -print -quit);
        image_imaris_path=$(find "$folder/ims/" -type f -name "*Amyloid.ims" -print -quit);
        work_dir="S:/yt133/To_delete_Statistics/history/data_5xFAD_Abeta_NewRegions/$folder_name/"


        # your regions dict should have the same basename as your imaris file ${specid}_${contrast}.ims vs ${specid}_${contrast}.json
        # and it should be in the ${NEURON_COUNTING_CODE_FOLDER}/region_dictionaries folder
        # TODO: maybe move this logic of finding json to MainAlgor main function, user can optionally override it with an input argument (just like how work_dir is handled)
        temp=$(basename ${image_imaris_path});
        temp=${temp%%.ims};
        regions_dict_path="S:/yt133/To_delete_Statistics/history/region_dicts/5xFAD_Amyloid.json";

        # work_dir is set by default to ${ims_dir}/NeuronCounting/${specimen_id}/${contrast}
        # this only works for specimen ims files that are named like ${ims_dir}/${specimen_id}_${contrast}.ims
        # script will create new subfolders within work_dir, one for each ROI to process.
        # this will be overridden if you provide 4 arguments
        python "K:/workstation/code/shared/img_processing/NeuronCounting/MainAlgor - multiRegions.py" ${label_imaris_path} ${label_nhdr_path} ${image_imaris_path} ${regions_dict_path} ${work_dir};

        # this one will infer the output directory and create it (creates a subdirectory wherever your ims file is)
        #python "../MainAlgor - multiRegions.py" ${label_imaris_path} ${label_nhdr_path} ${image_imaris_path} ${regions_dict_path}
    fi
	#exit 1;
done
