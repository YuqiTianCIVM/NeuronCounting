import subprocess
import time
import numpy as np
import nrrd
from tifffile import imwrite
import sys
from skimage.io import imread
import os
import shutil
import imagej
import json

# TODO: clean up ice clients
#https://stackoverflow.com/questions/41462415/python-communicator-not-destroyed-during-global-destruction

# these append statements are required to correctly find ImarisLib and all of its dependencies
# this sys.path.append is the correct way to modify your PYTHONPATH variable
sys.path.append(r'C:\Program Files\Bitplane\Imaris 9.9.0\XT\python3'.replace(r'\\', '/'))
sys.path.insert(0, 'k:/workstation/code/shared/pipeline_utilities/imaris')
import ImarisLib
import ij_classifier
#K:/workstation/code/shared/pipeline_utilities/imaris/imaris_hdr_update.py;
import imaris_hdr_update


def GetServer():
    vImarisLib = ImarisLib.ImarisLib()
    vServer = vImarisLib.GetServer()
    return vServer


def launch_imaris_and_open_data(image_imaris_path: str, label_imaris_path: str, label_nhdr_path: str, roi_dict: dict, work_dir: str):
    # Popen is non blocking. Is the under the hood function that drives subprocess.run()
    imaris_process = subprocess.Popen(['C:\\Program Files\\Bitplane\\Imaris 10.0.0\\Imaris.exe', 'id101'])
    # give imaris enough time to open
    time.sleep(10)

    # initialize python-to-imaris bridge
    vServer = GetServer()
    vImarisLib = ImarisLib.ImarisLib()
    v = vImarisLib.GetApplication(101)

    # initialize python-to-imagej bridge
    # TODO: research if it is possible to get this to work in "headless" mode to suppress output. Citrix GUI is unusable due to ImageJ spam while working
    # code FAILS when imagej is in h
    # eadless mode
    # it is recreating the previously seen error, where, for each region, the "tif" directory is populated, but the "tif_out" and "tif_processed" are NOT populated, and no csv results are created either
    # this fails for ALL regions, does not even succeed for the first in the list
    ij = imagej.init(r'K:\CIVM_Apps\Fiji.app', mode='interactive')
    #ij = imagej.init(r'K:\CIVM_Apps\Fiji.app',mode='headless')

    """FileOpen will for a scene clear before it loads in the new volume. Cannot be used to load in multiple imaris files into the same scene at the same time
    aOptions -- [in]  Set up extra options to specify file format, resampling or cropping parameters for loading.
        Use "" for default options (automatic file type detection, no cropping and no resampling)
        - reader="Imaris3" Advice to use the "Bitplane Imaris 3" format. There are several other formats available:  All Formats, Imaris5, Imaris3, Imaris, AndorIQ, Andor, DeltaVision, Biorad, IPLab, IPLabMac, Gatan, CXD, SlideBook, MRC, LeicaLif, LeicaSingle, LeicaSeries, LeicaVista, MicroManager, MetamorphSTK, MetamorphND, ICS, NikonND2, OlympusCellR, OlympusOIB, OlympusOIF, Olympus, OlympusVSI, OmeTiff, OmeXml, OpenlabLiff, OpenlabRaw, PerkinElmer2, Prairie, Till, AxioVision, Lsm510, Lsm410, BmpSeries, TiffSeries, ZeissCZI.
        - croplimitsmin="x0 y0 z0 c0 t0" Minimum crop position for x, y, z, ch and t. Use "0 0 0 0 0" for croplimitsmin and croplimitsmax to disable cropping.
        - croplimitsmax="x y z c t" The point next to the maximum crop position for x, y, z, ch and t. If one of the components is set to zero, the maximum crop position is automatically set to the size of the dataset along the corresponding dimension.
        - resample="rx ry rz rc rt"
        - LoadDataSet="eDataSetYes | eDataSetNo | eDataSetWithDialog" Down-sampling factor for x, y, z, ch and t. "1 1 1 1 1" does not resample. The size of the loaded dataset is equal to "(croplimitsmax - croplimitsmin) / resample" along each dimension (e.g. "(x-x0)/rx"). %% The following MATLAB code opens the example image "retina.ims" with the Imaris3 reader, with cropping and resampling enabled for x and y (while loading all slices, all channels, all time points) vImarisApplication.FileOpen('images\\retina.ims', ... ['reader="Imaris3" ' ... 'croplimitsmin="10 10 0 0 0" ' ... 'croplimitsmax="150 100 0 0 0" ' ... 'resample="2 2 1 1 1"']);
        EXAMPLE CALL:
            vImarisApplication.FileOpen('images\\retina.ims', ... ['reader="Imaris3" ' ... 'croplimitsmin="10 10 0 0 0" ' ... 'croplimitsmax="150 100 0 0 0" ' ... 'resample="2 2 1 1 1"']);"""
    load_options = ""
    v.FileOpen(label_imaris_path, load_options)
    img2 = v.GetImage(0)
    vExtentMin2 = [img2.GetExtendMinX(), img2.GetExtendMinY(), img2.GetExtendMinZ()]

    v.FileOpen(image_imaris_path, load_options)
    img = v.GetImage(0)
    vExtentMin = [img.GetExtendMinX(), img.GetExtendMinY(), img.GetExtendMinZ()]
    vExtentMax = [img.GetExtendMaxX(), img.GetExtendMaxY(), img.GetExtendMaxZ()]

    print('label vExtentMin:', vExtentMin2)
    print('image vExtentMin:', vExtentMin)
    print('image vExtentMax:', vExtentMax, flush=True)

    # load in label nhdr file. moved from mainalgor so the load step is not repeated N times
    label_data, label_nhdr = nrrd.read(label_nhdr_path)
    label_data = label_data[::-1, ::-1, :].astype(int)

    for region_name in roi_dict:
        label = roi_dict[region_name]["labels"]
        classifier = roi_dict[region_name]["classifier"]
        volume_bar = roi_dict[region_name]["volume_bar"]
        volume_avgbar = roi_dict[region_name]["volume_avgbar"]
        try:
            # what does an error actually mean in this case?
            mainAlgor(label_nhdr, label_data, label, classifier, work_dir, N=30, volume_bar=volume_bar, volume_avgbar=volume_avgbar, vExtentMin=vExtentMin, vExtentMax=vExtentMax, vExtentMin2=vExtentMin2, img=img, vServer=vServer, vImarisLib=vImarisLib, ij=ij)
        except:
            # this does not really work. I got an error, but I think (?) that the imagej process didn't die itself and tried to keep going.
            print("ERROR in neuron counting for ROI: {}".format(region_name))
            print("BURYING HEAD IN THE SAND. KEEP GOING TO THE NEXT REGION!", flush=True)
            # nope continue doesn't work either
            # there is something hanging up and causing other things to fail. maybe it's the imagej process?
            continue
            #imaris_process.kill()

    # very unsure how to handle this
    imaris_process.kill()

"""
# TODO: input arguments to function should be 2 imaris files and
    # full path to image ims file
    # full path to label ims file
    # dictionary of ROI to count and which classifier to use for that ROI

    # out_dir -- should it be set by caller and be an input arg (YES), or should it be decided where to put and created by this function?

Yuqi thinks it's better to make this below:
Write the script as a function, and make the regions as a large dictionary
"""
def mainAlgor(label_nhdr, label_data, label: list, classifier, out_dir: str, N: int = 10, volume_bar: int = 20, volume_avgbar: int = 100, vExtentMin=None, vExtentMax=None, vExtentMin2=None, img=None, vServer=None, vImarisLib=None, ij=None):
#def mainAlgor(label,classifier, N = 10, volume_bar = 20, volume_avgbar = 100):
    """# WHAT DOES VOLUME_VAR AND VOLUME_AVGBAR DO? IMPORTANT?
    ### label: list, classifier: path str, volume_bar: a num that represents the average volume size
    ### newdir: where the file will be saved"""

    # this gives us a "gateway" into ImageJ
    # maybe it is bad for me to reinitialize this every single loop? treat it like you treat the Imaris gateway.
    if ij is None:
        print("No ImageJ bridge found. Reinitializing now.", flush=True)
        ij = imagej.init(r'K:\CIVM_Apps\Fiji.app', mode='interactive')

    macro_path = r"K:\workstation\code\shared\img_processing\NeuronCounting\ij_macro\macroscript.ijm" #where you save your macro

    # I don't think we ever actually use this after the initialization?
    if vServer is None or vImarisLib is None:
        print("my vServer was None! reiniitalizing connection to Imaris", flush=True);
        vServer = GetServer()
        vImarisLib = ImarisLib.ImarisLib()
        # TODO: possible to ping the system to ask for application numbers of running applications named "Imaris"
        # this way, we could open and run imaris in any way we want, get it ready to go, and then
        v = vImarisLib.GetApplication(101)

    if len(label) == 1:
        newdir = "{}/{}".format(out_dir, str(label[0]))
        out_csv_file = "{}/{}_counts.csv".format(out_dir, str(label[0]))
    elif len(label) > 1:
        # this naming makes an assumption that when analyzing muyltipl regions at once, that THEY ARE SEQUENTIAL. what if we wanted toi use rois [5,12,41,99]
        newdir = "{}/{}-{}".format(out_dir, str(label[0]), str(label[-1]))
        out_csv_file = "{}/{}-{}_counts.csv".format(out_dir, str(label[0]), str(label[-1]))
    tif_dir = "{}/tif/".format(newdir)
    tif_out_dir = "{}/tif_out/".format(newdir)
    tif_processed_dir = "{}/tif_processed/".format(newdir)

    # os.makedirs recursively makes all parent directories, and exist_ok=True removes need for if exists check
    os.makedirs(tif_dir, exist_ok=True)
    os.makedirs(tif_out_dir, exist_ok=True)
    os.makedirs(tif_processed_dir, exist_ok=True)

    # completion check. If the csv file already exists, then we will assume this ROI completed successfully already
    if os.path.isfile(out_csv_file):
        print("SKIPPING. Already found output csv file here: {}".format(out_csv_file), flush=True)
        return

    # list comprehension to pull voxel sizes in um from the nrrd header
    # grab the elements from diagonal, *1000 to convert from mm to um, absolute value to make positive, and then cast to integer
    dir_array = label_nhdr['space directions']
    voxel_sizes = [int(abs(dir_array[x][x]*1000)) for x in [0, 1, 2]]

    print("imagej macro arguments:")
    print(f" folder={tif_dir} output={tif_out_dir} processed={tif_processed_dir} macro_path={macro_path}", flush=True)
    ij_macro, args = ij_classifier.save_classifier_macro(tif_dir,tif_out_dir,tif_processed_dir,classifier,macro_path,N=N)

    num = 0
    s = [6, 6, 6]  # Make sure this cube is larger than subvolume. The current cube size is (15um * 10)^3

    # argwhere finds the indices of array elements that are non-zero, grouped by element.
    # this line finds all indices (of the numpy array holding our LABEL file)
    # label data is the label.raw.gz label is the number of the current ROI
    indices = np.argwhere(np.isin(label_data, label))  # Check if elements in 'im' are in 'label'
    print(indices[:, 2])
    x_min = min(indices[:, 0])
    x_max = max(indices[:, 0])
    y_min = min(indices[:, 1])
    y_max = max(indices[:, 1])
    z_min = min(indices[:, 2])
    z_max = max(indices[:, 2])
    arr = np.empty((0, 3), int)
    # N is the number of random sub-regions to create and classify
    while num < N:
        # find the start z index of the "num"th random sub region
        z = int(np.floor(z_min + np.random.rand() * (z_max - s[2] - z_min)))
        if z not in np.squeeze(indices[:, 2]):
            continue
        for iteration in range(10):
            # then i work on individual slices
            label_data_xy = label_data[x_min:x_max, y_min:y_max, z]
            indices_xy = np.argwhere(np.isin(label_data_xy, label))
            if indices_xy.size == 0:
                continue
            x = x_min + np.random.choice(indices_xy[:, 0].tolist())
            y = y_min + np.random.choice(indices_xy[:, 1].tolist())
            result = np.all(np.isin(label_data[x:x+s[0], y:y+s[1], z:z+s[2]], label))
            if result:
                num += 1
                arr = np.append(arr, np.array([[x, y, z]]), axis=0)  # Pixel position under labelmap coordinate frame
                print('found at ', x, ',', y, ',', z)
                break

    # THIS BIT IS WITHIN IMARIS
    """#this arr contains the location in img 2 (label map). Each row is the 3D position of starting point of each cube.
    #but we are counting the cells under img (NeuN), so we need to convert the location to img.
    # TODO: hard-coded voxel sizes, these should automatically be inferred from the label nrrd volume"""
    # the pixel position relative to NeuN frame (starting origin)
    loc2 = (arr * voxel_sizes + vExtentMin2 - vExtentMin) / np.array([1.8, 1.8, 4])
    print("region starting point: {}".format(loc2), flush=True)
    #subvol = img2.GetDataSubVolumeShorts(0,0,250,0,0,700,1000,2)
    for i in range(N):
        subvol = img.GetDataSubVolumeShorts(loc2[i, 0], loc2[i, 1], loc2[i, 2], 0, 0, 56, 56, 25)  #the size of subvolume is (56,56,25) pixels with resolution (1.8,1.8,4)um.
        subvol = np.transpose(subvol, [2, 1, 0])  #Swap the dimensions because Imaris export will swap X and Z.
        #np.array(subvol).astype('int16').tofile(folder+'Region0.raw') #works
        print(type(subvol))
        # this filename variable is only used right here, in the next two lines.
        filename = tif_dir + 'Region_' + str(i) + '.tif'
        imwrite(filename, np.float32(subvol), imagej=True,
                metadata={'spacing': 4, 'unit': 'um', 'axes': 'ZYX'},
                resolution = (1/1.8, 1/1.8))  # ZYX axis is required by Imagej. Without ImageJ=True, there will be some annoying options and changing 'axes' doesn't work either.

    ij.ui().showUI()
    result = ij.py.run_macro(ij_macro,args)


    print("use Imagej now!")
    #volume_bar = 20 #8um ^ 3 / 4um / 1.8^2 um^2 ~ 35 pixel. Common neurons are larger than this. This is the size threshold to filter out the small components
    #volume_avgbar=100 #This number is the volume/pixel of a typical neuron and depends on brain regions. I suggest users observe the size of the neurons of tiff files before deciding this number.
    num_neuron = []
    for i in range(N):
        image = imread(tif_processed_dir + "morpho_" + str(i) + ".tif")
        N_n = image.max()  # the number of labels i.e. the number of individual neurons
        for j in range(1, image.max() + 1):  # j here is the integer label, range in [1,max]
            if image.size - np.count_nonzero(image-j) > 10*volume_avgbar:
                N_n = N_n - 1  # if the component is too large, then it's unlikely to be neuron. Should be deleted from the count.
            elif image.size - np.count_nonzero(image-j) < volume_bar:
                N_n = N_n - 1  # if too small, count --1
            elif image.size - np.count_nonzero(image-j) > volume_avgbar:
                N_n = N_n + np.floor((image.size - np.count_nonzero(image-j))/volume_avgbar) - 1  # if the volume is several times bigger than 1 typical neuron, then the volume/typical volume is regarded as the actual neuron numbers in the blob
        num_neuron.append(N_n)


    np.savetxt(out_csv_file, num_neuron, fmt='%d', delimiter="\n")


def main():
    label_imaris_path = sys.argv[1]
    label_nhdr_path = sys.argv[2]
    image_imaris_path = sys.argv[3]
    label_dict_path = sys.argv[4]
    if len(sys.argv) > 5:
        work_dir = sys.argv[5]
    else:
        dirname = os.path.dirname(image_imaris_path)
        # TODO: grab spec id and contrast out of the file path
        spec_id = "201026-1_1"
        contrast = "PV"
        temp = os.path.basename(image_imaris_path)
        # trim .ims off of the filename if the filename ends with .ims. Else do nothing
        temp = temp[:-4] if temp.endswith('.ims') else temp
        temp_split = temp.split("_")
        spec_id = "{}_{}".format(temp_split[0], temp_split[1])
        contrast = temp_split[-1]
        work_dir = "{}/NeuronCounting/{}/{}".format(dirname, spec_id, contrast)
    print(work_dir); exit()

    with open(label_dict_path,"r") as f:
        regions = json.load(f)

    launch_imaris_and_open_data(image_imaris_path, label_imaris_path, label_nhdr_path, regions, work_dir)

if __name__ == "__main__":
    main()

