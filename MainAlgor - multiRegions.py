import subprocess;
#subprocess.call(['C:\\Program Files\\Bitplane\\Imaris 10.0.0\\Imaris.exe', 'id101']);
import numpy as np;
import math as m;
import random;
import nrrd;
from tifffile import imwrite;
import sys;
# these append statements are required to correctly find ImarisLib and all of its dependencies
# this sys.path.append is the correct way to modify your PYTHONPATH variable
sys.path.append(r'C:\Program Files\Bitplane\Imaris 9.9.0\XT\python3'.replace(r'\\','/'));
sys.path.insert(0, 'k:/workstation/code/shared/pipeline_utilities/imaris');
import ImarisLib;
#K:/workstation/code/shared/pipeline_utilities/imaris/imaris_hdr_update.py;
import imaris_hdr_update;
from skimage.io import imread;
import os;
from time import sleep;
import shutil;
import imagej;


def GetServer():
   vImarisLib = ImarisLib.ImarisLib();
   vServer = vImarisLib.GetServer();
   return vServer;

# TODO: incorporate the above code into the function
# TODO: input arguments to function should be 2 imaris files and 
    # full path to image ims file
    # full path to label ims file
    # dictionary of ROI to count and which classifier to use for that ROI
    
    # out_dir -- should it be set by caller and be an input arg (YES), or should it be decided where to put and created by this function?
"""
Yuqi thinks it's better to make this below:
Write the script as a function, and make the regions as a large dictionary
"""
def mainAlgor(label,classifier, N = 10, volume_bar = 20, volume_avgbar = 100):
    # TODO: "root" is our output folder, it is set at top of the script. this should be an input to this function
    # WHAT DOES VOLUME_VAR AND VOLUME_AVGBAR DO? IMPORTANT?
    ### label: list, classifier: path str, volume_bar: a num that represents the average volume size
    ### newdir: where the file will be saved


    ij = imagej.init(r'K:\CIVM_Apps\Fiji.app',mode='interactive');


    #exec(open(r"K:\ProjectSpace\yt133\codes\Compilation of cell counting\5xFAD\CountingCodes\Auto_ver\ij_classifier.py").read())
    exec(open(r"K:\workstation\code\shared\img_processing\NeuronCounting\ij_classifier.py").read())


    fiji=r"K:\CIVM_Apps\Fiji.app\ImageJ-win64.exe".replace('\\','/')
    macro_path=r"K:\workstation\code\shared\img_processing\NeuronCounting\ij_macro\macroscript.ijm" #where you save your macro

    #root needs to be updated with different specimens
    # INPUT ARGUMENTS GET RID OF THESE!!
    project_code="22.gaj.49";
    strain="BXD77";
    specimen_id="DMBA";
    contrast="NeuN";
    runno="N59128NLSAM";

    root="B:/{}/DMBA/ims/LSFM/NeuronCounting/{}/{}".format(project_code, specimen_id, contrast); #working folder
    os.makedirs(root,exist_ok=True);

    #pattern for 20.5xFAD
    # contrast is not in the label filename 
    #filename="B:\{}\{}\{}\Aligned-Data\labels\RCCF\{}_labels.nhdr".format(project_code,strain,specimen_id,contrast,runno)
    filename="B:\{}\{}\{}\Aligned-Data\labels\RCCF\{}_labels.nhdr".format(project_code,strain,specimen_id,runno)

    #pattern for 22.gaj.49
    # contrast is not in the label filename
    #filename="B:\{}\{}\Aligned-Data\labels\RCCF\DMBA_RCCF_labels.nhdr".format(project_code,"DMBA",contrast)
    filename="B:\{}\{}\Aligned-Data\labels\RCCF\DMBA_RCCF_labels.nhdr".format(project_code,"DMBA")

    #filename needs to be updated with specimen label
    #import the labelmap to locate a brain region, within this brain region, generate N random subvolumes with size s (/pixels)
    # im changed to label_data, header changed to label_nhdr
    # i think label_nhdr is never used?
        # USE IT TO GET VOXEL SIZES
    label_data, label_nhdr = nrrd.read(filename)

    # this indexing starts from the end and goes all the way. i.e. reverse the first 2 dimensions and keep the third the same
    label_data = label_data[::-1, ::-1, :].astype(int) # this im will be sent as a variable

    vServer=GetServer()
    vImarisLib=ImarisLib.ImarisLib()

    # TOOD: possible to ping the system to ask for application numbers of running applications named "Imaris"
    # this way, we could open and run imaris in any way we want, get it ready to go, and then
    v = vImarisLib.GetApplication(101)

    # how do we know that img 0 is the NeuN and imgg1 is the label set? could it be the other way around?
    # ORDER of files loaded in is very important
    # possible to foolproof this? get image object, if "label" is in its name, then tha is img2 and the other is img1.
    # if there are more than 2 loaded volumes, then quit not knowing what to do

    # use v to open the LSFM file and then the label file
    # empty string is the default, you must include something
    # aOptions -- [in]  Set up extra options to specify file format, resampling or cropping parameters for loading. 
        # Use "" for default options (automatic file type detection, no cropping and no resampling) 
        # - reader="Imaris3" Advice to use the "Bitplane Imaris 3" format. There are several other formats available:  All Formats, Imaris5, Imaris3, Imaris, AndorIQ, Andor, DeltaVision, Biorad, IPLab, IPLabMac, Gatan, CXD, SlideBook, MRC, LeicaLif, LeicaSingle, LeicaSeries, LeicaVista, MicroManager, MetamorphSTK, MetamorphND, ICS, NikonND2, OlympusCellR, OlympusOIB, OlympusOIF, Olympus, OlympusVSI, OmeTiff, OmeXml, OpenlabLiff, OpenlabRaw, PerkinElmer2, Prairie, Till, AxioVision, Lsm510, Lsm410, BmpSeries, TiffSeries, ZeissCZI. 
        # - croplimitsmin="x0 y0 z0 c0 t0" Minimum crop position for x, y, z, ch and t. Use "0 0 0 0 0" for croplimitsmin and croplimitsmax to disable cropping. 
        # - croplimitsmax="x y z c t" The point next to the maximum crop position for x, y, z, ch and t. If one of the components is set to zero, the maximum crop position is automatically set to the size of the dataset along the corresponding dimension.  
        # - resample="rx ry rz rc rt" 
        # - LoadDataSet="eDataSetYes | eDataSetNo | eDataSetWithDialog" Down-sampling factor for x, y, z, ch and t. "1 1 1 1 1" does not resample. The size of the loaded dataset is equal to "(croplimitsmax - croplimitsmin) / resample" along each dimension (e.g. "(x-x0)/rx"). %% The following MATLAB code opens the example image "retina.ims" with the Imaris3 reader, with cropping and resampling enabled for x and y (while loading all slices, all channels, all time points) vImarisApplication.FileOpen('images\\retina.ims', ... ['reader="Imaris3" ' ... 'croplimitsmin="10 10 0 0 0" ' ... 'croplimitsmax="150 100 0 0 0" ' ... 'resample="2 2 1 1 1"']);
        # EXAMPLE CALL:
            #  vImarisApplication.FileOpen('images\\retina.ims', ... ['reader="Imaris3" ' ... 'croplimitsmin="10 10 0 0 0" ' ... 'croplimitsmax="150 100 0 0 0" ' ... 'resample="2 2 1 1 1"']);
    load_options = ""
    # in_file must be of format B:/path/to/file
        # B:\path\to\file (with backslashes) does not work
        # /b/path/to/file (full unix way) does not work
    #in_file = "B:/22.gaj.49/DMBA/ims/DMBA_dwi.ims"
    #v.FileOpen(in_file, load_options)

    # PROBLEM: this works, but when I use it twice in a row, it automatically deletes the previous file that is loaded. 
    # BOOO, this also does NOT WORK:
        # open the first volume on loading of imaris with the subprocess call
            # subprocess.call(['C:\\Program Files\\Bitplane\\Imaris 10.0.0\\Imaris.exe', 'B:\\22.gaj.49\\DMBA\\ims\\DMBA_dwi.ims', 'id101']);
        # use v.FileOpen() to load in the second volume
        # this gives the same result of v.FileOpen() clearing the previously loaded volume before loading in the new one. result is still only 1 volume loaded into the scene


    img = v.GetImage(0)#load NeuN
    img2 = v.GetImage(1)#load labelmap data
    #print(type(img2))
    vExtentMin=[img.GetExtendMinX(),img.GetExtendMinY(),img.GetExtendMinZ()]
    vExtentMax=[img.GetExtendMaxX(),img.GetExtendMaxY(),img.GetExtendMaxZ()]

    vExtentMin2 =[img2.GetExtendMinX(),img2.GetExtendMinY(),img2.GetExtendMinZ()]
    print('vExtentMin:',vExtentMin)
    print('vExtentMax:',vExtentMax)
    aChannel=0


    """
    classifiers={
    "K:\abababa.classifier" : [ 1, 2, 3 ],
    }
    for classer_file in classifiers:
    label=classifiers[classer_file]
    """

    if len(label)==1:
        newdir=root+str(label[0])+"/"
    elif len(label)>1:
        # this naming makes an assumption that when analyzing muyltipl regions at once, that THEY ARE SEQUENTIAL. what if we wanted toi use rois [5,12,41,99]
        newdir=root+str(label[0])+'-'+str(label[-1])+'/'
    if not os.path.exists(newdir):
        os.makedirs(newdir)
    folder=newdir+"Tif/"
    output=newdir+"tif_out/"
    processed=newdir+"tif_processed/"

    if not os.path.exists(folder):
        os.makedirs(folder)
    if not os.path.exists(output):
        os.makedirs(output)
    if not os.path.exists(processed):
        os.makedirs(processed)

    print(f" folder={folder} output={output} processed={processed} macro_path={macro_path}")
    ij_macro,args = save_classifier_macro(folder,output,processed,classifier,macro_path,N=N)

    num=0;
    s=[6,6,6] # Make sure this cube is larger than subvolume. The current cube size is (15um * 10)^3


    # argwhere finds the indices of array elements that are non-zero, grouped by element.
    # this line finds all indices (of the numpy array holding our LABEL file)
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

    #this arr contains the location in img 2 (label map). Each row is the 3D position of starting point of each cube.
    #but we are counting the cells under img (NeuN), so we need to convert the location to img.


    # TODO: hard-coded voxel sizes, these should automatically be inferred from the label nrrd volume
    loc2=(arr*[15,15,15]+vExtentMin2-vExtentMin)/np.array([1.8,1.8,4])#the pixel position relative to NeuN frame (starting origin)
    #loc2=(arr*[25,25,25]+vExtentMin2-vExtentMin)/np.array([1.8,1.8,4])#the pixel position relative to NeuN frame (starting origin)
    print(loc2)
    #subvol = img2.GetDataSubVolumeShorts(0,0,250,0,0,700,1000,2)
    for i in range(N):
        subvol = img.GetDataSubVolumeShorts(loc2[i,0],loc2[i,1],loc2[i,2],0,0,56,56,25) #the size of subvolume is (56,56,25) pixels with resolution (1.8,1.8,4)um.
        subvol=np.transpose(subvol,[2,1,0])#Swap the dimensions because Imaris export will swap X and Z.
        #np.array(subvol).astype('int16').tofile(folder+'Region0.raw') #works
        print(type(subvol))
        filename = folder+'Region_'+str(i)+'.tif'
        imwrite(filename,np.float32(subvol),imagej=True,
        metadata={'spacing': 4,'unit': 'um','axes': 'ZYX'},
        resolution=(1/1.8, 1/1.8))#ZYX axis is required by Imagej. Without ImageJ=True, there will be some annoying options and changing 'axes' doesn't work either.


    #ij = imagej.init(r'K:\CIVM_Apps\Fiji.app',mode='interactive')
    ij.ui().showUI()
    result = ij.py.run_macro(ij_macro,args)


    print("use Imagej now!")
    #volume_bar = 20 #8um ^ 3 / 4um / 1.8^2 um^2 ~ 35 pixel. Common neurons are larger than this. This is the size threshold to filter out the small components
    #volume_avgbar=100 #This number is the volume/pixel of a typical neuron and depends on brain regions. I suggest users observe the size of the neurons of tiff files before deciding this number.
    num_neuron = []
    for i in range(N):
        image = imread(processed+"morpho_"+str(i)+".tif")
        N_n=image.max() #the number of labels i.e. the number of individual neurons
        for j in range(1,image.max()+1): #j here is the integer label, range in [1,max]
            if image.size - np.count_nonzero(image-j)> 10*volume_avgbar:
                N_n = N_n - 1 #if the component is too large, then it's unlikely to be neuron. Should be deleted from the count.
            elif image.size - np.count_nonzero(image-j)<volume_bar:
                N_n = N_n -1 #if too small, count --1
            elif image.size - np.count_nonzero(image-j)>volume_avgbar:
                N_n = N_n + np.floor((image.size - np.count_nonzero(image-j))/volume_avgbar) -1 #if the volume is several times bigger than 1 typical neuron, then the volume/typical volume is regarded as the actual neuron numbers in the blob
        num_neuron.append(N_n)

    if len(label)==1:
        np.savetxt(root+str(label[0])+"_counts.csv", num_neuron,fmt='%d', delimiter="\n")
    elif len(label)>1:
        np.savetxt(root+str(label[0])+'-'+str(label[-1])+"_counts.csv", num_neuron,fmt='%d', delimiter="\n")

    shutil.move(processed,newdir+"tif_processed/")
    shutil.move(output,newdir+"tif_out/")
    shutil.move(folder,newdir+"Tif/")



#Above is the function. Below is defining all regions and call function
# TODO: maybe move this dict into its own file?
regions = {
    "Orbital": {
        "labels": [6],
        "classifier": r"K:\workstation\code\shared\img_processing\NeuronCounting\classifiers\200316auditory\short1.classifier".replace('\\','/'),
        "volume_bar": 20,
        "volume_avgbar": 100
    },
    "PrimarySomatosensory": {
        "labels": [16],
        "classifier":  r"K:\workstation\code\shared\img_processing\NeuronCounting\classifiers\200316auditory\short1.classifier".replace('\\','/'),
        "volume_bar": 20,
        "volume_avgbar": 100
    },
    "SupplementalSomatosensory": {
        "labels": [18],
        "classifier":  r"K:\workstation\code\shared\img_processing\NeuronCounting\classifiers\200316auditory\short1.classifier".replace('\\','/'),
        "volume_bar": 20,
        "volume_avgbar": 100
    },
    "Auditory": {
        "labels": [20],
        "classifier": r"K:\workstation\code\shared\img_processing\NeuronCounting\classifiers\200316auditory\short1.classifier".replace('\\','/'),
        "volume_bar": 20,
        "volume_avgbar": 100
    },
    "Retroplenial": {
        "labels": [24],
        "classifier": r"K:\workstation\code\shared\img_processing\NeuronCounting\classifiers\5xFAD\03subiculum.classifier".replace('\\','/'),
        "volume_bar": 15,
        "volume_avgbar": 100
    },
    "PrimaryVisualArea": {
        "labels": [26],
        "classifier": r"K:\workstation\code\shared\img_processing\NeuronCounting\classifiers\200316auditory\short1.classifier".replace('\\','/'),
        "volume_bar": 20,
        "volume_avgbar": 100
    },
    "Entorhinal": {
        "labels": [27],
        "classifier": r"K:\workstation\code\shared\img_processing\NeuronCounting\classifiers\200316auditory\short1.classifier".replace('\\','/'),
        "volume_bar": 20,
        "volume_avgbar": 100
    },
    "Subiculum": {
        "labels": [28],
        "classifier": r"K:\workstation\code\shared\img_processing\NeuronCounting\classifiers\5xFAD\03subiculum.classifier".replace('\\','/'),
        "volume_bar": 20,
        "volume_avgbar": 100
    },
    "CA1": {
        "labels": [31],
        "classifier": r"K:\workstation\code\shared\img_processing\NeuronCounting\classifiers\5xFAD\03subiculum.classifier".replace('\\','/'),
        "volume_bar": 10,
        "volume_avgbar": 100
    },
    "CA3": {
        "labels": [32],
        "classifier": r"K:\workstation\code\shared\img_processing\NeuronCounting\classifiers\5xFAD\03subiculum.classifier".replace('\\','/'),
        "volume_bar": 10,
        "volume_avgbar": 100
    },
    "BLA": {
        "labels": [41],
        "classifier":r"K:\workstation\code\shared\img_processing\NeuronCounting\classifiers\191209_BLA\BLAc_.classifier".replace('\\','/'),
        "volume_bar": 10,
        "volume_avgbar": 100
    },
    "LGd": {
        "labels": [82],
        "classifier": r"K:\workstation\code\shared\img_processing\NeuronCounting\classifiers\200316auditory\short1.classifier".replace('\\','/'),
        "volume_bar": 20,
        "volume_avgbar": 100
    },
    "Thalamus": {
        "labels": [58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86],
        "classifier": r"K:/ProjectSpace/yt133/Labelmap/191209_BLA/BLAc_.classifier".replace('\\','/'),
        "volume_bar": 10,
        "volume_avgbar": 100
    },
    #Something weird happens with Thalamus, a lot of java log, so Yuqi set up the last entry.
    # "delete": {
        # "labels": [83],
        # "classifier": r"K:\workstation\code\shared\img_processing\NeuronCounting\classifiers\200316auditory\short1.classifier".replace('\\','/'),
        # "volume_bar": 20,
        # "volume_avgbar": 100
    # }
}

# TODO: create a main function here
for region_name in regions:
  label = regions[region_name]["labels"]
  classifier = regions[region_name]["classifier"]
  volume_bar = regions[region_name]["volume_bar"]
  volume_avgbar = regions[region_name]["volume_avgbar"]
  mainAlgor(label, classifier, N = 30, volume_bar = volume_bar, volume_avgbar = volume_avgbar)
