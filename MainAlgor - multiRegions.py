import subprocess;
import time;
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

import ij_classifier


def GetServer():
   vImarisLib = ImarisLib.ImarisLib();
   vServer = vImarisLib.GetServer();
   return vServer;

def launch_imaris_and_open_data(image_imaris_path: str, label_imaris_path: str, label_nhdr_path: str, roi_dict: dict, work_dir: str):
    # give imaris enough time to open
    #subprocess.call(['C:\\Program Files\\Bitplane\\Imaris 10.0.0\\Imaris.exe', 'id101']);
    # Popen is non blocking
    imaris_process = subprocess.Popen(['C:\\Program Files\\Bitplane\\Imaris 10.0.0\\Imaris.exe', 'id101']);
    time.sleep(10);

    # initialize python-to-imaris bridge
    vServer=GetServer()
    vImarisLib=ImarisLib.ImarisLib()
    v = vImarisLib.GetApplication(101)

    # initialize python-to-imagej bridge
    ij = imagej.init(r'K:\CIVM_Apps\Fiji.app',mode='interactive');

    # load_options allows you to select which imagetype loader to use
    # also lets you force cropping or resampling on load
    # FileOpen will for a scene clear before it loads in the new volume. Cannot be used to load in multiple imaris files into the same scene at the same time
    load_options = "";
    v.FileOpen(label_imaris_path, load_options);
    img2 = v.GetImage(0)
    vExtentMin2 =[img2.GetExtendMinX(),img2.GetExtendMinY(),img2.GetExtendMinZ()]

    v.FileOpen(image_imaris_path, load_options);
    img = v.GetImage(0)
    vExtentMin=[img.GetExtendMinX(),img.GetExtendMinY(),img.GetExtendMinZ()]
    vExtentMax=[img.GetExtendMaxX(),img.GetExtendMaxY(),img.GetExtendMaxZ()]

    print('label vExtentMin:',vExtentMin2);
    print('image vExtentMin:',vExtentMin);
    print('image vExtentMax:',vExtentMax, flush=True);

    for region_name in roi_dict:
        label = roi_dict[region_name]["labels"]
        classifier = roi_dict[region_name]["classifier"]
        volume_bar = roi_dict[region_name]["volume_bar"]
        volume_avgbar = roi_dict[region_name]["volume_avgbar"]
        try:
            # what does an error actually mean in this case?
            mainAlgor(label_nhdr_path, label, classifier, work_dir, N = 30, volume_bar = volume_bar, volume_avgbar = volume_avgbar, vExtentMin=vExtentMin, vExtentMax=vExtentMax, vExtentMin2=vExtentMin2, img=img, ij=ij)
        except:
            # this does not really work. I got an error, but I think (?) that the imagej process didn't die itself and tried to keep going.
            print("ERROR in neuron counting for ROI: {}".format(region_name))
            print("BURYING HEAD IN THE SAND. KEEP GOING TO THE NEXT REGION!", flush=True)
            # nope continue doesn't work either
            # there is something hanging up and causing other things to fail. maybe it's the imagej process?
            continue;
            #imaris_process.kill()

    # very unsure how to handle this
    imaris_process.kill()



"""
# TODO: incorporate the above code into the function
# TODO: input arguments to function should be 2 imaris files and 
    # full path to image ims file
    # full path to label ims file
    # dictionary of ROI to count and which classifier to use for that ROI
    
    # out_dir -- should it be set by caller and be an input arg (YES), or should it be decided where to put and created by this function?

Yuqi thinks it's better to make this below:
Write the script as a function, and make the regions as a large dictionary
"""
def mainAlgor(label_nhdr_file: str, label: list, classifier, out_dir: str, N: int = 10, volume_bar: int = 20, volume_avgbar: int = 100, vExtentMin=None, vExtentMax=None, vExtentMin2=None, img=None, vServer=None, vImarisLib=None, ij=None):
#def mainAlgor(label,classifier, N = 10, volume_bar = 20, volume_avgbar = 100):
    """# WHAT DOES VOLUME_VAR AND VOLUME_AVGBAR DO? IMPORTANT?
    ### label: list, classifier: path str, volume_bar: a num that represents the average volume size
    ### newdir: where the file will be saved"""

    # this gives us a "gateway" into ImageJ
    # maybe it is bad for me to reinitialize this every single loop? treat it like you treat the Imaris gateway.
    if ij is None:
        print("No ImageJ bridge found. Reinitializing now.", flush=True)
        ij = imagej.init(r'K:\CIVM_Apps\Fiji.app',mode='interactive');


    #exec(open(r"K:\ProjectSpace\yt133\codes\Compilation of cell counting\5xFAD\CountingCodes\Auto_ver\ij_classifier.py").read())
    #exec(open(r"K:\workstation\code\shared\img_processing\NeuronCounting\ij_classifier.py").read())
    #fiji=r"K:\CIVM_Apps\Fiji.app\ImageJ-win64.exe".replace('\\','/')
    
    macro_path=r"K:\workstation\code\shared\img_processing\NeuronCounting\ij_macro\macroscript.ijm" #where you save your macro

    # TODO: use the label_nhdr file to get the voxel resolution of label files. this number is currently hard-coded below
    label_data, label_nhdr = nrrd.read(label_nhdr_file)

    # this indexing starts from the end and goes all the way. i.e. reverse the first 2 dimensions and keep the third the same
    label_data = label_data[::-1, ::-1, :].astype(int) # this im will be sent as a variable

    # i don't think we ever actually use this after the initialization?
    if vServer is None or vImarisLib is None:
        print("my vServer was None! reiniitalizing connection to Imaris", flush=True);
        vServer=GetServer()
        vImarisLib=ImarisLib.ImarisLib()
        # TOOD: possible to ping the system to ask for application numbers of running applications named "Imaris"
        # this way, we could open and run imaris in any way we want, get it ready to go, and then
        v = vImarisLib.GetApplication(101)

    if len(label)==1:
        newdir=out_dir+str(label[0])+"/"
    elif len(label)>1:
        # this naming makes an assumption that when analyzing muyltipl regions at once, that THEY ARE SEQUENTIAL. what if we wanted toi use rois [5,12,41,99]
        newdir=out_dir+str(label[0])+'-'+str(label[-1])+'/'
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

    print("imagej macro arguments:")
    print(f" folder={folder} output={output} processed={processed} macro_path={macro_path}", flush=True)
    ij_macro,args = ij_classifier.save_classifier_macro(folder,output,processed,classifier,macro_path,N=N)

    num=0;
    s=[6,6,6] # Make sure this cube is larger than subvolume. The current cube size is (15um * 10)^3


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
    loc2=(arr*[15,15,15]+vExtentMin2-vExtentMin)/np.array([1.8,1.8,4])#the pixel position relative to NeuN frame (starting origin)
    #loc2=(arr*[25,25,25]+vExtentMin2-vExtentMin)/np.array([1.8,1.8,4])#the pixel position relative to NeuN frame (starting origin)
    print("region starting point: {}".format(loc2), flush=True)
    #subvol = img2.GetDataSubVolumeShorts(0,0,250,0,0,700,1000,2)
    for i in range(N):
        subvol = img.GetDataSubVolumeShorts(loc2[i,0],loc2[i,1],loc2[i,2],0,0,56,56,25) #the size of subvolume is (56,56,25) pixels with resolution (1.8,1.8,4)um.
        subvol=np.transpose(subvol,[2,1,0])#Swap the dimensions because Imaris export will swap X and Z.
        #np.array(subvol).astype('int16').tofile(folder+'Region0.raw') #works
        print(type(subvol))
        # this filename variable is only used right here, in the next two lines. 
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
        np.savetxt(out_dir+str(label[0])+"_counts.csv", num_neuron,fmt='%d', delimiter="\n")
    elif len(label)>1:
        np.savetxt(out_dir+str(label[0])+'-'+str(label[-1])+"_counts.csv", num_neuron,fmt='%d', delimiter="\n")

    shutil.move(processed,newdir+"tif_processed/")
    shutil.move(output,newdir+"tif_out/")
    shutil.move(folder,newdir+"Tif/")



#Above is the function. Below is defining all regions and call function
# TODO: maybe move this dict into its own file?
regions = {
#    "Orbital": {
#        "labels": [6],
#        "classifier": r"K:\workstation\code\shared\img_processing\NeuronCounting\classifiers\200316auditory\short1.classifier".replace('\\','/'),
#        "volume_bar": 20,
#        "volume_avgbar": 100
#    },
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




"""# label nhdr file. used for grabbing voxel indices of a certain ROI
#root needs to be updated with different specimens
# INPUT ARGUMENTS GET RID OF THESE!!
project_code="22.gaj.49";
strain="BXD77";
specimen_id="DMBA";
contrast="PV";
runno="N59128NLSAM";
# pattern for 20.5xFAD
label_nhdr_path="B:\{}\{}\{}\Aligned-Data\labels\RCCF\{}_labels.nhdr".format(project_code,strain,specimen_id,runno)
# pattern for 22.gaj.49
label_nhdr_path="B:\{}\{}\Aligned-Data\labels\RCCF\DMBA_RCCF_labels.nhdr".format(project_code,"DMBA")"""


# hard-coded test data
label_imaris_path = "B:/22.gaj.49/DMBA/ims/labels/RCCF/DMBA_RCCF_labels.ims";
label_nhdr_path = "B:/22.gaj.49/DMBA/Aligned-Data-RAS/labels/RCCF/DMBA_RCCF_labels.nhdr";
image_imaris_path = "B:/22.gaj.49/DMBA/ims/LSFM/201026-1_1_PV.ims";
work_dir = "B:/{}/DMBA/ims/LSFM/NeuronCounting/{}/{}".format("22.gaj.49", "201026-1_1", "PV");

launch_imaris_and_open_data(image_imaris_path, label_imaris_path, label_nhdr_path, regions, work_dir);

exit()

# TODO: create a main function here
for region_name in regions:
  label = regions[region_name]["labels"]
  classifier = regions[region_name]["classifier"]
  volume_bar = regions[region_name]["volume_bar"]
  volume_avgbar = regions[region_name]["volume_avgbar"]
  mainAlgor(label, classifier, N = 30, volume_bar = volume_bar, volume_avgbar = volume_avgbar)
