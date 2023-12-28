import subprocess
#subprocess.call(['C:\\Program Files\\Bitplane\\Imaris 9.9.0\\Imaris.exe', 'id101'])
import ImarisLib
import numpy as np
import math as m
import random
import nrrd
from tifffile import imwrite
import sys
sys.path.insert(0, 'k:/workstation/code/shared/pipeline_utilities/imaris')
#K:/workstation/code/shared/pipeline_utilities/imaris/imaris_hdr_update.py
import imaris_hdr_update
from skimage.io import imread
import os
from time import sleep
import shutil
import imagej
ij = imagej.init(r'K:\CIVM_Apps\Fiji.app',mode='interactive')


exec(open(r"K:\ProjectSpace\yt133\codes\Compilation of cell counting\5xFAD\CountingCodes\Auto_ver\ij_classifier.py").read())

fiji=r"K:\CIVM_Apps\Fiji.app\ImageJ-win64.exe".replace('\\','/')
macro_path="S:/yt133/To_delete_Statistics/macroscript.ijm" #where you save your macro

#root needs to be updated with different specimens
root="S:/yt133/To_delete_Statistics/history/data_5xFAD_neuron_density/220114-3_1/" #working folder

#filename needs to be updated with specimen label
#import the labelmap to locate a brain region, within this brain region, generate N random subvolumes with size s (/pixels)

#filename=r"B:\20.5xfad.01\BXD77\220114-1_1\Aligned-Data\labels\RCCF\N59128NLSAM_labels.nhdr".replace("\\","/")#change this when change specimen
#filename=r"B:\20.5xfad.01\BXD77\220114-2_1\Aligned-Data\labels\RCCF\N59130NLSAM_labels.nhdr".replace("\\","/")#change this when change specimen
filename=r"B:\20.5xfad.01\BXD77\220114-3_1\Aligned-Data\labels\RCCF\N59132NLSAM_labels.nhdr".replace("\\","/")#change this when change specimen

#filename=r"B:\20.5xfad.01\BXD77\220114-20_1\Aligned-Data\labels\RCCF\N60206NLSAM_labels.nhdr".replace("\\","/")#change this when change specimen
#filename=r"B:\20.5xfad.01\BXD77\220114-21_1\Aligned-Data\labels\RCCF\N60208NLSAM_labels.nhdr".replace("\\","/")#change this when change specimen
#filename=r"B:\20.5xfad.01\BXD77\220114-22_1\Aligned-Data\labels\RCCF\N60215NLSAM_labels.nhdr".replace("\\","/")#change this when change specimen
#filename=r"B:\20.5xfad.01\BXD77\220114-23_1\Aligned-Data\labels\RCCF\N60213NLSAM_labels.nhdr".replace("\\","/")#change this when change specimen
im, header = nrrd.read(filename)
im = im[::-1, ::-1, :].astype(int) # this im will be sent as a variable


def GetServer():
   vImarisLib = ImarisLib.ImarisLib()
   vServer = vImarisLib.GetServer()
   return vServer;

vServer=GetServer()
vImarisLib=ImarisLib.ImarisLib()
v = vImarisLib.GetApplication(101)
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

"""
Yuqi thinks it's better to make this below:
Write the script as a function, and make the regions as a large dictionary
"""
def mainAlgor(label,classifier, N = 10, volume_bar = 20, volume_avgbar = 100):
### label: list, classifier: path str, volume_bar: a num that represents the average volume size
### newdir: where the file will be saved



  if len(label)==1:
    newdir=root+str(label[0])+"/"
  elif len(label)>1:
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

  indices = np.argwhere(np.isin(im, label))  # Check if elements in 'im' are in 'label'
  print(indices[:, 2])
  x_min = min(indices[:, 0])
  x_max = max(indices[:, 0])
  y_min = min(indices[:, 1])
  y_max = max(indices[:, 1])
  z_min = min(indices[:, 2])
  z_max = max(indices[:, 2])
  arr = np.empty((0, 3), int)

  while num < N:
      z = int(np.floor(z_min + np.random.rand() * (z_max - s[2] - z_min)))
      if z not in np.squeeze(indices[:, 2]):
          continue
      for iteration in range(10):
          im_xy = im[x_min:x_max, y_min:y_max, z]
          indices_xy = np.argwhere(np.isin(im_xy, label))
          if indices_xy.size == 0:
              continue
          x = x_min + np.random.choice(indices_xy[:, 0].tolist())
          y = y_min + np.random.choice(indices_xy[:, 1].tolist())
          result = np.all(np.isin(im[x:x+s[0], y:y+s[1], z:z+s[2]], label))
          if result:
              num += 1
              arr = np.append(arr, np.array([[x, y, z]]), axis=0)  # Pixel position under labelmap coordinate frame
              print('found at ', x, ',', y, ',', z)
              break

  #this arr contains the location in img 2 (label map). Each row is the 3D position of starting point of each cube.
  #but we are counting the cells under img (NeuN), so we need to convert the location to img.




  loc2=(arr*[25,25,25]+vExtentMin2-vExtentMin)/np.array([1.8,1.8,4])#the pixel position relative to NeuN frame (starting origin)

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
regions = {
    "Orbital": {
        "labels": [6],
        "classifier": r"K:\ProjectSpace\yt133\Labelmap\200316auditory\short1.classifier".replace('\\','/'),
        "volume_bar": 20,
        "volume_avgbar": 100
    },
    "PrimarySomatosensory": {
        "labels": [16],
        "classifier":  r"K:\ProjectSpace\yt133\Labelmap\200316auditory\short1.classifier".replace('\\','/'),
        "volume_bar": 20,
        "volume_avgbar": 100
    },
    "SupplementalSomatosensory": {
        "labels": [18],
        "classifier":  r"K:\ProjectSpace\yt133\Labelmap\200316auditory\short1.classifier".replace('\\','/'),
        "volume_bar": 20,
        "volume_avgbar": 100
    },
    "Auditory": {
        "labels": [20],
        "classifier": r"K:\ProjectSpace\yt133\Labelmap\200316auditory\short1.classifier".replace('\\','/'),
        "volume_bar": 20,
        "volume_avgbar": 100
    },
    "Retroplenial": {
        "labels": [24],
        "classifier": r"K:\ProjectSpace\yt133\Labelmap\5xFAD\03subiculum.classifier".replace('\\','/'),
        "volume_bar": 15,
        "volume_avgbar": 100
    },
    "PrimaryVisualArea": {
        "labels": [26],
        "classifier": r"K:\ProjectSpace\yt133\Labelmap\200316auditory\short1.classifier".replace('\\','/'),
        "volume_bar": 20,
        "volume_avgbar": 100
    },
    "Entorhinal": {
        "labels": [27],
        "classifier": r"K:\ProjectSpace\yt133\Labelmap\200316auditory\short1.classifier".replace('\\','/'),
        "volume_bar": 20,
        "volume_avgbar": 100
    },
    "Subiculum": {
        "labels": [28],
        "classifier": r"K:\ProjectSpace\yt133\Labelmap\5xFAD\03subiculum.classifier".replace('\\','/'),
        "volume_bar": 20,
        "volume_avgbar": 100
    },
    "CA1": {
        "labels": [31],
        "classifier": r"K:\ProjectSpace\yt133\Labelmap\5xFAD\03subiculum.classifier".replace('\\','/'),
        "volume_bar": 10,
        "volume_avgbar": 100
    },
    "CA3": {
        "labels": [32],
        "classifier": r"K:\ProjectSpace\yt133\Labelmap\5xFAD\03subiculum.classifier".replace('\\','/'),
        "volume_bar": 10,
        "volume_avgbar": 100
    },
    "BLA": {
        "labels": [41],
        "classifier":r"K:\ProjectSpace\yt133\Labelmap\191209_BLA\BLAc_.classifier".replace('\\','/'),
        "volume_bar": 10,
        "volume_avgbar": 100
    },
    "LGd": {
        "labels": [82],
        "classifier": r"K:\ProjectSpace\yt133\Labelmap\200316auditory\short1.classifier".replace('\\','/'),
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
        # "classifier": r"K:\ProjectSpace\yt133\Labelmap\200316auditory\short1.classifier".replace('\\','/'),
        # "volume_bar": 20,
        # "volume_avgbar": 100
    # }
}


for region_name in regions:
  label = regions[region_name]["labels"]
  classifier = regions[region_name]["classifier"]
  volume_bar = regions[region_name]["volume_bar"]
  volume_avgbar = regions[region_name]["volume_avgbar"]
  mainAlgor(label, classifier, N = 30, volume_bar = volume_bar, volume_avgbar = volume_avgbar)
