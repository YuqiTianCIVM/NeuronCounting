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

macro_path="S:/yt/To_delete_Statistics/macroscript.ijm" #where you save your macro
root="S:/yt/To_delete_Statistics/" #working folder
folder="S:/yt/To_delete_Statistics/Tif/" 
output="S:/yt/To_delete_Statistics/tif_out/"
processed="S:/yt/To_delete_Statistics/tif_processed/"
filename=r"R:\yt\Copy_200316\YT_reg_LS\reverse\N58204_RCCF_label_reverse2.nhdr".replace("\\","/")#this is the labelmap file. change this when change specimen

classifier=r"K:\ProjectSpace\yt\Labelmap\BLA\BLA01.classifier".replace('\\','/') #the labkit classifier
N=30#how many sampling subvolumes
label=41 #label LGd 82,Perirhinal cortex 23,orbital 6, retroplenial 24, auditory 20, subiculum 28, facial motor 126, 122 trigemetor,
#entorhinal 27, CA1 31, CA3 32, Primary visual area 26, thalamus 58 (definition may wrong), BLA 41, dentate gyrus 136
newdir=root+str(label)+"/"
if not os.path.exists(newdir):
    os.makedirs(newdir) 
fiji=r"K:\CIVM_Apps\Fiji.app\ImageJ-win64.exe".replace('\\','/')


if not os.path.exists(folder):
    os.makedirs(folder)
if not os.path.exists(output):
    os.makedirs(output)
if not os.path.exists(processed):
    os.makedirs(processed)




#Save the ij_macro to a macro script and later call this macro_command. 

ij_macro = ''
ij_macro=ij_macro+r'folder="'+folder+'"\n'
ij_macro=ij_macro+r'output="'+output+'"\n'
ij_macro=ij_macro+r'processed="'+processed+'"\n'
ij_macro=ij_macro+r's="'+classifier+r'";'+"\n"
ij_macro=ij_macro+r'N='+str(N)+"\n"
ij_macro=ij_macro+r'for (i = 0; i < N; i++) {'+"\n"
ij_macro=ij_macro+r'    open(folder+"Region_"+i+".tif");'+"\n"
ij_macro=ij_macro+r'    run("Segment Image With Labkit", "segmenter_file="+s+" use_gpu=false");'+"\n"
ij_macro=ij_macro+r'    close("Region_"+i+".tif");'+"\n"
ij_macro=ij_macro+r'    saveAs("Tiff", output + "segmentation_" + i + ".tif");'+"\n"
ij_macro=ij_macro+r'    run("Make Binary", "method=Default background=Default calculate");'+"\n"
ij_macro=ij_macro+r'    run("Watershed","stack");'+"\n"
ij_macro=ij_macro+r'    run("Connected Components Labeling", "connectivity=6 type=[16 bits]");'+"\n"
ij_macro=ij_macro+r'    saveAs("Tiff", processed + "morpho_" + i + ".tif");'+"\n"
ij_macro=ij_macro+r'    close("segmentation_" + i + ".tif");'+"\n"
ij_macro=ij_macro+r'    close("MASK_segmentation_" + i + ".tif");'+"\n"
ij_macro=ij_macro+r'    close("morpho_" + i + ".tif");'+"\n"
ij_macro=ij_macro+r'}'

with open(macro_path,'w') as my_file:
    my_file.write(ij_macro)



#filename=r"R:\yt\Copy_200316\YT_reg_LS\reverse\N58204_RCCF_label_reverse2.nhdr".replace("\\","/")#change this when change specimen
#import the labelmap to locate a brain region, within this brain region, generate N random subvolumes with size s (/pixels)
im, header = nrrd.read(filename)

num=0;
s=[10,10,10] # Make sure this cube is larger than subvolume. The current cube size is (15um * 10)^3

im=im[::-1,::-1,:].astype(int)#remember to check the header. Header fix: flip the image so that the coordinate system goes to positive entity matrix and Right-Anterior-Superior
indices=np.argwhere(im==label)#argwhere output each row as one pixel location
print(indices[:,2]) 
x_min=min(indices[:,0])
x_max=max(indices[:,0])
y_min=min(indices[:,1])
y_max=max(indices[:,1])
z_min=min(indices[:,2])
z_max=max(indices[:,2])
arr = np.empty((0,3), int)


while num<N:
    z=int(np.floor(z_min+np.random.rand()*(z_max - s[2] - z_min)))
    if z not in np.squeeze(indices[:,2]):
        continue
    for iteration in range(10):
        im_xy = im[x_min:x_max,y_min:y_max,z]
        indices_xy = np.argwhere(im_xy==label)
        x = x_min+np.random.choice(indices_xy[:,0].tolist())
        y = y_min+np.random.choice(indices_xy[:,1].tolist())       
        result = np.all(im[x:x+s[0],y:y+s[1],z:z+s[2]]==label)
        if result:
            num=num+1
            arr = np.append(arr, np.array([[x,y,z]]), axis=0) #this is the pixel position under labelmap (img2) coordinate frame
            print('found at ',x,',',y,',',z)
            #plt.matshow(im[x:x+s[0],y:y+s[1],z])
            break

#this arr contains the location in img 2 (label map). Each row is the 3D position of starting point of each cube.
#but we are counting the cells under img (NeuN), so we need to convert the location to img. 

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
# labels=[label-0.5,label+0.5]
# vSur = v.GetImageProcessing().DetectSurfacesWithUpperThreshold(img2,[],aChannel,0,0,True,False,labels[0],True,False,labels[1],None)
# v.GetSurpassScene().AddChild(vSur,1)
# vSur.SetName('Surface '+str(label))

loc2=(arr*[15,15,15]+vExtentMin2-vExtentMin)/np.array([1.8,1.8,4])#the pixel position relative to NeuN frame (starting origin)
#loc3=loc2+[56,56,25]#ending point

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
 
# run fiji to get segmentations
subprocess.call(run_macro_command); #can be annotated: if you open the macro in fiji before running the code, when this code prints "use Imagej now!", hit the "run" button. 

print("use Imagej now!")
sleep(60);
volume_bar = 35 #8um ^ 3 / 4um / 1.8^2 um^2 ~ 35 pixel. Common neurons are larger than this. This is the size threshold to filter out the small components
volume_avgbar=100 #This number is the volume/pixel of a typical neuron and depends on brain regions. I suggest users observe the size of the neurons of tiff files before deciding this number. 
num_neuron = []
for i in range(N):
    image = imread(processed+"morpho_"+str(i)+".tif")
    N_n=image.max() #the number of labels i.e. the number of individual neurons
    for j in range(1,image.max()+1): #j here is the integer label, range in [1,max]
        if image.size - np.count_nonzero(image-j)> 100*volume_bar:
            N_n = N_n - 1 #if the component is too large, then it's unlikely to be neuron. Should be deleted from the count. 
        elif image.size - np.count_nonzero(image-j)<volume_bar:
            N_n = N_n -1 #if too small, count --1
        elif image.size - np.count_nonzero(image-j)>volume_avgbar:
            N_n = N_n + np.floor((image.size - np.count_nonzero(image-j))/volume_avgbar) -1 #if the volume is several times bigger than 1 typical neuron, then the volume/typical volume is regarded as the actual neuron numbers in the blob
    num_neuron.append(N_n)
np.savetxt(root+str(label)+"_counts.csv", num_neuron,fmt='%d', delimiter="\n")
shutil.move(processed,newdir+"tif_processed/")
shutil.move(output,newdir+"tif_out/")
shutil.move(folder,newdir+"Tif/")