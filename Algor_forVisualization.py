#For generate labkit subregions 
#first open a python.exe, call imaris with a self-defined ID
#import subprocess 
#subprocess.call(['C:\\Program Files\\Bitplane\\Imaris 9.9.0\\Imaris.exe', 'id101'])                                                                            
import ImarisLib
import numpy as np
import math as m
import random
import nrrd

filename=r"R:\22.gaj.49\200826-1_1\Aligned-Data\labels\RCCF\N58646NLSAM_RCCF_labels.nhdr".replace("\\","/") #import the labelmap to locate a brain region, generate N random subvolumes with size s (/pixels)
im, header = nrrd.read(filename)

label=1024 #label,#LGd 82 Perirhinal cortex 23, retroplenial 24, auditory 20, subiculum 28
N=30#subvolume number
s=[10,10,10] #Make sure this region is larger than subvolume. The current cube size is (15um * 10)^3

im=im[::-1,::-1,:].astype(int)#remember to check the header.
#header fix: flip the image so that the coordinate system goes to positive entity matrix and Right-Anterior-Superior
indices=np.argwhere(im==label)
print(indices[:,2])
x_min=min(indices[:,0])
x_max=max(indices[:,0])
y_min=min(indices[:,1])
y_max=max(indices[:,1])
z_min=min(indices[:,2])
z_max=max(indices[:,2])
arr = np.empty((0,3), int)

num=0;
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

#this arr contains the location in img2 (labelmap). Each row is the 3D position of starting point of each cube.
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
print(type(img2)) 
vExtentMin=[img.GetExtendMinX(),img.GetExtendMinY(),img.GetExtendMinZ()]
vExtentMin2 =[img2.GetExtendMinX(),img2.GetExtendMinY(),img2.GetExtendMinZ()]
print('vExtentMin:',vExtentMin)
aChannel=0
labels=[label-0.5,label+0.5]
vSur = v.GetImageProcessing().DetectSurfacesWithUpperThreshold(img2,[],aChannel,0,0,True,False,labels[0],True,False,labels[1],None)
v.GetSurpassScene().AddChild(vSur,1)
vSur.SetName('Surface '+str(label))

loc2=(arr*[15,15,15]+vExtentMin2-vExtentMin)/np.array([1.8,1.8,4])
loc3=loc2+[56,56,25] ##the size of subvolume is (56,56,25) pixels with resolution (1.8,1.8,4)um. 

for i in range(N): 
    subregion_min=loc2[i]
    subregion_max=loc3[i]
    Range=list(subregion_min)+[0]+list(subregion_max)+[0]
    vsur1= v.GetImageProcessing().DetectSurfaces(img,[Range],aChannel,0,0,True,False,None)
    vsur1.SetName('Region'+str(label)+'_surface_'+str(i))
    v.GetSurpassScene().AddChild(vsur1,102+i)
    
    
# center=(SurfaceMin+SurfaceMax)/2+0.2*(SurfaceMax-SurfaceMin)*np.array([random.random(),random.random(),random.random()])
# subregion_min=(center-subregion_half - np.array(vExtentMin))/np.array([1.8,1.8,4])
# subregion_max=(center+subregion_half - np.array(vExtentMin))/np.array([1.8,1.8,4])
# Range=list(subregion_min)+[0]+list(subregion_max)+[0]
# vsur1= v.GetImageProcessing().DetectSurfaces(img,[Range],aChannel,0,0,True,False,None)
# vsur1.SetName(str(region_num)+'_surface_'+str(i))
# v.GetSurpassScene().AddChild(vsur1,102+i)

# test
# aSpotFiltersString='"Intensity Center Ch=1 Img=2" between '+str(labels[0])+' and '+str(labels[1])
# vsur2= v.GetImageProcessing().DetectSurfaces(img2,[],aChannel,0,0,True,False,aSpotFiltersString)#no, stuck, not working
# v.GetSurpassScene().AddChild(vsur2,101)