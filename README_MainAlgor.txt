#Read me document for neuron counting method

The code serves as a supplemental to our paper submitted to Frontiers Neurosceince. Please cite our bioxiv version for now if you use our method. 

This code consists of several modules:

The first part defines parameters such as the path for the labelmap and classifier, the label for the desired region, the number of subvolumes for sampling, etc.

The second part generates macro code for FIJI based on the selected parameters and runs it when the TIFF files are ready.

The third part generates the locations of the subvolumes for sampling based on the labelmap header file (nhdr). These coordinates are relative to the labelmap frame.

The fourth part connects to the Imaris server and obtains frames of both LSM and labelmap. The coordinates are converted to the LSM frame, and TIFF files are saved individually for each subvolume.

The fifth part applies the classifier to each region by running the macro code. The components are then watershed and processed, and the resulting numbers are saved as a CSV file in the current folder.

Please fork and let CIVM know if you have any requests to change a particular module.

To initialize running the code, input 

>>>import subprocess
>>>subprocess.call(['C:\\Program Files\\Bitplane\\Imaris 9.9.0\\Imaris.exe', 'id101']) 

in Python. This will call the Imaris with ID 101. This ID has to agree with the ID when getting Imaris application (Line 127: v = vImarisLib.GetApplication(101))

Then, import NeuN file as the first image, Labelmap as the second image. 

Finally, run this code in command line. 