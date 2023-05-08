This code serves as a supplemental tool to our paper submitted to Frontiers in Neuroscience. If you use our method, please cite our bioRxiv version for now.

The code is a manual version designed for visualization and contains the following modules:

The first part defines parameters, such as the path for the labelmap and classifier, the label for the desired region, and the number of subvolumes for sampling.
The second part generates the locations of the subvolumes for sampling based on the labelmap header file (nhdr). These coordinates are relative to the labelmap frame.
The third part connects to the Imaris server and obtains frames of both LSM and labelmap. The coordinates are converted to the LSM frame, and the subvolumes will be generated in the Imaris GUI.
The next steps involve manually configuring each subvolume by applying the neuron classifier, splitting touching objects and applying volume filters.

To start running the code, first open Python and input the following:

python

import subprocess
subprocess.call(['C:\\Program Files\\Bitplane\\Imaris 9.9.0\\Imaris.exe', 'id101'])

This will call Imaris with ID 101. Make sure that the ID in this command matches the ID used to get the Imaris application in the code (Line: v = vImarisLib.GetApplication(101)).

Next, import the NeuN file as the first image and the Labelmap as the second image in Imaris.

Finally, run the code in the command line.