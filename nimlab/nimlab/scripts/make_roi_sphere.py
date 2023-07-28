#!/usr/bin/env python

#Creates a spherical ROI from a single coordinate
#
#Jonas Kaplan
#jtkaplan@usc.edu
#July 2014
#

from __future__ import division
from __future__ import print_function
import argparse
import os
from subprocess import call

#parse command line arguments
parser = argparse.ArgumentParser(description="create spherical ROIs from coordinate")
parser.add_argument("x",help="x coordinate")
parser.add_argument("y",help="y coordinate")
parser.add_argument("z",help="z coordinate")
parser.add_argument("radius",help="radius in mm")
parser.add_argument("filename",help="output file name")
parser.add_argument("--mm",help="coordinates are mm coordinates in MNI space instead of voxel coordinates",action="store_true")
args = parser.parse_args()

def mmToVox(mmcoords):
	#function to convert mm coordinates in the standard 2mm MNI atlas into voxel coordinates
	voxcoords = ['','','']
	voxcoords[0] = str(int(round(int(mmcoords[0])/2))*-1+45)
	voxcoords[1] = str(int(round(int(mmcoords[1])/2))+63)
	voxcoords[2] = str(int(round(int(mmcoords[2])/2))+36)
	return voxcoords


coords =[args.x,args.y,args.z]

if args.mm:
	#convert from mm to voxel coords
	voxcoords = mmToVox(coords)
else:
	voxcoords = coords

outfile = args.filename
		
command = "fslmaths $FSLDIR/data/standard/MNI152_T1_2mm_brain_mask -roi %s 1 %s 1 %s 1 0 1 tmp" % (voxcoords[0],voxcoords[1],voxcoords[2])
print(command)
call(command,shell=True)

command = "fslmaths tmp -kernel sphere %s -fmean tmp" % (args.radius)
print(command)
call(command,shell=True)

command = "fslmaths tmp -thr .00001 -bin %s" % outfile
print(command)
call(command,shell=True)

command = "rm tmp.nii.gz"
call(command,shell=True)





