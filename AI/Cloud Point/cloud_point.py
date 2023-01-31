import argparse
import sys
import os
from PIL import Image

focalLength = 1
scalingFactor = 1

def generate_pointcloud(rgb_file,depth_file):
    rgb = Image.open(rgb_file)
    depth =  Image.open(depth_file)
    width, height =  rgb.size
    centerX = width/2 
    centerY = height/2

    
    if rgb.size != depth.size:
        raise Exception("Color and depth image do not have the same resolution.")

    

    points = []    
    for v in range(rgb.size[1]):
        for u in range(rgb.size[0]):
            color = rgb.getpixel((u,v))
            Z = depth.getpixel((u,v))[0] / scalingFactor * 10
            if Z==0: continue
            X = (u - centerX) * Z / focalLength * 10
            Y = (v - centerY) * Z / focalLength * 10
            points.append("%f %f %f %d %d %d 0\n"%(X,Y,Z,color[0],color[1],color[2]))
    file = open("output.ply","w")
    file.write('''ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar alpha
end_header
%s
'''%(len(points),"".join(points)))
    file.close()

if __name__ == '__main__':
    generate_pointcloud("depth.png","depth.png")
    
