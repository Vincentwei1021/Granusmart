import numpy as np
import skimage.morphology
import matplotlib.pyplot as plt
# from rotating_calipers import min_max_feret

# This file contains code taken from 
# http://code.activestate.com/recipes/117225-convex-hull-and-diameter-of-2d-point-sets/
# convex hull (Graham scan by x-coordinate) and diameter of a set of points
# David Eppstein, UC Irvine, 7 Mar 2002

# According to that website the code is under the PSF licencse
# https://en.wikipedia.org/wiki/Python_Software_Foundation_License


# modifications by Volker Hilsenstein

# from __future__ import generators
from math import sqrt


def orientation(p,q,r):
    '''Return positive if p-q-r are clockwise, neg if ccw, zero if colinear.'''
    return (q[1]-p[1])*(r[0]-p[0]) - (q[0]-p[0])*(r[1]-p[1])

def hulls(Points):
    '''Graham scan to find upper and lower convex hulls of a set of 2d points.'''
    U = []
    L = []
    Points.sort()
    for p in Points:
        while len(U) > 1 and orientation(U[-2],U[-1],p) <= 0: U.pop()
        while len(L) > 1 and orientation(L[-2],L[-1],p) >= 0: L.pop()
        U.append(p)
        L.append(p)
    return U,L

def rotatingCalipers(Points):
    '''Given a list of 2d points, finds all ways of sandwiching the points
between two parallel lines that touch one point each, and yields the sequence
of pairs of points touched by each pair of lines.'''
    U,L = hulls(Points)
    i = 0
    j = len(L) - 1
    while i < len(U) - 1 or j > 0:
        yield U[i],L[j]
        
        # if all the way through one side of hull, advance the other side
        if i == len(U) - 1: j -= 1
        elif j == 0: i += 1
        
        # still points left on both lists, compare slopes of next hull edges
        # being careful to avoid divide-by-zero in slope calculation
        elif (U[i+1][1]-U[i][1])*(L[j][0]-L[j-1][0]) > \
                (L[j][1]-L[j-1][1])*(U[i+1][0]-U[i][0]):
            i += 1
        else: j -= 1


def min_max_feret(Points):
    '''Given a list of 2d points, returns the minimum and maximum feret diameters.'''
    # print('points:', Points)
    squared_distance_per_pair = [((p[0]-q[0])**2 + (p[1]-q[1])**2, (p,q))
                     for p,q in rotatingCalipers(Points)]
    min_feret_sq, min_feret_pair = min(squared_distance_per_pair)
    max_feret_sq, max_feret_pair = max(squared_distance_per_pair)
    return sqrt(min_feret_sq), sqrt(max_feret_sq)


def diameter(Points):
    '''Given a list of 2d points, returns the pair that's farthest apart.'''
    diam,pair = max([((p[0]-q[0])**2 + (p[1]-q[1])**2, (p,q))
                     for p,q in rotatingCalipers(Points)])
    return diam, pair

def min_feret(Points):
    '''Given a list of 2d points, returns the pair that's farthest apart.'''
    min_feret_sq,pair = min([((p[0]-q[0])**2 + (p[1]-q[1])**2, (p,q))
                     for p,q in rotatingCalipers(Points)])
    return min_feret_sq, pair

def get_min_max_feret_from_labelim(label_im, labels=None):
    """ given a label image, calculate the oriented 
    bounding box of each connected component with 
    label in labels. If labels is None, all labels > 0
    will be analyzed.
    Parameters:
        label_im: numpy array with labelled connected components (integer)
    Output:
        obbs: dictionary of oriented bounding boxes. The dictionary 
        keys correspond to the respective labels
    """
    if labels is None:
        labels = set(np.unique(label_im)) - {0}
    results = {}
    for label in labels:
        results[label] = get_min_max_feret_from_mask(label_im == label)
    return results

    
def get_min_max_feret_from_mask(mask_im):
    """ given a binary mask, calculate the minimum and maximum
    feret diameter of the foreground object. This is done
    by calculating the outline of the object, transform
    the pixel coordinates of the outline into a list of
    points and then calling 
    Parameters:
        mask_im: binary numpy array
    """
    eroded = skimage.morphology.erosion(mask_im)
    outline = mask_im ^ eroded
    boundary_points = np.argwhere(outline > 0)
        
    center_point = np.average(boundary_points, axis = 0)
    boundary_points_0 = boundary_points - center_point
    
    theta = np.array(list(range(0, 100))) * np.pi/100
    
    # n=size(theta,1);
    # axis_l = zeros(n,1);

    dir_unit = np.zeros([2, theta.shape[0]])
    dir_unit[0, :] = np.cos(theta)
    dir_unit[1, :] = np.sin(theta)
    
    radius_center = np.matmul(boundary_points_0, dir_unit)
    
    # axis_l(i)=max(dis)-min(dis)
    radius_feret = np.max(radius_center, axis = 0) - np.min(radius_center, axis = 0) 
    # print('test:', radius_feret.max(), radius_feret.min())
    # convert numpy array to a list of (x,y) tuple points
    # boundary_point_list = list(map(list, list(boundary_points)))
    # aaa = min_max_feret(boundary_point_list)

    return radius_feret.min(), radius_feret.max()

def img_crop(image):
    
    # print(type(image[0,0,0]))
    image1 = image.astype(np.int16)
    Ir = image1[:, :, 0]
    Ig = image1[:, :, 1]
    Ib = image1[:, :, 2]
    
    Ip = ((Ir - Ig)/255 + (Ir - Ib)/255)/2
    Ip[Ip < 0] = 0
    
    print('Ip max:', Ip.max(), Ip.min())
    binarized = 1 * (Ip > Ip.max() * 0.8)
    
    # with np.printoptions(threshold=np.inf):
        # print('binarized: ', Ip[1000])
    # _, ax1 = plt.subplots(1)
    # ax1.imshow(binarized)
    # plt.show()
    
    indice_1 = np.where(binarized == 1.0)
    print('indice range: ', indice_1[0].max(), indice_1[0].min(), indice_1[1].max(), indice_1[1].min())
    
    return image[indice_1[0].min(): indice_1[0].max(), indice_1[1].min(): indice_1[1].max(), :]
    
