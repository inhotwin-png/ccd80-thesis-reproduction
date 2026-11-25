"""Original CCD80 grid geometry in metres and degrees."""
import numpy as np
from .map_inputs127 import RADIUS
STEP_DEG=62.0/RADIUS*180/np.pi
WIDTH=int(np.ceil(360/STEP_DEG))

def coordinates(rc):
    rc=np.asarray(rc);return -180+(rc[...,1]+.5)*STEP_DEG,90-(rc[...,0]+.5)*STEP_DEG

def distance(rc,centre):
    lon,lat=coordinates(rc);x,y=coordinates(np.asarray(centre));a=np.radians(lat);b=np.radians(y)
    h=np.sin((a-b)/2)**2+np.cos(a)*np.cos(b)*np.sin(np.radians(lon-x)/2)**2
    return 2*RADIUS*np.arcsin(np.sqrt(np.clip(h,0,1)))

def weight(rc):return np.cos(np.radians(coordinates(rc)[1]))
