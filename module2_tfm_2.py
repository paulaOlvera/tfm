import zmq
import msgpack
import numpy
import base64


from PIL import Image
import cv2
import sys
import os
import math
# import matplotlib

from vpython import *

from primesense import openni2#, nite2
from primesense import _openni2 as c_api

# from matplotlib import pyplot as plt
from statistics import mean
from time import sleep
import time
from PIL import ImageDraw

directory = r'C:\Users\paula\OneDrive\Escritorio\proyecto\images'
os.chdir(directory)

import msgpack


#  intrinsic camera paramameters Logitech C615: 320x240px
focal_length_front_x=348.54
focal_length_front_y=316.7
principal_point_front_x=187.62
principal_point_front_y=88.52

#  intrinsic camera paramameters Logitech C615: 1280x720px
# focal_length_front_x=1229.8
# focal_length_front_y=1121.18
# principal_point_front_x=750.34
# principal_point_front_y=231.65


# intrinsic camera parameters xtion: 320x240px
focal_length_xtion_x=287.06
focal_length_xtion_y=286.59
principal_point_xtion_x=156.64
principal_point_xtion_y=119.5049

# intrinsic camera parameters frontal: 320x240px
# focal_length_front_x=204.109
# focal_length_front_y=205.26
# principal_point_front_x=156.72
# principal_point_front_y=132.757

# intrinsic camera parameters frontal: 1280x720px
# focal_length_front_x=699.094
# focal_length_front_y=697.304
# principal_point_front_x=632.28
# principal_point_front_y=409.71

# distance in milimeters
distance_x_front_xtion=10
distance_y_front_xtion=150

# distance between cameras wrt xtion
distance_x_xtion_front=distance_x_front_xtion
distance_y_xtion_front=distance_y_front_xtion
distance_z_xtion_front=20

toRad=2*numpy.pi/360



def initialize_socket():

    host = "localhost"
    port = "555"

    # Creates a socket instance
    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)



    # Subscribes to all topics
    subscriber.subscribe("")
    subscriber.setsockopt(zmq.CONFLATE,1)
    # Connects to a bound socket
    subscriber.connect("tcp://{}:{}".format(host, port))

    
    return subscriber


def initialize_xtion():
    openni2.initialize() 
    ## Register the device
    dev = openni2.Device.open_any()
    ## Create the streams stream
    rgb_stream = dev.create_color_stream()
    depth_stream = dev.create_depth_stream()
    
    # print(depth_stream.get_video_mode())
    depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM, resolutionX = 320, resolutionY = 240, fps = 30))
    rgb_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, resolutionX = 320, resolutionY = 240, fps = 30))

    rgb_stream.set_mirroring_enabled(False)
    depth_stream.set_mirroring_enabled(False)

    depth_stream.start()
    rgb_stream.start()
    return rgb_stream,depth_stream

def get_all_depth_xtion(depth_stream):
    dmap=numpy.frombuffer(depth_stream.read_frame().get_buffer_as_uint16(),dtype=numpy.uint16).reshape(240,320)
    return dmap

def get_image_xtion(rgb_stream):
    bgr = numpy.frombuffer(rgb_stream.read_frame().get_buffer_as_uint8(),dtype=numpy.uint8).reshape(240,320,3)
    rgb=Image.fromarray(bgr).convert('L')

    return rgb


def get_depth_xtion(depth_stream,x,y):
    dmap = numpy.frombuffer(depth_stream.read_frame().get_buffer_as_uint16(),dtype=numpy.uint16).reshape(240,320)  # Works & It's FAST
    depth=dmap[x,y]
    return depth

def get_image_front(subscriber):
    done=False
    while done==False:

        message = subscriber.recv()
        message = msgpack.loads(message)
        
        if message['frame']:
            r = base64.decodebytes(message['frame'])
            img=numpy.frombuffer(r,dtype=numpy.uint8)
            img=img.reshape(int(message['height']),int(message['width']),3)
 
            img_front=Image.fromarray(img).convert('L')
            # img_front=img_front.resize((320,240))
            widht,height=img_front.size
            # print("witdht"+str(widht)+"height"+str(height))
            done=True
            # Coordinates wrt the front camera of the Pupil Labs Core in mm.
            x=message['x']
            y=message['y']
            z=message['z']
            
            
            
            return img_front,x,y,z


def apply_surf_ransac(img1,img2,number):
    
    img1=numpy.asarray(img1)
    img2=numpy.asarray(img2)

    points_img1_x=[]
    points_img1_y=[]
    points_img2_x=[]
    points_img2_y=[]
    
    # Initialize SURF detector
    surf=cv2.xfeatures2d.SURF_create()

    # Find the keypoints and descriptors with SURF
    kp1, des1 = surf.detectAndCompute(img1,None)
    kp2, des2 = surf.detectAndCompute(img2,None)



    # For surf or sift pass the following: 
    # FLANN: Fast Library for Approximate Nearest Neighbors. It doesn't mean it finds the best 
    # matches. 
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params= dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    # print("Descriptors image 1:",des1)
    # print("Descriptors image 2: ",des2)

    if len(kp1)>=2 and len(kp2)>=2:
        matches = flann.knnMatch(des1,des2,k=2)
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            # it has to be enough difference between the best and the second-best matches. 
            # Lowe always have the best two matches (k=2 of the descriptors of the images). 
            # m.distance is first because i want the one with the lowest value...
            # https://stackoverflow.com/questions/51197091/how-does-the-lowes-ratio-test-work
            
            if m.distance < 0.6*n.distance:
                good.append(m)
            
        # print("Good points are: ",good)
        # when you apply knn match make sure number of features in both test and query image is 
        # greater or equal to number of nearest neighbors in knn match (in this case, k=2)
        MIN_MATCH_COUNT=15
        if len(good)>MIN_MATCH_COUNT:
            src_pts = numpy.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = numpy.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            
            # mask gives the points that are inliers. 
            matchesMask = mask.ravel().tolist()
        
            h,w = img1.shape

            # points transformed to integer format for the depth.
            src_points=numpy.int32(src_pts)
            src_points=numpy.concatenate(src_points,axis=0)
            
            dst_points=numpy.int32(dst_pts)
            dst_points=numpy.concatenate(dst_points,axis=0)
            
            # mean point for the xtion
            for i in range(0,len(matchesMask)):
                if matchesMask[i]==1:
                    points_img1_x.append(src_points[i,0])
                    points_img1_y.append(src_points[i,1])
                    points_img2_x.append(dst_points[i,0])
                    points_img2_y.append(dst_points[i,1])
        
            if len(points_img1_x)!=0 and len(points_img2_x)!=0:
                x_img1=mean(points_img1_x)
                y_img1=mean(points_img1_y)
                x_img2=mean(points_img2_x)
                y_img2=mean(points_img2_y)
                img1=cv2.circle(img1,(x_img1,y_img1),radius=4,color=(0,0,255),thickness=6)
                img2 = cv2.circle(img2, (x_img2,y_img2), radius=4, color=(0, 0, 255), thickness=6)

                h_x,w_x=img2.shape

                draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                        singlePointColor = None,
                        matchesMask = matchesMask, # draw only inliers
                        flags = 2)
                img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
                img3_final=Image.fromarray(img3)
               
        
    else:
        matchesMask = None
        points_img1_x=[]
        points_img1_y=[]
        points_img2_x=[]
        points_img2_y=[]

    return points_img1_x,points_img1_y,points_img2_x,points_img2_y

def calculate_angle_displacement(point,focal_length,principal_point):
    angle=numpy.degrees(numpy.arctan((point-principal_point)/focal_length))
    return angle

def person_rotation_visualization():
    scene.range=20
    toRad=2*numpy.pi/360
    toDeg=1/toRad

    # initialize the view
    scene.forward=vector(-1,-1,-1)
    scene.width=800
    scene.height=800
    scene.background=color.white

    # create the person and the wheelchair to visualize it in VPython
    ear1=cylinder(pos=vector(-0.8,0.2,0),axis=vector(0.15,0,0),radius=0.3,color=color.white)
    ear2=cylinder(pos=vector(1,0.2,0),axis=vector(0.15,0,0),radius=0.3,color=color.white)
    eye1=cylinder(pos=vector(0.45,0.45,0.8),axis=vector(0,0,0.1),radius=0.25,color=color.white)
    eye2=cylinder(pos=vector(-0.45,0.45,0.8),axis=vector(0,0,0.1),radius=0.25,color=color.white)
    cornea1=cylinder(pos=vector(0.42,0.45,0.9),axis=vector(0,0,0.05),radius=0.15,color=color.black)
    cornea2=cylinder(pos=vector(-0.42,0.45,0.9),axis=vector(0,0,0.05),radius=0.15,color=color.black)
    v0=vertex( pos=vec(-0.4,-0.1,1.05),color=color.black )
    v1=vertex( pos=vec(-0.1,-0.3,1.05),color=color.black)
    v2=vertex( pos=vec(0.4,-0.1,1.05),color=color.black )
    mouth = triangle(vs=[v0,v1,v2])
    wheel1=cylinder(pos=vector(1.3,-4,0),axis=vector(0.2,0,0),radius=1.5)
    wheel2=cylinder(pos=vector(-1.3,-4,0),axis=vector(-0.2,0,0),radius=1.5)
    rect1=box(pos=vector(0,-3,0),size=vector(2.8,0.2,2),color=color.black)
    rect2=box(pos=vector(0,-1,-1),size=vector(2.8,3,0.2),color=color.black)
    body=cone(pos=vector(0,-3,0),axis=vector(0,5.5,0),radius=0.9)
    nouse=cone(pos=vector(0,0.1,1),axis=vector(0,0,0.2),color=color.white,radius=0.15)
    person_face=sphere(radius=1,opacity=1)


    # xtion reference axis
    sidextionRef=arrow(length=3,shaftwidth=0.2,axis=vector(-1,0,0),pos=vector(-5,3,0),color=color.red)
    upxtionRef=arrow(length=3,shaftwidth=0.2,axis=vector(0,-1,0),pos=vector(-5,3,0),color=color.green)
    frontxtionRef=arrow(length=3,shaftwidth=0.2,axis=vector(0,0,1),pos=vector(-5,3,0),color=color.blue)
    # create the camera xtion in the scene 
    camera_xtion1=box(pos=vector(-5.5,2.7,-1),size=vector(1.5,1,0.3))
    camera_xtion2=box(pos=vector(-5.5,2.8,-0.5),size=vector(0.75,0.5,0.3))

    # create the pupil labs camera axis
    xArrow=arrow(length=2,shaftwidth=0.1,axis=vector(-1,0,0),pos=vector(0,2.75,1),color=color.red)
    zArrow=arrow(length=2,shaftwidth=0.1,axis=vector(0,-1,0),pos=vector(0,2.75,1),color=color.green)
    yArrow=arrow(length=2,shaftwidth=0.1,axis=vector(0,0,1),pos=vector(0,2.75,1),color=color.blue)

    frontArrow=arrow(length=1.5,shaftwidth=0.1,axis=vector(0,0,1),pos=vector(0,2.75,1),color=color.orange)
    upArrow=arrow(length=1.5,shaftwidth=0.1,axis=vector(0,-1,0),pos=vector(0,2.75,1),color=color.magenta)
    sideArrow=arrow(length=1.5,shaftwidth=0.1,axis=vector(-1,0,0),pos=vector(0,2.75,1),color=color.purple)
    # create the gaze arrow
    gazeArrow=arrow(length=2,shaftwidth=0.1,axis=vector(2,0,2),pos=vector(0,2.75,1),color=color.black)

    person_face=compound([person_face,ear1,ear2,eye1,eye2,cornea1,cornea2,mouth,nouse])
    person_face.pos=vector(0,2,0)
    return frontArrow,upArrow,sideArrow,person_face,frontxtionRef,upxtionRef,sidextionRef,xArrow,zArrow,yArrow,gazeArrow



# main

# initialize socket for data flow between the file of Pupil Labs and this file
subscriber=initialize_socket()
# initialize the asus xtion camera
rgb_stream,depth_stream=initialize_xtion()
# create the scene in VPython
front_arrow,up_arrow,side_arrow,person_face,front_xtion_ref,up_xtion_ref,side_xtion_ref,x_Arrow,z_Arrow,y_Arrow,gaze_Arrow=person_rotation_visualization()

# initialize variables
angle_yaw_xtion=0
angle_pitch_xtion=0
angle_roll_xtion=0
increment_yaw=0
increment_pitch=0
increment_roll=0
time_vector=[]
count=0
number=0
beta=0
alpha=0.9
alpha_front=0.1
counter=0

while True:
    dmap=[]
    start=time.time()
    # obtain Xtion depth and colour image
    xtion_img1=get_image_xtion(rgb_stream)
    dmap=get_all_depth_xtion(depth_stream)
    
    sys.stdout.flush()

    # #######################################################################################################################
    # # Matching XTION & FRONT
    # #######################################################################################################################
    # obtain front image
    front_img1,x,y,z=get_image_front(subscriber)

    # apply SURF and matching algorithm
    points_xtion_img1_x,points_xtion_img1_y,points_front_img1_x,points_front_img1_y=apply_surf_ransac(xtion_img1,front_img1,number)
    depth_xtion=-100

    if len(points_xtion_img1_x)!=0 and len(points_front_img1_x)!=0:
        # obtain the mean descriptor points
        x_xtion_img1=mean(points_xtion_img1_x)
        y_xtion_img1=mean(points_xtion_img1_y)
        x_front_img1=mean(points_front_img1_x)
        y_front_img1=mean(points_front_img1_y)
        depth_xtion=get_depth_xtion(depth_stream,y_xtion_img1,x_xtion_img1)

        counter=counter+1


    if depth_xtion>0:
        number=number+1
            
        print()
        print("--------------------------------------------------------------------------------")
        print()
        print()
        print(number)
        print("using images: "+str(counter-1)+"and"+str(counter))

        #######################################################################################
        # Angle YAW: Matching XTION & FRONT
        #######################################################################################
        
        angle_a=calculate_angle_displacement(x_xtion_img1,focal_length_xtion_x,principal_point_xtion_x)
        angle_b=calculate_angle_displacement(x_front_img1,focal_length_front_x,principal_point_front_x)
        angle_c=90-angle_a
        distance_E=cos(numpy.deg2rad(angle_c))*depth_xtion
        angle_d=numpy.degrees(atan2((depth_xtion*sin(numpy.deg2rad(angle_c))),(distance_x_front_xtion+distance_E)))
        angle_d=(angle_d+360)%360
        angle_yaw_xtion_front=90-angle_d-angle_b
        
        print()
        print("Head displacement in YAW of the XTION and FRONT: ",str(round(angle_yaw_xtion_front,2)))
        
        ######################################################################################
        # Angle PITCH: Matching XTION & FRONT
        #######################################################################################

        angle_e=calculate_angle_displacement(y_xtion_img1,focal_length_xtion_y,principal_point_xtion_y)
        distance_P=sin(numpy.deg2rad(angle_e))*depth_xtion
        distance_M=cos(numpy.deg2rad(angle_e))*depth_xtion
        angle_g=numpy.degrees(atan2(distance_M,(distance_y_front_xtion+distance_P)))
        angle_g=(angle_g+360)%360
        angle_f=calculate_angle_displacement(y_front_img1,focal_length_front_y,principal_point_front_y)
        angle_pitch=90-angle_f-angle_g
        print("Head displacement in PITCH of the XTION and FRONT: ",round(-angle_pitch,2))
        angle_pitch_xtion_front=-angle_pitch

        ######################################################################################
        # Angle ROLL: Matching XTION & FRONT
        #######################################################################################

        angle_roll_vector=[]
        for i in range(len(points_xtion_img1_x)):
                for j in range(len(points_xtion_img1_y)):
                    if i!=j:
                        
                        angle_roll_front=numpy.degrees(numpy.arctan2((points_front_img1_y[j]-points_front_img1_y[i]),(points_front_img1_x[j]-points_front_img1_x[i])))
                        angle_roll_front=(angle_roll_front+360)%360
                        
                        angle_roll_xtion=numpy.degrees(numpy.arctan2((points_xtion_img1_y[j]-points_xtion_img1_y[i]),(points_xtion_img1_x[j]-points_xtion_img1_x[i])))
                        angle_roll_xtion=(angle_roll_xtion+360)%360
                        
                        angle_roll=angle_roll_front-angle_roll_xtion
                        if angle_roll>180:
                            angle_roll=-(360-abs(angle_roll))
                        if angle_roll<-180:
                            angle_roll=+(360-abs(angle_roll))
                        angle_roll_vector.append(angle_roll)

        print("Head displacement in ROLL of the XTION and FRONT : ",round(mean(angle_roll_vector),2))
        angle_roll=mean(angle_roll_vector)
        sys.stdout.flush()
        
        if x:
            print("Gaze position wrt front is in x: "+str(round(x,2))+" in y: "+str(round(y,2))+" in z: "+str(round(z,2)))
           
            yaw=angle_yaw_xtion_front*toRad
            pitch=-angle_pitch_xtion_front*toRad
            roll=angle_roll*toRad

            R_z_roll=[[cos(roll), -sin(roll), 0],[sin(roll), cos(roll), 0],[0,0,1]]
            R_y_yaw=[[cos(yaw), 0, sin(yaw)],[0,1,0],[-sin(yaw),0,cos(yaw)]]
            R_x_pitch=[[1, 0, 0],[0, cos(pitch), -sin(pitch)],[0, sin(pitch),cos(pitch)]]
                
            coord_wrt_R_xtion=numpy.matmul(numpy.matmul(numpy.matmul(R_z_roll,R_y_yaw),(R_x_pitch)),[x,y,z])
            coord_wrt_T_xtion=[distance_x_xtion_front,distance_y_xtion_front,distance_z_xtion_front]
            
            coord_wrt_xtion_final=coord_wrt_T_xtion+coord_wrt_R_xtion


            print("Gaze position wrt xtion rotation is in x: "+str(round(coord_wrt_R_xtion[0],2))+" in y: "+str(round(coord_wrt_R_xtion[1],2))+" in z: "+str(round(coord_wrt_R_xtion[2],2)))
            print("Distance from xtion to front camera is in x: "+str(round(coord_wrt_T_xtion[0],2))+" in y: "+str(round(coord_wrt_T_xtion[1],2))+" in z: "+str(round(coord_wrt_T_xtion[2],2)))
            print("Gaze position wrt xtion is in x: "+str(round(coord_wrt_xtion_final[0],2))+" in y: "+str(round(coord_wrt_xtion_final[1],2))+" in z: "+str(round(coord_wrt_xtion_final[2],2)))


        # relate the head rotation angles to rotation in the tridimensional space. 
        yaw=-angle_yaw_xtion_front*toRad
        pitch=angle_pitch_xtion_front*toRad
        roll=-angle_roll*toRad
        k=vector(sin(yaw)*cos(pitch),sin(pitch),cos(yaw)*cos(pitch))
        y=vector(0,1,0)
        s=cross(k,y)
        v=cross(s,k)
        vrot=v*cos(roll)+cross(k,v)*sin(roll)
        
        # show the head rotation in the tridimensional space in the Vpython scene
        front_arrow.axis=k
        front_arrow.length=1.5
        person_face.axis=cross(vrot,k)
        person_face.up=vrot
        side_arrow.axis=-cross(vrot,k)
        side_arrow.length=1.5
        up_arrow.axis=-vrot
        up_arrow.length=1.5

        if x:
            # show gaze direction
            angle_x_y=numpy.arctan2(coord_wrt_R_xtion[1],coord_wrt_R_xtion[0])
            x_coordinate=cos(angle_x_y)*1
            y_coordinate=sin(angle_x_y)*1
            angle_y_z=numpy.arctan2(-coord_wrt_R_xtion[1],coord_wrt_R_xtion[2])
            z_coordinate=cos(angle_y_z)*1
            gaze_Arrow.axis=vector(-x_coordinate,-y_coordinate,2*z_coordinate)
            gaze_Arrow.length=4

    end=time.time()
    print("Time to process is: ",end-start)
    
   
