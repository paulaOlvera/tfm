import os
import collections

import msgpack
import numpy as np
PLData = collections.namedtuple("PLData", ["data", "timestamps", "topics"])
import keyboard

from pupil_detectors import Detector2D
import uvc
import cv2
detector = Detector2D()
import sys


import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) 

from sklearn.linear_model import LinearRegression

binocular_model = LinearRegression(fit_intercept=True)

display_image =True
def serialized_dict_from_msgpack_bytes(data):
    return msgpack.unpackb(
        data, raw=False, use_list=False, ext_hook=msgpack_unpacking_ext_hook,
    )


def msgpack_unpacking_ext_hook(code, data):
    SERIALIZED_DICT_MSGPACK_EXT_CODE = 13
    if code == SERIALIZED_DICT_MSGPACK_EXT_CODE:
        return serialized_dict_from_msgpack_bytes(data)
    return msgpack.ExtType(code, data)


def load_pldata_file(directory, topic):
    ts_file = os.path.join(directory, topic + "_timestamps.npy")
    msgpack_file = os.path.join(directory, topic + ".pldata")
    try:
        data = []
        topics = []
        data_ts = np.load(ts_file)
        with open(msgpack_file, "rb") as fh:
            for topic, payload in msgpack.Unpacker(fh, raw=False, use_list=False):
                datum = serialized_dict_from_msgpack_bytes(payload)
                data.append(datum)
                topics.append(topic)
    except FileNotFoundError:
        data = []
        data_ts = []
        topics = []

    return PLData(data, data_ts, topics)

def filter_pupil_list_by_confidence(pupil_list,threshold):
    if not pupil_list:
        return []

    len_pre_filter = len(pupil_list)
    pupil_list = [p for p in pupil_list if p["confidence"] >= threshold]
    len_post_filter = len(pupil_list)
    dismissed_percentage = 100 * (1.0 - len_post_filter / len_pre_filter)
    logger.debug(
        f"Dismissing {dismissed_percentage:.2f}% pupil data due to "
        f"confidence < {threshold:.2f}"
    )
    max_expected_percentage = 20.0
    if dismissed_percentage >= max_expected_percentage:
        logger.warning(
            "An unexpectedly large amount of pupil data "
            f"(> {max_expected_percentage:.0f}%) was dismissed due to low confidence. "
            "Please check the pupil detection."
        )
    return pupil_list

def match_pupil_to_ref(pupil_data,ref_data):
    # matches: bino,right,left
    matches = _match_data_batch(pupil_data, ref_data)
    return matches

def _match_data_batch(pupil_list, ref_list):
    assert pupil_list, "No pupil data to match"
    assert ref_list, "No reference data to match"
    pupil0 = [p for p in pupil_list if p["id"] == 0]
    pupil1 = [p for p in pupil_list if p["id"] == 1]
    matched_binocular_data = closest_matches_binocular_batch(ref_list, pupil0, pupil1)

    return matched_binocular_data

def closest_matches_binocular_batch(ref_pts,pupil0,pupil1,max_dispersion=1 / 15.0):
    matched = [[], [], []]
    if not (ref_pts and pupil0 and pupil1):
        return matched

    pupil0_ts = np.array([p["timestamp"] for p in pupil0])
    pupil1_ts = np.array([p["timestamp"] for p in pupil1])

    for r in ref_pts:
        closest_p0_idx = _find_nearest_idx(pupil0_ts, r["timestamp"])
        closest_p0 = pupil0[closest_p0_idx]
        closest_p1_idx = _find_nearest_idx(pupil1_ts, r["timestamp"])
        closest_p1 = pupil1[closest_p1_idx]

        dispersion = max(
            closest_p0["timestamp"], closest_p1["timestamp"], r["timestamp"]
        ) - min(closest_p0["timestamp"], closest_p1["timestamp"], r["timestamp"])
        if dispersion < max_dispersion:
            matched[0].append(r)
            matched[1].append(closest_p0)
            matched[2].append(closest_p1)
        else:
            logger.debug("Binocular match rejected due to time dispersion criterion")
    return matched


def _find_nearest_idx(array, value):
    """Find the index of the element in array which is closest to value"""

    idx = np.searchsorted(array, value, side="left")
    try:
        if abs(value - array[idx - 1]) < abs(value - array[idx]):
            return idx - 1
        else:
            return idx
    except IndexError:
        return idx - 1
 
def extract_features_from_matches_binocular(bino):
        ref,pupil_right,pupil_left=bino
        y_coordinate=[]
        for r in ref:
            norm_pos=r["norm_pos"]
            norm_pos_x=norm_pos[0]
            norm_pos_y=norm_pos[1]
            # flip coordinate
            norm_pos_y=1-norm_pos_y
            norm_pos_rev=[norm_pos_x,norm_pos_y]
            y_coordinate.append(norm_pos_rev)
        Y=np.array(y_coordinate)
        # Y = np.array([r["norm_pos"] for r in ref])
        # reverse coordinate Y 

        X_right = np.array([p["norm_pos"] for p in pupil_right])
        
        X_left = np.array([p["norm_pos"] for p in pupil_left])
        X=np.hstack((X_left,X_right))
        
        return X, Y

def _polynomial_features(norm_xy):
    # slice data to retain ndim
    norm_x = norm_xy[:, :1]
    norm_y = norm_xy[:, 1:]
    norm_x_squared = norm_x**2
    norm_y_squared = norm_y**2

    return np.hstack(
        (
            norm_x,
            norm_y,
            norm_x * norm_y,
            norm_x_squared,
            norm_y_squared,
            norm_x_squared * norm_y_squared,
        )
    )

def fit(X, Y, outlier_removal_iterations=1):

    X_left_polynomial = _polynomial_features(X[:,slice(0,2)])
    X_right_polynomial = _polynomial_features(X[:,slice(2,4)])
    polynomial_features=np.hstack((X_left_polynomial,X_right_polynomial))
    binocular_model.fit(polynomial_features,Y)
    # print("outlier_removal",outlier_removal_iterations)
    # iteratively remove outliers and refit the model on a subset of the data
    errors_px, rmse = _test_pixel_error(polynomial_features, Y)
    if outlier_removal_iterations > 0:
        outlier_threshold_pixel = 70 # set number 
        filter_mask = errors_px < outlier_threshold_pixel
        X_filtered = polynomial_features[filter_mask]
        Y_filtered = Y[filter_mask]
        n_filtered_out = polynomial_features.shape[0] - X_filtered.shape[0]
        # print("size x",X_filtered.shape)
        if n_filtered_out > 0:
            # if we don't filter anything, we can skip refitting the model here
            logger.debug(
                f"Fitting. RMSE = {rmse:>7.2f}px ..."
                f" discarding {n_filtered_out}/{X.shape[0]}"
                f" ({100 * (n_filtered_out) / X.shape[0]:.2f}%)"
                f" data points as outliers."
            )
            # recursively remove outliers
            return fit(
                X_filtered,
                Y_filtered,
                outlier_removal_iterations=outlier_removal_iterations - 1,
            )

    logger.debug(f"Fitting. RMSE = {rmse:>7.2f}px in final iteration.")

def _test_pixel_error(polynomial_features, Y):
    # screen size is the frame size 
    screen_size= (1280,720)
    Y_predict = binocular_model.predict(polynomial_features)
    difference_px = (Y_predict - Y) * screen_size
    errors_px = np.linalg.norm(difference_px, axis=1)
    root_mean_squared_error_px = np.sqrt(np.mean(np.square(errors_px)))
    return errors_px, root_mean_squared_error_px

def initialize_cameras():
    # create 2D detector
    
    dev_list =  uvc.device_list()
    # print("devices: ", dev_list)
    right_eye_camera = uvc.Capture(dev_list[0]['uid'])
    left_eye_camera = uvc.Capture(dev_list[1]['uid'])
    front_camera = uvc.Capture(dev_list[3]['uid'])

    width_right=192
    height_right=192
    fps_right=30

    for mode_right in right_eye_camera.available_modes:
        
        if mode_right.width==width_right and mode_right.height==height_right and mode_right.fps==fps_right:
            right_eye_camera.frame_mode = mode_right
            # the left eye uses the same resolution and fps as the right eye
            left_eye_camera.frame_mode = mode_right

    width_front=1280
    height_front=720
    fps_front=30

    for mode_front in front_camera.available_modes:
        if mode_front.width==width_front and mode_front.height==height_front and mode_front.fps==fps_front:
            front_camera.frame_mode = mode_front

    # print("front",front_camera.frame_mode)
    # print("right",right_eye_camera.frame_mode)
    # print("left",left_eye_camera.frame_mode) 

    return front_camera, right_eye_camera, left_eye_camera 

def normalize(pos,size,flip_y):
    width,height=size
    x = pos[0]
    y = pos[1]
    x /= float(width)
    y /= float(height)

    if flip_y==True:
        return x, 1 - y
    return x,y

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'Reference coordinates in pixels: ({x}, {y})')

if __name__ == "__main__":

    # edit `path` s.t. it points to your recording
    # path = "/home/paula/src/recordings/single_marker"
    # path = "/home/paula/src/recordings/5_markers/3"
    path = r'C:\Users\paula\OneDrive\Escritorio\proyecto\calibration'
    # Read "gaze.pldata" and "gaze_timestamps.npy" data
    get_data = load_pldata_file(path, "notify")
    # print(get_data)
    data_gaze = get_data.data
    data_ts = get_data.timestamps
    topics = get_data.topics

    import pprint

    cv2.namedWindow("Real-Time Image Display", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Real-Time Image left eye", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Real-Time Image right eye", cv2.WINDOW_NORMAL)
    
    calib_data=data_gaze[0].get("calib_data")
    # print(calib_data["ref_list"])
    # extract reference data
    ref_data=calib_data["ref_list"]
    
    # extract and filter pupil data
    pupil_data=calib_data["pupil_list"]
    # print("pupil data",pupil_data)
    pupil_data_post=filter_pupil_list_by_confidence(pupil_data,0.9)
    frame_count=0
    
    if pupil_data and ref_data:
        match_bino=match_pupil_to_ref(pupil_data_post,ref_data)
        
        X,Y = extract_features_from_matches_binocular(match_bino)
        
        
        fit(X,Y)

        # initialize cameras
        front_cam,right_eye_cam,left_eye_cam=initialize_cameras()

        while True:
            image_front_raw=front_cam.get_frame()
            image_front=image_front_raw.gray
            
            image_right_eye_raw=right_eye_cam.get_frame()
            timestamp_right=image_right_eye_raw.timestamp
            image_right_eye=image_right_eye_raw.gray
            result_2d_right_eye=detector.detect(image_right_eye)
            
            image_left_eye_raw=left_eye_cam.get_frame()
            timestamp_left=image_left_eye_raw.timestamp
            image_left_eye=image_left_eye_raw.gray
            result_2d_left_eye=detector.detect(image_left_eye)

            image_size_right=[image_right_eye_raw.width,image_right_eye_raw.height]
            image_size_left = [image_left_eye_raw.width,image_left_eye_raw.height]
            
            norm_pos_right = normalize(result_2d_right_eye["location"],image_size_right,flip_y=True)
            norm_pos_left = normalize(result_2d_left_eye["location"],image_size_left,flip_y=True)
            
            # create eye pupil datum 
            if result_2d_left_eye["confidence"]>0.9 and result_2d_right_eye["confidence"]>0.9:
                X_pred_right=   [norm_pos_right[0],
                                norm_pos_right[1],
                                norm_pos_right[0]*norm_pos_right[1],
                                norm_pos_right[0]**2,
                                norm_pos_right[1]**2,
                                (norm_pos_right[0]**2)*(norm_pos_right[1]**2)]
                
                X_pred_left=  [norm_pos_left[0],
                                norm_pos_left[1],
                                norm_pos_left[0]*norm_pos_left[1],
                                norm_pos_left[0]**2,
                                norm_pos_left[1]**2,
                                (norm_pos_left[0]**2)*(norm_pos_left[1]**2)]

                X_pred=np.hstack((X_pred_left,X_pred_right)).reshape(1,-1)

                gaze_predicted = binocular_model.predict(X_pred).tolist()
                # print()
                # print()
                # print("-------------------------------------------------------------------------------------------------------------")
                # print("norm gaze pos in x:"+str(round(gaze_predicted[0][0],3))+"gaze pos in y:  "+str(round(gaze_predicted[0][1],3)))
                # print()
                
                gaze_pos_px_x=gaze_predicted[0][0]*image_front_raw.width
                gaze_pos_px_y=gaze_predicted[0][1]*image_front_raw.height

                # print("gaze pos in px in x: "+str(round(gaze_pos_px_x))+" gaze pos in px in y : "+str(round(gaze_pos_px_y)))
                sys.stdout.flush()
                if display_image == True:
                    # if 0<gaze_predicted[0][0]<1 and 0<gaze_predicted[0][1]<1:   
                    
                    # Set circle parameters
                    center = (gaze_pos_px_x, gaze_pos_px_y)
                    center = tuple(map(int, center))
                    radius = 20
                    color = (255, 255, 255)  # Green color in BGR format
                    thickness = -1
                    # Draw the circle on the image
                    cv2.circle(image_front, center, radius, color, thickness)
                    
                    # Add text to the frame
                    text = "gaze pos in x: "+str(round(gaze_pos_px_x))+"px    / gaze pos in y : "+str(round(gaze_pos_px_y))+"px"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    font_color = (0, 0, 255)  # Red color in BGR format
                    font_thickness = 2
                    text_position = (50, 50)  # Position to display the text

                    
                    cv2.putText(image_front, text, text_position, font, font_scale, font_color, font_thickness)

                    # show image front 
                    cv2.imshow("Real-Time Image Display", image_front)
                    cv2.setMouseCallback("Real-Time Image Display", click_event)

                    # show image right eye
                    ellipse_right =result_2d_right_eye["ellipse"]
                    cv2.ellipse(
                    image_right_eye,
                    tuple(int(v) for v in ellipse_right["center"]),
                    tuple(int(v / 2) for v in ellipse_right["axes"]),
                    ellipse_right["angle"],
                    0, 360, # start/end angle for drawing
                    (0, 0, 255) # color (BGR): red
                    )
                    image_right_eye=cv2.flip(image_right_eye,0)
                    cv2.imshow("Real-Time Image right eye", image_right_eye)
                    

                    # show image left eye
                    ellipse_left =result_2d_left_eye["ellipse"]
                    cv2.ellipse(
                    image_left_eye,
                    tuple(int(v) for v in ellipse_left["center"]),
                    tuple(int(v / 2) for v in ellipse_left["axes"]),
                    ellipse_left["angle"],
                    0, 360, # start/end angle for drawing
                    (0, 0, 255) # color (BGR): red
                    )
                    image_left_eye=cv2.flip(image_left_eye,1)
                    cv2.imshow("Real-Time Image left eye", image_left_eye)
                    

                    key = cv2.waitKey(1) & 0xFF

                    # Break the loop if 'q' key is pressed
                    if key == ord('q'):
                        break
    del binocular_model
