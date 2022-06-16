## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################
import featureExtraction as fe
# First import the realsense library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
import imageProcessing as ip
import segmentation as seg
import KNearestNeighbour as knn
import segmentation3 as TSeg
from PIL import Image



bottleTrainingData = np.loadtxt('FeatureData/DAT/trainingBottlesNoCilindricality.dat', dtype=float, delimiter=',')
randomObjTrainingData = np.loadtxt('FeatureData/DAT/trainingRandomObjNoCilindricality.dat', dtype=float, delimiter=',')

allFileData = np.concatenate((bottleTrainingData, randomObjTrainingData))
combinedLists = [bottleTrainingData, randomObjTrainingData]

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

# We will be removing the background of objects more than
# clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Streaming loop
try:
    i = 0
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # # Save color and depth data in Numpy array format

        # # Save color data in PNG format
        # im_col = Image.fromarray(color_image)
        # im_col.save("color{}.png".format(i))

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 0
        depth_image_3d = np.dstack((depth_image, depth_image, depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        # # Save color data without background in PNG and Numpy format
        # im_no_bg = Image.fromarray(bg_removed)
        # im_no_bg.save("../SavedImages/Training/Bottles2/Background removed/bgRemoved{}.png".format(i))
        # np.save("../SavedImages/2ImgTest/Color/bgRemoved{}.npy".format(i), bg_removed)
        # # Save the frame number for naming saved images

        img = []
        # img.append(bg_removed)
        img.append(color_image)
        img2 = [img]
        # print("wo")
        # seg.segmentation(img2)
        # cv2.imshow("output",img2[0][0])



        # Render images:

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_OCEAN)
        depth = []
        depth.append(depth_image)

        images = np.hstack((bg_removed, depth_colormap))

        # cv2.imwrite('depthAligned.png', depth_colormap)
        # im_depth = Image.fromarray(depth_colormap)
        # im_depth.save("depthAligned{}.png".format(i))

        # testImg = TSeg.segmentation([img],depth_colormap)

        # If there are no contours in the image we get this UnboundLocalError
        # There is probably a way to fix it but don't want to spend time on it
        try:
            cor = ip.imageProcessing(img, depth, depth_colormap)
        except UnboundLocalError:
            print("Contour error")



        key = cv2.waitKey(0)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

        # if the key press is on of the following words in the if statements
        # If you try to print a coordinate for a contour that isn't there this will save the code
        elif key == ord('b'):
            try:
                # Right now this is print, this is different on the ubuntu code
                # If the ubuntu code this instead publishes the coordinates onto a topic
                print("cor for contour 1:", cor[0])
            except IndexError:
                print("No coordinates")
        elif key == ord('n'):
            try:
                print("cor for contour 2:", cor[1])
            except IndexError:
                print("No coordinates")
        elif key == ord('m'):
            try:
                print("cor for contour 3:", cor[2])
            except IndexError:
                print("No coordinates")

finally:
    pipeline.stop()