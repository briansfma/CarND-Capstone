import os

os_command = "export PYTHONPATH=$PYTHONPATH:" + os.getcwd() + "/models/research:" \
             + os.getcwd() + "/models/research/slim"
os.system(os_command)

import rospy
import cv2
import numpy as np
import tensorflow as tf
from styx_msgs.msg import TrafficLight
from object_detection.utils import ops as utils_ops


class TLClassifier(object):
    def __init__(self):
        # Load classifier
        MODEL_NAME = 'faster_rcnn_inception_v2_coco_2017_11_08'

        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        PATH_TO_FROZEN_GRAPH = 'models/research/' + MODEL_NAME + '/frozen_inference_graph.pb'

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()

            with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    def run_inference(self, image, graph):
        with graph.as_default():
            with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict

    def avg_near_point(self, img, x, y, radius):
        avg = 0
        count = 0
        for i in range(-radius, radius+1):
            if 0 <= y + i < img.shape[0]:
                for j in range(-radius, radius+1):
                    if 0 <= x + j < img.shape[1]:
                        avg += img[y + i, x + j]
                        count += 1
        if count > 0:
            return avg // count
        else:
            return 0

    def adjust_gamma(self, image, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                         for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)

    def classify_real_image(self, image):
        # Default state: Unknown
        color_logic = 4
        
        # Detect traffic light outline
        output_dict = self.run_inference(image, self.detection_graph)

        # Preprocess image slightly
        image_ga = self.adjust_gamma(image, 0.4)  # Gamma correction
        image_hls = cv2.cvtColor(image_ga, cv2.COLOR_RGB2HLS)
        l_chan = image_hls[:, :, 1]

        # Ignore any detections that have a very low confidence
        detect_thres = 0.02
        idx_filtered = [i for i in range(len(output_dict['detection_classes']))
                        if (output_dict['detection_classes'][i] == 10 and
                            output_dict['detection_scores'][i] > detect_thres)]

        # Rough logic - the network tends to detect only traffic-light sized objects and smaller,
        # but no larger. So we will "trust" the tallest box found with sufficient confidence.
        ht = 0
        belief = [0, 0, 0]
        for i in idx_filtered[0:3]:
            score = output_dict['detection_scores'][i]
            box = output_dict['detection_boxes'][i]

            # Get bounding box coordinates (fraction from [0,1] originally)
            # Scale coordinates for input image size
            xmin = int(box[1] * image_ga.shape[1])
            xmax = int(box[3] * image_ga.shape[1])
            ymin = int(box[0] * image_ga.shape[0])
            ymax = int(box[2] * image_ga.shape[0])

            # Keep working on only the tallest (most likely) bounding box
            new_ht = ymax - ymin
            if new_ht > ht:
                ht = new_ht

                # Calculate approximate centers of traffic light circles
                # Rarely accurate, but the points tend to fall on or near enough to the
                # actual center, so we can evaluate color and brightness
                light_x = int(xmin + (xmax - xmin) / 2)
                red_y = int(ymin + (ymax - ymin) / 5)
                yellow_y = int(ymin + (ymax - ymin) / 2)
                green_y = int(ymax - (ymax - ymin) / 5)

                # Gauge lightness of pixels around each spot
                red_avg = self.avg_near_point(l_chan, light_x, red_y, 5)
                yellow_avg = self.avg_near_point(l_chan, light_x, yellow_y, 5)
                green_avg = self.avg_near_point(l_chan, light_x, green_y, 5)

                belief = [red_avg, yellow_avg, green_avg]

                # # Visualization
                # print("Index: ", i)
                # print("Score: ", score)
                # print("Box: ", box)
                # cv2.circle(image_ga, (light_x, red_y), 6, (255, 0, 0), 1)
                # cv2.circle(image_ga, (light_x, yellow_y), 6, (255, 0, 0), 1)
                # cv2.circle(image_ga, (light_x, green_y), 6, (255, 0, 0), 1)
                # cv2.rectangle(image_ga, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)
                # print("xmin:", xmin, ", xmax:", xmax, ", ymin:", ymin, ", ymax:", ymax)
                # print("Avgs: R/Y/G ", red_avg, yellow_avg, green_avg)

        MARGIN = 12
        red_avg = belief[0]
        yellow_avg = belief[1]
        green_avg = belief[2]
        if ht > 0:
            if green_avg > yellow_avg + MARGIN and green_avg > red_avg + MARGIN:
                color_logic = 2     # Green
            elif yellow_avg > green_avg + MARGIN and yellow_avg > red_avg + MARGIN:
                color_logic = 1     # Yellow
            else:
                color_logic = 0     # Red

        return color_logic

    def classify_sim_image(self, image):
        # Default state: Unknown
        color_logic = 4

        img = cv2.GaussianBlur(image, (3,3), 0)
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        l_chan = hls[:,:,1]
        s_chan = hls[:,:,2]
        edges = cv2.Canny(img,50,120)

        im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if hierarchy is not None:
            hier_hasparent = hierarchy[0,:,3]

            # TODO: Find contours round enough to be lights
            contours_keep = []
            for i in range(len(contours)):
                perimeter = cv2.arcLength(contours[i], True)
                area = cv2.contourArea(contours[i])
                if perimeter > 10 and hier_hasparent[i] == -1:
                    roundness = 4 * np.pi * area / perimeter / perimeter
                    if 0.7 < roundness < 1.25:
                        contours_keep.append(contours[i])

            # TODO: Find center of contours
            contours_final = []
            centroids_keep = []
            extent_pts = []
            for con in contours_keep:
                M = cv2.moments(con)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                con_Xs = con[:,0,0]
                con_Ys = con[:,0,1]
                # print(con_Xs)
                # print(con_Ys)
                # print(cX, cY)
                # print((min(con_Xs), cY))
                # print((max(con_Xs), cY))
                # print((cX, min(con_Ys)))
                # print((cX, max(con_Ys)))

                # print("lightness at ", cX, ",", cY, ": ", l_chan[cY,cX])
                # print("    average near it: ", avg_near_point(l_chan, cX, cY))
                # print("    average north: ", avg_near_point(l_chan, cX, min(con_Ys)-3))
                # print("    average south: ", avg_near_point(l_chan, cX, max(con_Ys)+3))
                # print("    average east: ", avg_near_point(l_chan, max(con_Xs)+3, cY))
                # print("    average west: ", avg_near_point(l_chan, min(con_Xs)-3, cY))

                avpt = self.avg_near_point(l_chan, cX, cY, 1)
                north_y = min(con_Ys)-3
                south_y = max(con_Ys)+3
                east_x = max(con_Xs)+3
                west_x = min(con_Xs)-3

                if (avpt > self.avg_near_point(l_chan, cX, north_y, 1) and
                    avpt > self.avg_near_point(l_chan, cX, south_y, 1) and
                    avpt > self.avg_near_point(l_chan, east_x, cY, 1) and
                    avpt > self.avg_near_point(l_chan, west_x, cY, 1)):
                    contours_final.append(con)
                    centroids_keep.append((cX, cY))
                    extent_pts.append([(east_x, cY),
                                       (west_x, cY),
                                       (cX, north_y),
                                       (cX, south_y)])

            # Create mask for the circles found
            mask = np.zeros_like(s_chan)
            cv2.drawContours(mask, contours_final, -1, 255, -1)
            masked_img = cv2.bitwise_and(s_chan, mask)

            # Create mask to weed out any blue color
            lower_red = np.array([0, 0, 0])
            upper_red = np.array([80, 255, 255])
            hue_mask1 = cv2.inRange(hls, lower_red, upper_red)
            lower_red = np.array([170, 0, 0])
            upper_red = np.array([180, 255, 255])
            hue_mask2 = cv2.inRange(hls, lower_red, upper_red)
            hue_mask = hue_mask1 + hue_mask2
            masked_img = cv2.bitwise_and(masked_img, hue_mask)

            # Create mask for high saturation (as colored lights are expected to be)
            ret, sat_thresh = cv2.threshold(masked_img, 180, 255, cv2.THRESH_BINARY)

            color_mean = cv2.mean(img, sat_thresh)
            if color_mean[2] > 100:
                if color_mean[1] > 128:
                    color_logic = 1     # Yellow light
                else:
                    color_logic = 0     # Red light
            elif color_mean[1] > 128:
                color_logic = 2         # Green light

        #TODO implement light color prediction
        return color_logic

    def get_classification(self, image, is_site):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if is_site:
            return self.classify_real_image(image)
        else:
            return self.classify_sim_image(image)