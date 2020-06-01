import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# Helper function: gets average pixel value of the 9 pixels +1 to -1 from the inputted x and y.
# Only works on 1-channel images for now, but you can use it with n-channel arrays by changing
# 'avg' to be an array.
# Eg. avg = [0, 0, 0] for a 3-channel image
def avg_near_point(img, x, y):
    avg = 0
    count = 0
    for i in [-1,0,1]:
        if 0 <= y+i < img.shape[0]:
            for j in [-1,0,1]:
                if 0 <= x+j < img.shape[1]:
                    avg += img[y+i, x+j]
                    count += 1
    if count > 0:
        return avg//count
    else:
        return 0


# Main processing function.
def process_image(folder, file):
    img = cv2.imread(folder + file)
    img = cv2.GaussianBlur(img, (3,3), 0)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_chan = hls[:,:,1]
    s_chan = hls[:,:,2]
    edges = cv2.Canny(img,50,120)

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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

            avpt = avg_near_point(l_chan, cX, cY)
            north_y = min(con_Ys)-3
            south_y = max(con_Ys)+3
            east_x = max(con_Xs)+3
            west_x = min(con_Xs)-3

            if (avpt > avg_near_point(l_chan, cX, north_y) and
                avpt > avg_near_point(l_chan, cX, south_y) and
                avpt > avg_near_point(l_chan, east_x, cY) and
                avpt > avg_near_point(l_chan, west_x, cY)):
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
        color_logic = "DUNNO"
        if color_mean[2] > 100:
            if color_mean[1] > 128:
                color_logic = "YELLOW"     # Yellow light
            else:
                color_logic = "RED"     # Red light
        elif color_mean[1] > 128:
            color_logic = "GREEN"         # Green light


        print(file, color_mean, color_logic)

        plt.imshow(sat_thresh, cmap='gray')
        f = file.split(".")
        plt.imsave(folder + 'processed/' + f[0] + "-mask.jpg", sat_thresh)

        # Plots for visualization
        cv2.drawContours(img, contours_final, -1, (0,255,255), 1)
        for i in range(len(contours_final)):
            cv2.circle(img, centroids_keep[i], 1, (0, 255, 255), 2)
            cv2.circle(img, extent_pts[i][0], 1, (255, 255, 0), 2)
            cv2.circle(img, extent_pts[i][1], 1, (255, 255, 0), 2)
            cv2.circle(img, extent_pts[i][2], 1, (255, 255, 0), 2)
            cv2.circle(img, extent_pts[i][3], 1, (255, 255, 0), 2)

    # Save image
    plt.imshow(img)
    plt.imsave(folder + 'processed/' + file, img)


# Obtain list of images in directory
FOLDER = 'frames/'

filelist=os.listdir(FOLDER)
filelist.sort()
for fichier in filelist[:]: # filelist[:] makes a copy of filelist.
    if not (fichier.endswith(".jpg")):
        filelist.remove(fichier)
    elif (fichier.endswith("-a.jpg")):
        os.system('rm ' + FOLDER + fichier)
    else:
        process_image(FOLDER, fichier)

        # img = cv2.imread(FOLDER + fichier)
        # img = cv2.GaussianBlur(img, (5, 5), 0)
        # l_chan = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:, :, 2]
        #
        # # Make mask for all remaining contours
        #
        # # Multiply mask to image to isolate pixels
        #
        # # Threshold remaining pixels on saturation
        #
        # # Average remaining pixels
        #
        # # Read hue value
        #
        # plt.imshow(l_chan)
        # plt.show()