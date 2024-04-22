import cv2
import numpy as np
import open3d as o3d

MIN_MATCH_COUNT = 10

# Load Different Image Pairs
dir_name = "input_data/3DImagesStatue/images/"
img_pairs = [(dir_name + "a.jpg", dir_name + "b.jpg"),
             (dir_name + "b.jpg", dir_name + "c.jpg"),
             (dir_name + "c.jpg", dir_name + "d.jpg"),
             (dir_name + "d.jpg", dir_name + "e.jpg"),
             (dir_name + "e.jpg", dir_name + "f.jpg"),
             (dir_name + "f.jpg", dir_name + "g.jpg"),
             (dir_name + "g.jpg", dir_name + "h.jpg"),
             (dir_name + "h.jpg", dir_name + "i.jpg"),
             (dir_name + "i.jpg", dir_name + "j.jpg"),
             (dir_name + "j.jpg", dir_name + "k.jpg")]
counter = 0

# Intrinsic Matrix
K = np.array([[3410.99, 0.210676, 3048.69],
              [0., 3412.37, 2006.96],
              [0.,   0.,   1.]])

for img_pair_1, img_pair_2 in img_pairs:
    counter += 1
    img1 = cv2.imread(img_pair_1)
    img2 = cv2.imread(img_pair_2)

    ###############################
    # 1----SIFT feature matching---#
    ###############################

    # detect sift features for both images
    sift = cv2.SIFT_create(nfeatures=10000)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # use flann to perform feature matching
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    num_matches = len(good)
    print("Number of matches for image pair {}: {}".format(counter, num_matches))

    if num_matches > MIN_MATCH_COUNT:
        p1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        p2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        #######################
        # 2----essential matrix--#
        #######################
        E, mask = cv2.findEssentialMat(p1, p2, K, cv2.RANSAC, 0.999, 1.0)
        matchesMask = mask.ravel().tolist()

        ####################
        # 3----recoverpose--#
        ####################
        _, R, t, _ = cv2.recoverPose(E, p1, p2)

        #######################
        # 4----triangulation---#
        #######################

        # calculate projection matrix for both camera
        M_r = np.hstack((R, t))
        M_l = np.hstack((np.eye(3, 3), np.zeros((3, 1))))

        P_l = np.dot(K, M_l)
        P_r = np.dot(K, M_r)

        # undistort points
        p1 = p1[np.asarray(matchesMask) == 1, :, :]
        p2 = p2[np.asarray(matchesMask) == 1, :, :]

        p1_un = cv2.undistortPoints(p1, K, None)
        p2_un = cv2.undistortPoints(p2, K, None)
        p1_un = np.squeeze(p1_un)
        p2_un = np.squeeze(p2_un)

        # triangulate points, this requires points in normalized coordinate
        point_4d_hom = cv2.triangulatePoints(P_l, P_r, p1_un.T, p2_un.T)
        point_3d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
        point_3d = point_3d[:3, :].T

        ##############################
        # Draw inlier matches on images
        ##############################
        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)

        img_inliermatch = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
        cv2.imwrite('inlier_match_' + str(counter) + '.png', img_inliermatch)

        #######################
        # Overlay on original image
        #######################
        
        # Project 3D points to 2D
        points_2d, _ = cv2.projectPoints(point_3d, np.eye(3), np.zeros((3, 1)), K, None)

        # Convert to integer
        points_2d = np.int32(points_2d).reshape(-1, 2)

        # Draw points on original image
        for point in points_2d:
            cv2.circle(img1, tuple(point), 5, (0, 0, 255), -1)

        # Save the image with overlaid points
        cv2.imwrite('overlay_points_' + str(counter) + '.png', img1)
