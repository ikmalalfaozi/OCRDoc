import cv2
import numpy as np
import imutils

BF_MATCHER = "BF_MATCHER"
FLANN = "FLANN"

class SIFTFeatureMatcher(object):
    def __init__(self, max_features=500, keep_percent=0.2, matcher_type=BF_MATCHER):
        self.max_features = max_features
        self.keep_percent = keep_percent
        self.matcher_type = matcher_type.upper()

    def align_images(self, template_path, image_path, debug=False):
        # convert both the input document and template to grayscale
        template = cv2.imread(template_path, 0)
        document = cv2.imread(image_path)
        document = imutils.resize(document, height=template.shape[0])
        gray = cv2.cvtColor(document, cv2.COLOR_BGR2GRAY)

        # initialize SIFT detector
        sift = cv2.SIFT_create(self.max_features)

        # find the keypoints and descriptors with SIFT
        keypoints1, descriptors1 = sift.detectAndCompute(gray, None)
        keypoints2, descriptors2 = sift.detectAndCompute(template, None)

        # Create a matcher object
        # default BFMatcher
        if self.matcher_type == FLANN:
            # Set FLANN parameters
            index_params = dict(algorithm=1, trees=5)
            search_params = dict()

            # Create FLANN-based matcher object
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        # Match descriptors
        matches = matcher.match(descriptors1, descriptors2)

        # Sort matches by distance (best matches first)
        matches = sorted(matches, key=lambda x: x.distance)

        # keep only the top matches
        keep = int(len(matches) * self.keep_percent)
        matches = matches[:keep]

        # check to see if we should visualize the matched keypoints
        if debug:
            matchedVis = cv2.drawMatches(gray, keypoints1, template, keypoints2, matches, None)
            matchedVis = imutils.resize(matchedVis, width=1000)
            cv2.imshow("MatchedKeypoints", matchedVis)
            cv2.waitKey(0)

        # allocate memory for the keypoints (x, y-coordinates) from the
        # top matches -- we'll use these coordinates to compute our homography
        # matrix
        ptsA = np.zeros((len(matches), 2), dtype='float')
        ptsB = np.zeros((len(matches), 2), dtype='float')

        # loop over the top matches
        for (i, m) in enumerate(matches):
            # indicate that the two keypoints in the respective images
            # map to each other
            ptsA[i] = keypoints1[m.queryIdx].pt
            ptsB[i] = keypoints2[m.trainIdx].pt

        # compute the homography matrix between the two sets of matched
        # points
        (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)

        # use the homography matrix to align the images
        (h, w) = template.shape[:2]
        aligned = cv2.warpPerspective(document, H, (w, h))

        return aligned
