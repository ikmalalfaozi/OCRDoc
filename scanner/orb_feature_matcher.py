import cv2
import numpy as np
import imutils

class ORBFeatureMathcer(object):
    def __init__(self, max_features=500, keep_percent=0.2):
        self.max_features = max_features
        self.keep_percent = keep_percent

    def align_images(self, template_path, image_path, debug=False):
        # convert both the input document and template to grayscale
        template = cv2.imread(template_path, 0)
        document = cv2.imread(image_path)
        document = imutils.resize(document, height=template.shape[0])
        gray = cv2.cvtColor(document, cv2.COLOR_BGR2GRAY)

        # use ORB to detect keypoints and extract (binary local invariant features)
        orb = cv2.ORB_create(self.max_features)
        (kpsA, descsA) = orb.detectAndCompute(gray, None)
        (kpsB, descsB) = orb.detectAndCompute(template, None)

        # match the features
        method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
        matcher = cv2.DescriptorMatcher_create(method)
        mathces = matcher.match(descsA, descsB, None)

        # sort the matches by their distance (the smaller the distance, the "more similiar" the features are)
        matches = sorted(mathces, key=lambda x: x.distance, reverse=False)

        # keep only the top matches
        keep = int(len(matches) * self.keep_percent)
        matches = matches[:keep]

        # check to see if we should visualize the matched keypoints
        if debug:
            matched_vis = cv2.drawMatches(gray, kpsA, template, kpsB, matches, None)
            matched_vis = imutils.resize(matched_vis, width=1000)
            cv2.imshow("MatchedKeypoints", matched_vis)
            cv2.waitKey(0)

        # allocate memory for the keypoints (x, y-coordintaes) from the
        # top matches -- we'll use these coordinate to compute our homography
        # matrix
        ptsA = np.zeros((len(matches), 2), dtype='float')
        ptsB = np.zeros((len(matches), 2), dtype='float')

        # loop over the top matches
        for (i, m) in enumerate(matches):
            # indicate that the two keypoints in the respective images
            # map to each other
            ptsA[i] = kpsA[m.queryIdx].pt
            ptsB[i] = kpsB[m.trainIdx].pt

        # compute the homography matrix between the two sets of matched points
        (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)

        # use the homography matrix to align the images
        (h, w) = template.shape[:2]
        aligned = cv2.warpPerspective(document, H, (w, h))

        # returned the aligned image
        return aligned
