import gc
import cv2
import imutils
import numpy as np
import torch
import torchvision.transforms as torch_transforms
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
import scanner.utils as utils


class DocSegmentationScanner(object):
    def __init__(self, model_path='models/model_mbv3_iou_mix_2C_aux_ld.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path)
        self.preprocess_transforms = torch_transforms.Compose(
            [
                torch_transforms.ToTensor(),
                torch_transforms.Normalize((0.4611, 0.4359, 0.3905), (0.2193, 0.2150, 0.2109)),
            ]
        )
        self.IMAGE_SIZE = 384

    def scan(self, image: str | np.ndarray, shape=None, interactive=False):
        # read image
        if isinstance(image, str):
            img = cv2.imread(image, cv2.IMREAD_COLOR)[:, :, ::-1]
        else:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # get the corners of the document
        corners, img = self.get_corners(img)

        if interactive:
            corners = utils.interactive_get_contour(corners, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            corners = corners.astype(np.float32)

        # apply the perspective transformation
        warped = utils.four_point_transform(img, corners, shape)

        return warped[:, :, ::-1]

    def load_model(self, model_path):
        model = deeplabv3_mobilenet_v3_large(num_classes=2)
        model.to(self.device)
        checkpoints = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoints, strict=False)
        model.eval()
        _ = model(torch.randn((2, 3, 384, 384)).to(self.device))
        return model

    def get_corners(self, image):
        """
        Returns a numpy array of shape (4, 2) containing the vertices of the four corners
        of the document in the image. If no corners were found, it returns the original
        four corners.
        """
        out = self.extract(image)

        h, w, c = image.shape
        scale_x = w / self.IMAGE_SIZE
        scale_y = h / self.IMAGE_SIZE
        half = self.IMAGE_SIZE // 2
        r_h, r_w = out.shape

        _out_extended = np.zeros((self.IMAGE_SIZE + r_h, self.IMAGE_SIZE + r_w), dtype=out.dtype)
        _out_extended[half: half + self.IMAGE_SIZE, half: half + self.IMAGE_SIZE] = out
        out = _out_extended.copy()

        del _out_extended
        gc.collect()

        # Edge Detection.
        canny = cv2.Canny(out.astype(np.uint8), 225, 255)
        canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        contours, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        page = sorted(contours, key=cv2.contourArea, reverse=True)[0]

        epsilon = 0.02 * cv2.arcLength(page, True)
        corners = cv2.approxPolyDP(page, epsilon, True)

        corners = np.concatenate(corners).astype(np.float32)

        corners[:, 0] -= half
        corners[:, 1] -= half

        corners[:, 0] *= scale_x
        corners[:, 1] *= scale_y

        # check if corners are inside.
        # if not find smallest enclosing box, expand_image then extract document
        # else extract document
        if not (np.all(corners.min(axis=0) >= (0, 0)) and np.all(corners.max(axis=0) <= (w, h))):
            left_pad, top_pad, right_pad, bottom_pad = 0, 0, 0, 0

            rect = cv2.minAreaRect(corners.reshape((-1, 1, 2)))
            box = cv2.boxPoints(rect)
            box_corners = np.int32(box)

            box_x_min = np.min(box_corners[:, 0])
            box_x_max = np.max(box_corners[:, 0])
            box_y_min = np.min(box_corners[:, 1])
            box_y_max = np.max(box_corners[:, 1])

            # Find corner point which doesn't satisfy the image constraint
            # and record the amount of shift required to make the box
            # corner satisfy the constraint
            if box_x_min <= 0:
                left_pad = abs(box_x_min) + 10

            if box_x_max >= w:
                right_pad = (box_x_max - w) + 10

            if box_y_min <= 0:
                top_pad = abs(box_y_min) + 10

            if box_y_max >= h:
                bottom_pad = (box_y_max - h) + 10

            # new image with additional zeros pixels
            image_extended = np.zeros((top_pad + bottom_pad + h, left_pad + right_pad + w, c),
                                      dtype=image.dtype)

            # adjust original image within the new 'image_extended'
            image_extended[top_pad: top_pad + h, left_pad: left_pad + w, :] = image

            # find the closest corner from box corners
            closest_corners = self.find_closest_corners(corners, box_corners)
            # shifting 'closest_corners' the required amount
            closest_corners[:, 0] += left_pad
            closest_corners[:, 1] += top_pad

            corners = closest_corners
            image = image_extended

        corners = corners.tolist()
        corners = utils.order_points(corners)

        return corners, image

    def extract(self, image):
        image_model = cv2.resize(image, (self.IMAGE_SIZE, self.IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
        image_model = self.preprocess_transforms(image_model)
        image_model = torch.unsqueeze(image_model, dim=0).to(self.device)

        with torch.no_grad():
            out = self.model(image_model)["out"].cpu()

        del image_model
        gc.collect()

        out = torch.argmax(out, dim=1, keepdim=True).permute(0, 2, 3, 1)[0].numpy().squeeze().astype(np.int32)
        out = out * 255

        return out

    def find_closest_corners(self, corners, box_corners):
        closest_corners = []

        for box_corner in box_corners:
            min_dist = float('inf')
            closest_corner = None

            for corner in corners:
                dist = np.linalg.norm(corner - box_corner)
                if dist < min_dist:
                    min_dist = dist
                    closest_corner = corner

            closest_corners.append(closest_corner)

        return np.array(closest_corners)
