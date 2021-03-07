
import collections
import typing

import cv2


Region = collections.namedtuple(
    "Region",
    ["x0", "y0", "x1", "y1", "w", "h", "size", "image"]
)


# REGION_BOX_COLOR = (87, 177, 130)
REGION_BOX_COLOR = (109, 177, 84)
REGION_BOX_THICKNESS = 3


def detect_regions_from_image(
        image: typing.Optional[typing.Any] = None,
        image_path: typing.Optional[str] = None,
        region_size_threshold: typing.Optional[float] = None
):
    if image is None and image_path is None:
        raise ValueError(
            "either provide an `image` or `image_path`; "
            "both cannot be None"
        )

    if image is None:
        image = cv2.imread(image_path)
    else:
        # defensive copy
        image = image.copy()

    # default values
    percent_threshold = 1.5
    absolute_threshold = min(
        image.shape[0]*percent_threshold/100.0,
        image.shape[1]*percent_threshold/100.0)
    region_size_threshold = region_size_threshold or absolute_threshold

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9,9), 0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,30)

    # Dilate to combine adjacent text contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    dilate = cv2.dilate(thresh, kernel, iterations=4)

    # Find contours, highlight text areas, and extract ROIs
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    regions = []
    image_with_regions = image.copy()
    for c in cnts:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)

        if w < region_size_threshold or h < region_size_threshold:
            continue

        image_with_regions = cv2.rectangle(
            image_with_regions,
            (x, y),
            (x+w, y+h),
            color=REGION_BOX_COLOR,
            thickness=REGION_BOX_THICKNESS
        )

        regions.append(
            Region(
                x0=x, y0=y, x1=(x+w), y1=(y+h),
                w=w, h=h, size=(w*h),
                image=image,
            )
        )

    return image_with_regions, regions


def extract_region_from_image(
        region: Region,
        image: typing.Optional[typing.Any] = None,
):
    # default to the region's image if none is provided
    image = image or region.image

    # crop out the region using numpy's array notation
    return image[
       region.y0:region.y1,
       region.x0:region.x1
    ]

