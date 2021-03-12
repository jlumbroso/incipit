
import collections
import io
import pathlib
import random
import string
import tempfile
import typing

import cv2
import numpy
import PIL
import PyPDF4
import loguru


Image = numpy.ndarray
ImagePIL = PIL.Image.Image

Region = collections.namedtuple(
    "Region",
    ["x0", "y0", "x1", "y1", "w", "h", "size", "area", "image"]
)


def log_region(region: Region, region_id: typing.Optional[int] = None):

    # caption
    caption = "Region"
    if region_id is not None:
        caption = "{} {}".format(caption, region_id)

    loguru.logger.opt(depth=1).debug(
        "{caption}: ({x0}, {y0}) -> ({x1}, {y1}), {w}x{h}, size: {size}, area: {area}",
        caption=caption,
        x0=region.x0, y0=region.y0, x1=region.x1, y1=region.y1,
        w=region.w, h=region.h, size=region.size, area=region.area,
    )


# REGION_BOX_COLOR = (87, 177, 130)
REGION_BOX_COLOR = (109, 177, 84)
REGION_BOX_COLOR_IGNORE = (200, 161, 74)
REGION_BOX_THICKNESS = 3
REGION_BOX_FONT = cv2.FONT_HERSHEY_SIMPLEX
REGION_BOX_FONT_COLOR = (87, 177, 130)


# global variable
_LOG_OUTPUT_FOLDER: typing.Optional[pathlib.Path] = None


# Borrowed from:
# - https://stackoverflow.com/a/65634189/408734

def convert_from_cv2_to_image(img: Image) -> ImagePIL:
    # return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return PIL.Image.fromarray(img)


def convert_from_image_to_cv2(
        img: ImagePIL,
        resize: typing.Optional[float] = None,
        to_8bit: bool = False,
) -> Image:
    # return cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)
    # return numpy.asarray(img)

    array = numpy.float32(numpy.asarray(img))

    if resize is not None:
        scale_percent = resize  # percent of original size
        width = int(array.shape[1] * scale_percent / 100)
        height = int(array.shape[0] * scale_percent / 100)
        dim = (width, height)

        # resize image
        resized = cv2.resize(array, dim, interpolation=cv2.INTER_AREA)
        array = resized

    # normalize image
    if to_8bit:
        normalized = cv2.normalize(array, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        array = normalized

    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def get_log_output_folder() -> pathlib.Path:
    global _LOG_OUTPUT_FOLDER

    if _LOG_OUTPUT_FOLDER is None:
        dirpath = tempfile.mkdtemp(suffix="-incipit")
        _LOG_OUTPUT_FOLDER = pathlib.Path(dirpath)

    return _LOG_OUTPUT_FOLDER


def log_intermediate_image(img: Image, caption: str = "", ext: str = "tiff"):

    def _write_image_file(img, caption, ext):
        # validation
        if caption is None or type(caption) is not str:
            caption = ""

        # unique id
        letters = string.ascii_letters
        unique_str = ''.join(random.choice(letters) for i in range(10))

        # filename
        filename = "{}_{}.{}".format(caption or "image", unique_str, ext)
        filepath = get_log_output_folder() / filename

        # output
        cv2.imwrite(str(filepath), img)

        return filepath

    loguru.logger.opt(depth=1, lazy=True).debug(
        "Logged intermediate `{caption}' image: {filepath}",
        caption=lambda: caption,
        filepath=lambda: _write_image_file(
            img=img, caption=caption, ext=ext))


# Adapted from these remarkable articles by @akash-ch2812:
# - https://towardsdatascience.com/extracting-text-from-scanned-pdf-using-pytesseract-open-cv-cd670ee38052
# - https://gist.github.com/akash-ch2812/d42acf86e4d6562819cf4cd37d1195e7

def detect_regions_from_image(
        image: typing.Optional[Image] = None,
        image_path: typing.Optional[str] = None,
        region_size_threshold: typing.Optional[float] = None,
        number_from: typing.Optional[int] = None,
        callback_skip_region: typing.Optional = None,
        callback_ignore_region: typing.Optional = None,
) -> typing.Tuple[Image, typing.List[Region]]:

    if image is None and image_path is None:
        raise ValueError(
            "either provide an `image` or `image_path`; "
            "both cannot be None"
        )

    if image is None:
        image = cv2.imread(image_path)
        loguru.logger.debug("Read image from: {image_path}", image_path=image_path)
    else:
        # defensive copy
        image = image.copy()

    # log image that is being worked on
    log_intermediate_image(img=image, caption="0_original")

    # default values
    image_h = image.shape[0]  # number of rows
    image_w = image.shape[1]

    loguru.logger.debug(
        "Image: {width}x{height}, {type}",
        width=image_w,
        height=image_h,
        type=type(image),
    )

    percent_threshold = 1.5
    absolute_threshold = min(
        image_w*percent_threshold/100.0,
        image_h*percent_threshold/100.0)
    region_size_threshold = region_size_threshold or absolute_threshold

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    log_intermediate_image(img=gray, caption="1_grayscaled")

    blur_kernel_size = (9, 9)
    blur_stddev = 0
    blur = cv2.GaussianBlur(
        src=gray, ksize=blur_kernel_size, sigmaX=blur_stddev)
    log_intermediate_image(img=blur, caption="2_blurred")

    thresh = cv2.adaptiveThreshold(
        src=blur,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=11,
        C=30)
    log_intermediate_image(img=thresh, caption="3_adaptive_threshold")

    # Dilate to combine adjacent text contours
    kernel = cv2.getStructuringElement(
        shape=cv2.MORPH_RECT,
        ksize=(9, 9))
    dilate = cv2.dilate(
        src=thresh,
        kernel=kernel,
        iterations=4)
    log_intermediate_image(img=thresh, caption="4_dilated")

    # Find contours, highlight text areas, and extract ROIs
    contours = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    loguru.logger.debug("# contours found = {len}", len=len(contours))

    regions = []
    image_with_regions = image.copy()
    numbering = 0

    # somehow the contours are from bottom to top
    for c in reversed(contours):
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)

        region = Region(
            x0=x, y0=y, x1=(x+w), y1=(y+h),
            w=w, h=h, size=(w*h), area=area,
            image=image,
        )

        log_region(region=region)

        # default region skipping
        if region.w < region_size_threshold or region.h < region_size_threshold:
            loguru.logger.debug(
                "-> skipped, too small (region_size_threshold={})",
                region_size_threshold
            )
            continue

        # callback provided region skipping
        if callback_skip_region is not None and callback_skip_region(region):
            loguru.logger.debug("-> skipped, callback_skip_region == True")
            continue

        color = REGION_BOX_COLOR
        if callback_ignore_region is not None and callback_ignore_region(region):
            loguru.logger.debug("-> colored ignore, callback_ignore_region == True")
            color = REGION_BOX_COLOR_IGNORE

        image_with_regions = cv2.rectangle(
            image_with_regions,
            (region.x0, region.y0),
            (region.x1, region.y1),
            color=color,
            thickness=REGION_BOX_THICKNESS
        )

        if number_from is not None:
            image_with_regions = cv2.putText(
                image_with_regions,
                str(numbering),
                (region.x0, region.y1),
                REGION_BOX_FONT, 4, REGION_BOX_FONT_COLOR, 10, cv2.LINE_AA)

        loguru.logger.debug("=> drew region and added to return value")
        log_region(region=region, region_id=numbering)

        regions.append(region)
        numbering += 1

    loguru.logger.debug("Found {} regions", len(regions))

    return image_with_regions, regions


def extract_region_from_image(
        region: Region,
        image: typing.Optional[Image] = None,
) -> Image:

    # default to the region's image if none is provided
    image = image or region.image

    # crop out the region using numpy's array notation
    return image[
       region.y0:region.y1,
       region.x0:region.x1
    ]


# Adapted from:
# - https://github.com/claird/PyPDF4/blob/2e2eec1d09d61a6127ad5b21a3743af389a03744/scripts/pdf-image-extractor.py
# - http://stackoverflow.com/questions/2693820/extract-images-from-pdf-without-resampling-in-python
# with help from:
# - https://stackoverflow.com/a/32908899/408734

def extract_image_from_pdf_x_obj(x_object, obj, mode, size, data):

    image = None

    if "/Filter" in x_object[obj]:

        x_filter = x_object[obj]["/Filter"]

        if x_filter == "/FlateDecode":
            # uncompressed data
            image = PIL.Image.frombytes(mode, size, data)

        elif x_filter == "/DCTDecode":
            # JPG file (.jpg)
            image = PIL.Image.open(io.BytesIO(data))

        elif x_filter == "/JPXDecode":
            # JP2 file (.jp2)
            image = PIL.Image.open(io.BytesIO(data))

        elif x_filter == "/CCITTFaxDecode":
            # TIFF file (.tif)
            image = PIL.Image.open(io.BytesIO(data))

    return image


def extract_images_from_pdf(
        pdf_path,
        resort_by_page_number: bool = True,
        include_page_number: bool = False,
        as_numpy_images: bool = True,
):
    # NOTE:
    # a more powerful version is available here:
    # https://gist.github.com/gstorer/f6a9f1dfe41e8e64dcf58d07afa9ab2a

    pdf_file = PyPDF4.PdfFileReader(open(pdf_path, "rb"))

    images = []

    for page_number in range(pdf_file.numPages):

        page_obj = pdf_file.getPage(page_number)

        if "/XObject" not in page_obj["/Resources"]:
            # skip a page with no embedded X-Object
            continue

        x_object = page_obj["/Resources"]["/XObject"].getObject()

        image_index = 0

        for obj in x_object:

            if x_object[obj]["/Subtype"] != "/Image":
                # skip objects that are not image
                continue

            size = (x_object[obj]["/Width"], x_object[obj]["/Height"])
            data = x_object[obj].getData()

            if x_object[obj]["/ColorSpace"] == "/DeviceRGB":
                mode = "RGB"
            else:
                mode = "P"

            image = extract_image_from_pdf_x_obj(
                x_object,
                obj,
                mode, size, data
            )

            if image is None:
                # skip when image extraction is unsuccessful
                continue

            if as_numpy_images:
                image = convert_from_image_to_cv2(image)

            images.append((page_number, image_index, image))
            image_index += 1

    if resort_by_page_number:
        images.sort(key=lambda item: (item[0], item[1]))

    if include_page_number:
        images = [
            (item[0], item[2])
            for item in images
        ]
    else:
        images = [
            item[2]
            for item in images
        ]

    return images

