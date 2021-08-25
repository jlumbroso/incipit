from recordtype import recordtype
import binascii
import collections
import io
import os
import pathlib
import random
import string
import tempfile
import typing
import sys
import cv2
import numpy
import PIL.Image
import PyPDF4
import loguru


Image = numpy.ndarray
ImagePIL = PIL.Image.Image

Region = recordtype(
    "Region",
    ["x0", "y0", "x1", "y1", "w", "h", "size", "area", "image", "image_crc32"]
)

IndexedRegion = recordtype(
    "Region",
    ["page", "index", "x0", "y0", "x1", "y1", "w", "h", "size", "area", "image", "image_crc32"]
)


# REGION_BOX_COLOR = (87, 177, 130)
REGION_BOX_COLOR = (109, 177, 84)
REGION_BOX_COLOR_IGNORE = (200, 161, 74)
REGION_BOX_THICKNESS = 3
REGION_BOX_FONT = cv2.FONT_HERSHEY_SIMPLEX
REGION_BOX_FONT_COLOR = (87, 177, 130)


# global variable
_LOG_OUTPUT_FOLDER: typing.Optional[pathlib.Path] = None


def region_to_indexed_region(
        region: Region,
        index: int,
        page: int,
) -> IndexedRegion:
    region_dict = region._asdict()
    region_dict.update({
        "page": page,
        "index": index,
    })
    return IndexedRegion(**region_dict)


def reindex_indexed_regions(
        regions: typing.List[IndexedRegion]
) -> typing.List[IndexedRegion]:

    reindexed_regions = []

    for index, region in enumerate(regions):
        region_dict = region._asdict()
        region_dict.update({
            "index": index,
        })
        reindexed_regions.append(IndexedRegion(**region_dict))

    return reindexed_regions


def log_region(
        region: typing.Union[Region, IndexedRegion],
        region_id: typing.Optional[int] = None
):

    # caption
    caption = "Region"
    if region_id is not None:
        caption = "{} {}".format(caption, region_id)

    loguru.logger.opt(depth=1).debug(
        (
            "{caption}: ({x0}, {y0}) -> ({x1}, {y1}), {w}x{h}, "
            "size: {size}, area: {area}, image: {image_crc32}"
        ),
        caption=caption,
        x0=region.x0, y0=region.y0, x1=region.x1, y1=region.y1,
        w=region.w, h=region.h, size=region.size, area=region.area,
        image_crc32=region.image_crc32,
    )


def split_regions_by_image(
        regions: typing.List[typing.Union[Region, IndexedRegion]],
) -> typing.List[typing.List[typing.Union[Region, IndexedRegion]]]:
    regions.sort(
        key=(
            lambda region:
            (
                region._asdict().get("page"),
                region.image_crc32,
                region._asdict().get("index")
            )
        ))

    distinct_images = []
    distinct_images_crc32 = []
    regions_by_image_index = []

    for region in regions:

        if region.image_crc32 not in distinct_images_crc32:
            # a new image is found!
            distinct_images.append(region.image)
            distinct_images_crc32.append(region.image_crc32)

            # create empty list for this image's regions
            regions_by_image_index.append([])

        # at this point the crc32 should definitely be in the list
        image_index = distinct_images_crc32.index(region.image_crc32)

        regions_by_image_index[image_index].append(region)

    return regions_by_image_index


def adjust_region_position_by_image(
        regions: typing.List[typing.Union[Region, IndexedRegion]],
        expand_vertical_ratio_of_system: float = 0.5,
        expand_vertical_ratio_of_gap: float = 0.45,
)-> typing.List[typing.Union[Region, IndexedRegion]]:
    """The function is to expand the rectangle boundary for the systems detected within the same page. The purpose of this function is to make sure the adjusted system covers the whole area including those not detected initially as the core rectangle.
       At horizontal level, it moves each system's left and right boundary to be the same as the leftmost and rightmost boundary among all the systems.
       At vertical level, it moves the top boundary higher by the size of the smaller value of the minimum gap between systems adjusted by expand_vertical_ratio_of_gap and the height of the current system adjusted by expand_vertical_ratio_of_system.
       It also moves the bottom boundary lower by the same amount as for the top boundary.

    Args:
        regions (typing.List[typing.Union[Region, IndexedRegion]]): The list of systems detected within the same page
        expand_vertical_ratio_of_system (float): The ratio applied to the current height of the system
        expand_vertical_ratio_of_gap(float): The ratio applied to the smallest vertical gap among all systems

    Returns:
        The adjusted systems within the same page
    """
    leftmost_system_boundary = sys.maxsize
    rightmost_system_boundary = 0
    previous_system_bottom_y_coordinate = 0
    index = 0
    min_vertical_gap_between_systems = sys.maxsize
    # In first pass of regions, we detect leftmost, rightmost boundaries and the minimum vertical gap between systems
    for region in regions:
        leftmost_system_boundary = min(leftmost_system_boundary, region.x0)
        rightmost_system_boundary = max(rightmost_system_boundary, region.x1)
        if( index > 0 ):
            min_vertical_gap_between_systems = min(region.y0 - previous_system_bottom_y_coordinate, min_vertical_gap_between_systems)
        previous_system_bottom_y_coordinate = region.y1
        index += 1

    # In second pass of regions, we normalize regions widths to leftmost and rightmost boundaries,
    # and regions height based on minimum gap between systems and the current system height
    # (FIXME: assumes image is not crooked/skewed; future work should attempt to detect skew and
    # limit corrections when too much skew, or fail when too much skew)
    for region in regions:
        region.x0 = leftmost_system_boundary
        region.x1 = rightmost_system_boundary
        current_region_height = region.y1 - region.y0
        image_height = region.image.shape[0]
        region.y0 = max(region.y0 - int(min ( min_vertical_gap_between_systems * expand_vertical_ratio_of_gap, current_region_height * expand_vertical_ratio_of_system )), 1)
        region.y1 = min(region.y1 + int(min ( min_vertical_gap_between_systems * expand_vertical_ratio_of_gap, current_region_height * expand_vertical_ratio_of_system)), image_height)

    return regions


def draw_region_on_image(
        image: Image,
        region: typing.Union[Region, IndexedRegion],
        callback_ignore_region: typing.Optional[typing.Callable] = None,
) -> Image:
    color = REGION_BOX_COLOR

    if callback_ignore_region is not None and callback_ignore_region(region):
        color = REGION_BOX_COLOR_IGNORE

    image = cv2.rectangle(
        image,
        (region.x0, region.y0),
        (region.x1, region.y1),
        color=color,
        thickness=REGION_BOX_THICKNESS
    )

    if type(region) is IndexedRegion:
        image = cv2.putText(
            image,
            str(region.index),
            (region.x0, region.y1),
            REGION_BOX_FONT, 4, REGION_BOX_FONT_COLOR, 10, cv2.LINE_AA)

    return image


def draw_regions_on_images(
        regions: typing.List[typing.Union[Region, IndexedRegion]],
):
    regions_by_image = split_regions_by_image(regions=regions)
    images_with_regions = []

    for image_index, regions in enumerate(regions_by_image):

        # skip empty images
        # (should never happen BTW, but better be cautious...)
        if len(regions) == 0:
            continue

        # image is the same in all regions, so take the one from
        # the first region
        image = regions[0].image

        # defensively copy
        image_with_regions = image.copy()

        # expand the regions' boundary to cover the whole system
        regions = adjust_region_position_by_image(regions)

        # draw regions
        for region in regions:
            image_with_region = draw_region_on_image(
                image=image_with_regions,
                region=region,
            )

        images_with_regions.append(image_with_regions)

    return images_with_regions


# Borrowed from:
# - https://stackoverflow.com/a/65634189/408734

def convert_from_cv2_to_image(image: Image) -> ImagePIL:
    # return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return PIL.Image.fromarray(image)


def convert_from_image_to_cv2(
        image: ImagePIL,
        resize: typing.Optional[float] = None,
        to_8bit: bool = False,
) -> Image:
    # return cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)
    # return numpy.asarray(img)

    array = numpy.float32(numpy.asarray(image))

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
        adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=9,
        C=30)
    log_intermediate_image(img=thresh, caption="3_adaptive_threshold")

    # Dilate to combine adjacent text contours
    kernel = cv2.getStructuringElement(
        shape=cv2.MORPH_RECT,
        ksize=(3, 3))
    dilate_start = cv2.dilate(
        src=thresh,
        kernel=kernel,
        iterations=4)
    log_intermediate_image(img=dilate_start, caption="4_dilated")

    kernel_erode_vertical = cv2.getStructuringElement(
        shape=cv2.MORPH_RECT,
        ksize=(1, 150)
    )
    erode_vertical = cv2.erode(
        src=dilate_start,
        kernel= kernel_erode_vertical,
        iterations = 1
    )
    kernel_erode_vertical_thin_line = cv2.getStructuringElement(
        shape=cv2.MORPH_RECT,
        ksize=(8, 1)
    )
    erodeVerticalAfterThinLine = cv2.erode(
        src=erode_vertical,
        kernel= kernel_erode_vertical_thin_line,
        iterations = 1
    )

    kernel_erode_horizontal = cv2.getStructuringElement(
        shape=cv2.MORPH_RECT,
        ksize=(200, 1)
    )
    erode_horizontal = cv2.erode(
        src=dilate_start,
        kernel= kernel_erode_horizontal,
        iterations = 1
    )

    erodeMerge = erode_horizontal + erodeVerticalAfterThinLine

    # Dilate to combine adjacent text contours
    kernel = cv2.getStructuringElement(
        shape=cv2.MORPH_RECT,
        ksize=(50, 9))
    dilate_end = cv2.dilate(
        src=erodeMerge,
        kernel=kernel,
        iterations=4)
    log_intermediate_image(img=dilate_end, caption="4_dilateEnd")

    # Find contours, highlight text areas, and extract ROIs
    contours = cv2.findContours(dilate_end, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    loguru.logger.debug("# contours found = {len}", len=len(contours))

    regions = []
    image_with_regions = image.copy()
    numbering = 0

    # hash image to make it easier to identify regions from different images
    image_crc32 = binascii.crc32(image.data)

    # somehow the contours are from bottom to top
    for c in reversed(contours):
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)

        region = Region(
            x0=x, y0=y, x1=(x+w), y1=(y+h),
            w=w, h=h, size=(w*h), area=area,
            image=image,
            image_crc32=image_crc32,
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


def compute_region_to_image_ratios(
        region: typing.Union[Region, IndexedRegion],
        image: typing.Optional[Image] = None,
) -> typing.Tuple[float, float]:

    # default to the region's image if none is provided
    image = image or region.image

    # find image size; shape as (H, W, D)
    img = region.image
    shape = img.shape
    h, w = shape[0], shape[1]

    landscape = (w < h * 0.90)

    ratio_w = region.w / w * 100.0
    ratio_h = region.h / h * 100.0

    ratios = (ratio_w, ratio_h)

    return ratios


def extract_region_from_image(
        region: typing.Union[Region, IndexedRegion],
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
                # not clear these parameters are always the right ones
                image = convert_from_image_to_cv2(
                    image=image,
                    resize=100,
                    to_8bit=True)

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


def load_images_from_input_document(
        path: str
) -> typing.List[Image]:

    loguru.logger.debug("Loading input document: {}", path)

    abspath = os.path.abspath(path)
    _, ext = os.path.splitext(abspath)

    # check existence!
    if not os.path.exists(abspath):
        loguru.logger.info(
            "=> DOES NOT EXIST, returning empty list"
        )
        return []

    # normalize to lowercase + remove dot
    ext = ext.lower()[1:]

    images = []

    if ext == "pdf":
        loguru.logger.debug("=> detecting it is PDF, using `extract_image_from_pdf()`")
        images = extract_images_from_pdf(
            pdf_path=abspath,
            as_numpy_images=True)

    elif ext in ["png", "jpg", "jpeg", "jp2", "tif", "tiff"]:
        loguru.logger.debug("=> detecting it is image, using `cv2.imread()`")
        image = cv2.imread(abspath)
        images = [image]

    else:
        loguru.logger.info("=> unsupported input document: '{}", len(images))

    loguru.logger.debug("=> loaded {} images", len(images))

    return images


def detect_regions_from_images(
        images: typing.List[Image],
        page_numbers_to_keep: typing.Optional[typing.List[int]] = None,
) -> typing.List[IndexedRegion]:

    overall_regions = []

    number_from = 0
    for page, image in enumerate(images):

        # see if this page should be skipped
        if page_numbers_to_keep is not None and page not in page_numbers_to_keep:
            loguru.logger.debug(
                "skipping page {} as it is not in `page_numbers_to_keep`: {}",
                page=page, page_numbers_to_keep=page_numbers_to_keep,
            )
            continue

        # call function to extract regions of interest from image
        _, regions = detect_regions_from_image(
            image=image,
            number_from=number_from,
        )

        # add all regions with page number to selection
        for index, region in enumerate(regions):
            overall_regions.append(
                region_to_indexed_region(
                    region=region,
                    index=(number_from + index),
                    page=page,
                ))

        # shift numbering
        number_from += len(regions)

    loguru.logger.debug(
        "found {} regions in total from {} images",
        len(overall_regions), len(images),
    )

    return overall_regions


def detect_regions_from_input_document(
        path: str,
        page_numbers_to_keep: typing.Optional[typing.List[int]] = None,
) -> typing.List[IndexedRegion]:

    images = load_images_from_input_document(path=path)
    regions = detect_regions_from_images(
        images=images,
        page_numbers_to_keep=page_numbers_to_keep,
    )

    loguru.logger.info(
        "found {} regions from {} images in document: '{}'",
        len(regions), len(images), path,
    )

    return regions

