
import typing

import cv2
import loguru

import src.incipit.processing


def detect_staves_from_input_document(
        input_path: str,
        width_proportion: float = 55.0,
        height_proportion: float = 8.0,
        page_numbers_to_keep: typing.Optional[typing.List[int]] = None,
) -> typing.List[src.incipit.processing.IndexedRegion]:

    regions = src.incipit.processing.detect_regions_from_input_document(
        path=input_path,
        page_numbers_to_keep=page_numbers_to_keep,
    )

    # alias for shortness
    compute_ratios = src.incipit.processing.compute_region_to_image_ratios

    # select regions that meet these criteria
    staves = [
        region
        for region in regions
        if compute_ratios(region)[0] >= width_proportion
        if compute_ratios(region)[1] >= height_proportion
    ]
    
    # reindex staves
    staves = src.incipit.processing.reindex_indexed_regions(regions=staves)

    # log outcome
    loguru.logger.info(
        "Found {} staves from {} regions from document: '{}'",
        len(staves), len(regions), input_path,
    )

    return staves


def write_staff_detection_images_from_input_document(
        input_path: str,
        output_filename_pattern: str,
        width_proportion: float = 55.0,
        height_proportion: float = 8.0,
        page_numbers_to_keep: typing.Optional[typing.List[int]] = None,
) -> typing.List[str]:

    staves = detect_staves_from_input_document(
        input_path=input_path,
        width_proportion=width_proportion,
        height_proportion=height_proportion,
        page_numbers_to_keep=page_numbers_to_keep,
    )

    images_with_stave_regions = src.incipit.processing.draw_regions_on_images(
        regions=staves,
    )

    filenames = []
    for index, image in enumerate(images_with_stave_regions):
        filename = output_filename_pattern.format(index)
        cv2.imwrite(
            filename=output_filename_pattern.format(index),
            img=image,
        )
        filenames.append(filename)

    return filenames


def extract_staves_from_input_document(
        input_path: str,
        output_filename_pattern: typing.Optional[str] = None,
        staves_to_keep: typing.Optional[typing.List[int]] = None,
        width_proportion: float = 55.0,
        height_proportion: float = 8.0,
        page_numbers_to_keep: typing.Optional[typing.List[int]] = None,
) -> typing.Union[

        typing.List[src.incipit.processing.Image],
        typing.List[typing.Tuple[str, src.incipit.processing.Image]],
]:

    staves = detect_staves_from_input_document(
        input_path=input_path,
        width_proportion=width_proportion,
        height_proportion=height_proportion,
        page_numbers_to_keep=page_numbers_to_keep,
    )

    # by default, save the first staff ("incipit"!)
    normalized_staves_to_keep = [0]

    if staves_to_keep is not None and len(staves_to_keep) > 0:

        # process the indexes to get a list of staves to keep

        normalized_staves_to_keep = []
        for index in staves_to_keep:

            # positive index gets added if not out of range
            if index >= 0 and index < len(staves):
                normalized_staves_to_keep.append(index)
                continue

            # negative index wraps around (like expected in Python)
            if index < 0:
                index = len(staves) + index
                if index < len(staves):
                    normalized_staves_to_keep.append(index)
                    continue

    returned_staves = []

    for index, staff_region in enumerate(staves):

        # skip indexes we are not interested in
        if index not in normalized_staves_to_keep:
            continue

        staff_image = src.incipit.processing.extract_region_from_image(staff_region)

        if output_filename_pattern is not None:
            filename = output_filename_pattern.format(index)
            cv2.imwrite(
                filename=output_filename_pattern.format(index),
                img=staff_image,
            )
            returned_staves.append((filename, staff_image))

        else:
            returned_staves.append(staff_image)

    return returned_staves
