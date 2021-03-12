
import os
import typing

import click
import click_help_colors
import loguru

import incipit
import incipit.processing
import incipit.heuristics


@click.command(
    cls=click_help_colors.HelpColorsCommand,
    help_headers_color='bright_green',
    help_options_color='green'
)
@click.option("-a", "--audit", is_flag=True, help="Visualize staff detection across document")
@click.option("-c", "--count", is_flag=True, help="Output number of detected staves")
@click.option("-p", "--pages", type=str, help="List of pages to process (e.g., '0', '0,-1')")
@click.option("-#", "--staves", type=str, help="List of staves to extract (e.g., '0', '0,-1')")
@click.option("-h", "--height-threshold", type=float, default=10.0, help="% of height threshold for staff detection")
@click.option("-w", "--width-threshold", type=float, default=70.0, help="% of width threshold for staff detection")
@click.option("-o", "--output", help="Output file pattern")
@click.option("-v", "--verbose", is_flag=True, default=False, help="Print debug information")
@click.argument("input", type=click.Path(exists=True))
def main(audit, count, pages, staves, height_threshold, width_threshold, output, verbose, input):
    """
    Extract the first (or any) staff from a black-and-white modern
    musical score, either available as an image file or PDF. For instance,

        $ incipit -o "555-incipit.png" ./input/pdf/555-gilbert.pdf

    will extract the first staff of sheet music `555-gilbert.pdf' as the
    image `555-incipit.png`. And,

        $ incipit --output "1__{}.png" -# '0,-1' ./input/pdf/1-gilbert.pdf

    will extract the first and last staves of `1-gilbert.pdf` and output
    these as the images `1__0.png' (first staff) and `1__10.png' (last staff).
    """

    if not verbose:
        loguru.logger.remove()

    loguru.logger.debug(
        (
            "CLI called with: output={output}, audit={audit}, pages={pages}, "
            "staves={staves}, count={count}, input={input},"
            "width_treshold={width_threshold}, height_threshold={height_threshold}"
        ),
        output=output, audit=audit, pages=pages, staves=staves, count=count, input=input,
        width_threshold=width_threshold, height_threshold=height_threshold,
    )

    if pages is not None:
        pages = list(map(int, pages.split(",")))
        loguru.logger.debug(
            "requested pages={pages}",
            pages=pages,
        )

    if audit:
        loguru.logger.debug("AUDIT command activated")

        if output is None:
            basename, _ = os.path.splitext(os.path.basename(input))
            output = "{}_audit_p{{:02}}.png".format(basename)
            loguru.logger.debug("using '{}' as output pattern", output)

        staff_audit_filenames = incipit.heuristics.write_staff_detection_images_from_input_document(
            input_path=input,
            output_filename_pattern=output,
            width_proportion=width_threshold,
            height_proportion=height_threshold,
            page_numbers_to_keep=pages,
        )

        loguru.logger.info("audit yielded following files: {}", ", ".join(staff_audit_filenames))

        return

    if staves is not None:
        staves = list(map(int, staves.split(",")))
        loguru.logger.debug(
            "requested staves={staves}",
            staves=staves,
        )

    loguru.logger.debug(
        "requested pages={pages}; staves={staves}",
        pages=pages, staves=staves,
    )

    if count and output is None and staves is None:
        loguru.logger.debug("COUNT method activated")
        staves = incipit.heuristics.detect_staves_from_input_document(
            input_path=input,
            width_proportion=width_threshold,
            height_proportion=height_threshold,
            page_numbers_to_keep=pages,
        )
        print(len(staves))
        return

    if output is None:
        basename, _ = os.path.splitext(os.path.basename(input))
        output = "{}_staff_{{:02}}.png".format(basename)
        loguru.logger.debug("using '{}' as output pattern", output)

    extracted_staves = incipit.heuristics.extract_staves_from_input_document(
        input_path=input,
        output_filename_pattern=output,
        staves_to_keep=staves,
        width_proportion=width_threshold,
        height_proportion=height_threshold,
        page_numbers_to_keep=pages,
    )

    extract_staves_filenames = list(map(lambda x: x[0], extracted_staves))
    loguru.logger.debug("Extracted staves: {}", ",".join(extract_staves_filenames))

    if count:
        print(len(extract_staves_filenames))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        loguru.logger.debug("KeyboardInterrupted happened; caught gracefully...")
        pass


