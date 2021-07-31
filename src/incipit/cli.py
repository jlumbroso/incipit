
import os

import click
import click_help_colors
import loguru

import src.incipit
import src.incipit.processing
import src.incipit.heuristics

systemNumberInPDFList = [11,11,	14,	13,	12,	12,	21,	12,	12,	12,	10,	24,	23,	13,	19,	23,	22,	20,	21,	16,	24,	18,	23,	24,	19,	22,	18,	22,	26,	24,	23,	4,	21,	5,	20,	18,	18,	15,	17,	4,	27,	5,	16,	22,	18,	28,	24,	22,	31,	24,	17,	18,	22,	19,	22,	21,	24,	19,	10,	11,	32,	19,	11,	10,	12,	11,	10,	19,	12,	11,	10,	12,	17,	9,	11,	12,	20,	11,	16,	6,	33,	31,	23,	18,	17,	21,	12,	47,	32,	49,	37,	14,	29,	5,	8,	24,	30,	19,	21,	16,	20,	18,	17,	33,	28,	15,	24,	20,	20,	20,	18,	20,	28,	30,	33,	20,	34,	23,	32,	20,	33,	22,	22,	26,	18,	22,	24,	26,	22,	18,	20,	21,	22,	26,	23,	23,	22,	21,	26,	14,	28,	19,	20,	11,	20,	17,	22,	12,	12,	18,	20,	17,	10,	19,	20,	14,	22,	15,	16,	22,	16,	25,	10,	20,	15,	17,	24,	21,	23,	25,	11,	30,	20,	31,	20,	28,	14,	12,	15,	18,	22,	19,	20,	24,	12,	15,	25,	27,	21,	18,	24,	24,	21,	23,	23,	17,	12,	18,	21,	20,	21,	25,	26,	11,	48,	26,	20,	10,	22,	12,	34,	25,	24,	23,	23,	32,	26,	19,	20,	30,	27,	19,	22,	22,	23,	18,	27,	20,	20,	24,	22,	22,	24,	20,	21,	20,	20,	15,	20,	42,	19,	22,	17,	20,	24,	20,	20,	28,	28,	21,	17,	18,	18,	21,	21,	21,	19,	24,	20,	37,	29,	22,	23,	36,	32,	19,	18,	26,	18,	28,	19,	22,	20,	15,	17,	20,	12,	18,	21,	21,	17,	27,	20,	24,	21,	16,	15,	17,	12,	18,	17,	18,	30,	21,	22,	33,	22,	28,	15,	19,	21,	30,	21,	12,	20,	23,	18,	18,	20,	22,	20,	20,	18,	32,	16,	23,	22,	20,	23,	21,	15,	17,	12,	20,	17,	19,	16,	21,	29,	12,	20,	22,	15,	17,	22,	18,	22,	17,	20,	20,	12,	18,	17,	16,	26,	15,	16,	18,	22,	17,	36,	23,	20,	20,	21,	35,	28,	20,	20,	26,	15,	17,	18,	22,	18,	20,	18,	34,	25,	20,	16,	15,	19,	21,	15,	15,	17,	21,	15,	18,	20,	18,	15,	13,	15,	22,	14,	25,	16,	24,	11,	30,	12,	31,	20,	30,	15,	25,	16,	15,	22,	33,	22,	35,	19,	20,	12,	16,	23,	20,	11,	20,	12,	29,	10,	22,	50,	23,	16,	27,	21,	30,	21,	20,	20,	22,	13,	12,	15,	15,	4,	16,	27,	20,	18,	16,	15,	24,	22,	12,	20,	18,	22,	20,	19,	12,	20,	20,	20,	20,	11,	18,	12,	26,	25,	19,	16,	23,	21,	36,	25,	18,	17,	20,	15,	21,	22,	25,	22,	31,	15,	15,	24,	18,	24,	24,	24,	24,	32,	24,	16,	22,	16,	20,	18,	30,	20,	20,	18,	23,	21,	24,	24,	25,	20,	24,	21,	20,	24,	23,	28,	20,	18,	15,	15,	16,	22,	20,	20,	20,	21,	20,	20,	15,	20,	21,	24,	24,	22,	20,	20,	21,	11,	20,	18,	23,	17,	15,	17,	21,	23,	22,	22,	11,	21,	21,	22,	20,	24,	22,	22,	22,	20,	10,	22,	20,	28,	20,	22,	21,	23,	20,	21,	21,	20,	20,	20,	11]
#systemNumberInPDFList = [3,	5,	5,	6,	6,	6,	5,	5,	6,	6,	7,	7,	5,	6,	6,	7,	6,	6,	6,	6,	6,	7,	5,	6,	6,	6,	6,	6,	6,	7,	6,	6,	5,	6,	6,	6,	6,	7,	5,	5,	5,	6,	6,	6,	6,	6,	6,	6,	5,	6,	6,	6,	6,	5,	5,	6,	5,	5,	6,	5,	5,	4,	6,	6,	5,	5,	5,	5,	6,	6,	6,	6,	6,	6,	6,	6,	5,	6,	6,	6,	6,	6,	6,	6,	5,	5,	5,	6,	6,	6,	6,	6,	6,	6,	6,	5,	6,	5,	5,	6,	5,	5,	6,	5,	6,	5,	6,	5,	5,	5,	5,	5,	5,	5,	5,	5,	6,	5,	5,	4,	5,	5,	7,	6,	6,	6,	6,	5,	6,	6,	6,	6,	5,	6,	5,	5,	4,	5,	5,	6,	6,	6,	6,	5,	5,	6,	6,	7,	6,	5,	5,	5,	5,	5,	5,	5,	7,	2,	2,	2,	2,	2,	5,	5,	5,	6,	6,	6,	6,	6,	6,	5,	5,	6,	6,	5,	5,	5,	6,	6,	6,	5,	5,	6,	6,	5,	5,	6,	6,	6,	6,	7,	6,	6,	6,	5,	5,	4,	5,	5,	5,	5,	5,	6,	6,	4,	7,	5,	7,	5,	6,	6,	6,	6,	6,	5,	5,	5,	5,	6,	5,	5,	5,	4,	5,	6,	6,	5,	5,	5,	5,	5,	5,	5,	6,	5,	5,	6]

@click.command(
    # Green is the best color in the world.
    cls=click_help_colors.HelpColorsCommand,
    help_headers_color='bright_green',
    help_options_color='green'
)
@click.option("-a", "--audit", is_flag=True, help="Visualize system detection across document")
@click.option("-c", "--count", is_flag=True, help="Output number of detected systems")
@click.option("-p", "--pages", type=str, help="List of pages to process (e.g., '0', '0,-1')")
@click.option("-#", "--systems", type=str, help="List of systems to extract (e.g., '0', '0,-1')")
@click.option("-h", "--height-threshold", type=float, default=9.0, help="% of height threshold for system detection")
@click.option("-w", "--width-threshold", type=float, default=55.0, help="% of width threshold for system detection")
@click.option("-o", "--output", help="Output file pattern")
@click.option("-v", "--verbose", is_flag=True, default=False, help="Print debug information")
@click.version_option(src.incipit.__version__)
@click.argument("input", type=click.Path(exists=True))
def main(audit, count, pages, systems, height_threshold, width_threshold, output, verbose, input):
    """
    Extract the first (or any) system of staves from a black-and-white modern
    musical score, either available as an image file or PDF. For instance,

        $ incipit -o "555-incipit.png" ./input/pdf/555-gilbert.pdf

    will extract the first system of staves of the sheet music `555-gilbert.pdf'
    as the image `555-incipit.png`. And,

        $ incipit --output "1__{}.png" -# '0,-1' ./input/pdf/1-gilbert.pdf

    will extract the first and last systems of staves of `1-gilbert.pdf` and output
    these as the images `1__0.png' (first system) and `1__10.png' (last system).
    """

    if not verbose:
        loguru.logger.remove()

    loguru.logger.debug(
        (
            "CLI called with: output={output}, audit={audit}, pages={pages}, "
            "systems={systems}, count={count}, input={input},"
            "width_treshold={width_threshold}, height_threshold={height_threshold}"
        ),
        output=output, audit=audit, pages=pages, systems=systems, count=count, input=input,
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

        staff_audit_filenames = src.incipit.heuristics.write_staff_detection_images_from_input_document(
            input_path=input,
            output_filename_pattern=output,
            width_proportion=width_threshold,
            height_proportion=height_threshold,
            page_numbers_to_keep=pages,
        )

        loguru.logger.info("audit yielded following files: {}", ", ".join(staff_audit_filenames))

        return

    if systems is not None:
        systems = list(map(int, systems.split(",")))
        loguru.logger.debug(
            "requested systems={systems}",
            systems=systems,
        )

    loguru.logger.debug(
        "requested pages={pages}; systems={systems}",
        pages=pages, systems=systems,
    )

    totalError = 0
    if count and output is None and systems is None:
        loguru.logger.debug("COUNT method activated")
        #for i in range(14, 252):
        for i in range(1, 559):
          input =   "C:\\Users\\user\\Google Drive\\projects\\python\\pdf2\\page" + str(i).zfill(3) + ".pdf"
          input = "C:\\Users\\user\\Google Drive\\projects\\python\\pdf\\" + str(i) + ".pdf"
          systems = src.incipit.heuristics.detect_staves_from_input_document(
            input_path=input,
            width_proportion=width_threshold,
            height_proportion=height_threshold,
            page_numbers_to_keep=pages,
          )
          #if( len(systems) - systemNumberInPDFList[i-1] != 0 ):
          print(str(len(systems)) + "\t" + str(len(systems) - systemNumberInPDFList[i-1]) + "\t" + input )
          #else:
          #  print("matched " + str(i) + ".pdf")
          totalError += abs(len(systems) - systemNumberInPDFList[i-1])
        print(" total error " + str(totalError))
        return


    if output is None:
        basename, _ = os.path.splitext(os.path.basename(input))
        output = "{}_system_{{:02}}.png".format(basename)
        loguru.logger.debug("using '{}' as output pattern", output)

    extracted_systems = src.incipit.heuristics.extract_staves_from_input_document(
        input_path=input,
        output_filename_pattern=output,
        staves_to_keep=systems,
        width_proportion=width_threshold,
        height_proportion=height_threshold,
        page_numbers_to_keep=pages,
    )

    extracted_systems_filenames = list(map(lambda x: x[0], extracted_systems))
    loguru.logger.debug("Extracted staves: {}", ",".join(extracted_systems_filenames))

    if count:
        print(len(extracted_systems_filenames))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        loguru.logger.debug("KeyboardInterrupted happened; caught gracefully...")
        pass


