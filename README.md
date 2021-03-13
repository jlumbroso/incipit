# `incipit` — extract systems of staves from musical scores

This command-line tool (and Python package) allows for the easy extraction of
systems of staves from musical scores. In particular, it makes it easy to extract
the first system of any musical score that is in B&W format, and has a noiseless
enough background

This tool was built to automatically generate incipits for large sets of scores,
like those downloaded from IMSLP. It was specifically built to create an index
of the 555 keyboard sonatas by Domenico Scarlatti, in a related project.

## Installation

The package is available on PyPI as `incipit` and so is available the usual way, i.e.,
```
sudo pip install incipit
```
In addition to the Python package, this should also install a CLI binary that is
runnable, called `incipit`.

## Usage

```
Usage: incipit [OPTIONS] INPUT

  Extract the first (or any) system of staves from a black-and-white modern
  musical score, either available as an image file or PDF. For instance,

      $ incipit -o "555-incipit.png" ./input/pdf/555-gilbert.pdf

  will extract the first system of staves of the sheet music
  `555-gilbert.pdf' as the image `555-incipit.png`. And,

      $ incipit --output "1__{}.png" -# '0,-1' ./input/pdf/1-gilbert.pdf

  will extract the first and last systems of staves of `1-gilbert.pdf` and
  output these as the images `1__0.png' (first system) and `1__10.png' (last
  system).

Options:
  -a, --audit                   Visualize system detection across document
  -c, --count                   Output number of detected systems
  -p, --pages TEXT              List of pages to process (e.g., '0', '0,-1')
  -#, --systems TEXT            List of systems to extract (e.g., '0', '0,-1')
  -h, --height-threshold FLOAT  % of height threshold for system detection
  -w, --width-threshold FLOAT   % of width threshold for system detection
  -o, --output TEXT             Output file pattern
  -v, --verbose                 Print debug information
  --version                     Show the version and exit.
  --help                        Show this message and exit.
```