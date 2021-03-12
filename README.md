# `incipit` — extract staves from musical scores

This command-line tool (and Python package) allows for the easy extraction of
staves from musical scores. In particular, it makes it easy to extract the first
staff of any musical score that is in B&W format, and has a reasonable amount
of background noise.

This tool was built to automatically generate incipits for large sets of scores,
like those downloaded from IMSLP. It was specifically built to create an index
of sonatas by Domenico Scarlatti.

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

  Extract the first (or any) staff from a black-and-white modern musical
  score, either available as an image file or PDF. For instance,

      $ incipit -o "555-incipit.png" ./input/pdf/555-gilbert.pdf

  will extract the first staff of sheet music `555-gilbert.pdf' as the image
  `555-incipit.png`. And,

      $ incipit --output "1__{}.png" -# '0,-1' ./input/pdf/1-gilbert.pdf

  will extract the first and last staves of `1-gilbert.pdf` and output these
  as the images `1__0.png' (first staff) and `1__10.png' (last staff).

Options:
  -a, --audit                   Visualize staff detection across document
  -c, --count                   Output number of detected staves
  -p, --pages TEXT              List of pages to process (e.g., '0', '0,-1')
  -#, --staves TEXT             List of staves to extract (e.g., '0', '0,-1')
  -h, --height-threshold FLOAT  % of height threshold for staff detection
  -w, --width-threshold FLOAT   % of width threshold for staff detection
  -o, --output TEXT             Output file pattern
  -v, --verbose                 Print debug information
  --help                        Show this message and exit.
```