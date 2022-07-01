from __future__ import absolute_import, division, print_function, unicode_literals

from time import time
import sys
import urllib

__all__ = ["file_len", "download_file_from_url"]


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def download_file_from_url(url, fname):
    """Function to download a file from the web to a specific location,
    and print a progress bar along the way.

    Parameters
    ----------
    url : string
        web location of desired file, e.g.,
        ``http://www.some.website.com/somefile.txt``.

    fname : string
        Location and filename to store the downloaded file, e.g.,
        ``/Users/username/dirname/possibly_new_filename.txt``
    """
    start = time()
    print("\n... Downloading data from the following location: \n%s\n" % url)
    print(" ... Saving the data with the following filename: \n%s\n" % fname)

    def reporthook(blocks_thus_far, bytes_per_block, file_size_in_bytes):
        try:
            blocks_in_file = int(round(file_size_in_bytes / bytes_per_block))
            printout_condition = int(round(blocks_in_file / 20))
            if (blocks_thus_far % printout_condition == 0) & (blocks_thus_far > 0):
                frac_complete = blocks_thus_far / float(blocks_in_file)
                runtime = time() - start
                print(
                    "{0:.0f}% complete, elapsed time = {1:.0f} seconds".format(
                        frac_complete * 100, runtime
                    )
                )

        except ZeroDivisionError:
            pass
        sys.stdout.flush()

    urllib.request.urlretrieve(url, fname, reporthook)
