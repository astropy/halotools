#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from os import system
from subprocess import PIPE, Popen
import fileinput


def file_prepend_line(filename, line_to_prepend):
    f = fileinput.input(filename, inplace=1)
    for xline in f:
        if f.isfirstline():
            print(line_to_prepend.rstrip('\r\n') + '\n' + xline,)
        else:
            print(xline,)


def get_asterisks_line(header):
    asterisks = ''
    for ii in range(len(header)):
        asterisks += '*'
    return asterisks


def test_ipynb(fname, enforce_pass=True):
    """Function to use in a test suite to
    verify that an IPython Notebook
    executes without raising an exception.

    The method is to convert the Notebook to a python script,
    and then assert that the script does not raise an exception
    when run as an executable. Useful for
    incorporating IPython Notebooks into documentation.

    Parameters
    ----------
    fname : string
        Name of IPython Notebook being tested.

    Notes
    -----
    Requires pandoc to be installed so that
    the Notebook can be converted to an executable python script.

    Do not include the file extension in fname.
    If '.ipynb' is accidentally included, the function will strip
    the extension and otherwise behave properly.

    Credit to Padriac Cunningham for the idea to spawn a subprocess
    to handle the exception handling in an external script.

    """

    if fname[-6:] == '.ipynb':
        fname = fname[0:-6]

#   convert the notebook to a python script
#   conversion_string = "ipython nbconvert --to python "+fname+".ipynb"
    conversion_string = "ipython nbconvert --to python " + fname
    c = system(conversion_string)

#   Use subprocess.Popen to spawn a subprocess
#   that executes the tutorial script
    s = Popen(["python", fname+".py"], stderr=PIPE)
#   After the following line,
#   err will be an empty string if the program runs without
#   raising any exceptions
    _, err = s.communicate()
    if enforce_pass is True:
        assert err == ''

    # The script version of the .ipynb file
    # is no longer necessary, so delete it
    system("rm -rf " + fname + ".py")

    if err == '':
        return 'pass'
    else:
        print("error msg = \n" + err)
        return 'fail'


########################################################
########### MAIN ###########

def main():
    tutorial_dict = {'example_tutorial':'Example IPython Notebook Tutorial'}
    tutorial_list = tutorial_dict.keys()

    failure_list = []
    for fname in tutorial_list:
        # Check to see whether this notebook raises an exception
        fname_test = test_ipynb(fname, enforce_pass=False)

        if fname_test == 'pass':
            # convert the notebook to rst for inclusion in the docs
            conversion_string = "ipython nbconvert --to rst "+fname
            c = system(conversion_string)
            header_line = get_asterisks_line(tutorial_dict[fname])
            file_prepend_line(fname+'.rst', header_line)
            file_prepend_line(fname+'.rst', tutorial_dict[fname])
            file_prepend_line(fname+'.rst', header_line)
        else:
            failure_list.append(fname)

    if failure_list == []:
        print("\n")
        print("Each notebook executes without raising an exception")
        print("\n")
    else:
        print("\n")
        print("The following notebooks were not converted to rst "
            "because they raise an exception:")
        for failure in failure_list:
            print(failure+".ipynb\n")
        print("\n")


########################################################

############################
### Trigger
############################
if __name__ == '__main__':
    main()
