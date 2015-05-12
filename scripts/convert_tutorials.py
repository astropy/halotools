#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import system
from os import listdir
from os.path import isfile, join

from subprocess import PIPE,Popen
import fileinput

def rewrite_first_line(fname):
    f = open(fname, 'r')
    lines = f.readlines()
    f.close()
    line = lines[1]
    line = lines[1]
    line = line.replace('-','_')
    line = line.replace('\n', ':')
    line = '.. _'+line
    lines[1] = line
    lines.insert(2, '\n')
    f = open(fname,'w')
    for line in lines:
        f.write(line)
    f.close()


def add_asterisk_header(fname):

    def get_asterisks_line(header):
        asterisks=''
        for ii in range(len(header)):
            asterisks+='*'
        return asterisks+'\n'

    f = open(fname, 'r')
    lines = f.readlines()
    f.close()
    title = lines[3]
    asterisks = get_asterisks_line(title)
    lines.insert(4, asterisks)
    lines.insert(3, asterisks)
    f = open(fname,'w')
    for line in lines:
        f.write(line)
    f.close()

def correct_docs_hyperlinks(fname):

    def correct_dashes(line):
        line = line.replace('`--', '')
        line = line.replace('--`', '')
        return line

    def find_reflink_substrings(line):
        reflink_substr = ':ref:``'

        first_idx_list = []
        if reflink_substr in line:
            idx = line.find(reflink_substr)
            while idx > -1:
                first_idx_list.append(idx)
                idx = line.find(reflink_substr, idx+1)

        strlist = []
        for idx in first_idx_list:
            last_idx = line.find('``', idx + len(reflink_substr))
            strlist.append(line[idx:last_idx+2])

        return strlist

    def correct_reflinks(line):
        incorrect_substrings = find_reflink_substrings(line)
        if incorrect_substrings != []:
            for s in incorrect_substrings:
                correct_substr = s.replace('``', '`')
                line = line.replace(s, correct_substr)
        return line

    with open(fname, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        line = correct_dashes(line)
        line = correct_reflinks(line)
        lines[i] = line

    with open(fname, 'w') as f:
        for line in lines:
            f.write(line)


def get_list_of_tutorials(relative_dirname):
    tutorial_tag = '.ipynb'
    tutorial_list = (
        [f for f in listdir(relative_dirname) if 
        isfile(join(relative_dirname,f)) & 
        (f[-len(tutorial_tag):] == tutorial_tag)]
        )
    return tutorial_list


def test_ipynb(fname):
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

    if fname[-6:]=='.ipynb':
        fname = fname[0:-6]

#   convert the notebook to a python script
#   conversion_string = "ipython nbconvert --to python "+fname+".ipynb"
    conversion_string = "ipython nbconvert --to python "+fname
    c=system(conversion_string)

#   Use subprocess.Popen to spawn a subprocess 
#   that executes the tutorial script
    s = Popen(["python" ,fname+".py"],stderr=PIPE)
#   After the following line, 
#   err will be an empty string if the program runs without 
#   raising any exceptions
    _, err = s.communicate()  

    # The script version of the .ipynb file 
    # is no longer necessary, so delete it
    system("rm -rf "+fname+".py")

#    if err == '':
#        return 'pass'
#    else:
#        print("error msg = \n"+err)
#        return 'fail'
    return err


########################################################
########### MAIN ###########

def main():

    tutorial_dir_list = ['./docs/tutorials/']

    for tutorial_loc in tutorial_dir_list:

        tutorial_list = get_list_of_tutorials(tutorial_loc)

        failure_list = []
        for fname in tutorial_list:
            command = "cp "+tutorial_loc+fname+" ./"
            system(command)

            # Check to see whether this notebook raises an exception
            fname_test = test_ipynb(fname)

            if (fname_test=='') or ('FutureWarning' in fname_test):
                # convert the notebook to rst for inclusion in the docs
                conversion_string = "ipython nbconvert --to rst "+fname
                c=system(conversion_string)

                rst_fname = fname[:-len('.ipynb')]+'.rst'
                add_asterisk_header(rst_fname)
                rewrite_first_line(rst_fname)
                correct_docs_hyperlinks(rst_fname)
                system("rm "+fname)
                system("mv "+rst_fname+" "+tutorial_loc)

            else:
                print("error msg = %s " % fname_test )
                failure_list.append(fname)

        if failure_list==[]:
            print("\n")
            print("Each notebook executes without raising an exception")
            print("\n")
        else:
            print("\n")
            print("The following notebooks were not converted to rst "
                "because they raise an exception:")
            for failure in failure_list:
                print(failure+"\n")
            print("\n")

            
########################################################




############################
### Trigger
############################
if __name__ == '__main__':
    main()
