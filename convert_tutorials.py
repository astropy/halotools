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
    print("changing line %s" %line)
    line = lines[1]
    line = line.replace('-','_')
    line = line.replace('\n', ':')
    line = '.. _'+line
    print("line is now %s" % line)
    lines[1] = line
    lines.insert(2, '\n')
    #lines[1] = lines[1]+'\n'
    f = open(fname,'w')
    for line in lines:
        f.write(line)
    f.close()


def add_asterisk_header(fname):
    f = open(fname, 'r')
    lines = f.readlines()
    f.close()
    title = lines[3]
    print("Title = %s" %title)
    asterisks = get_asterisks_line(title)
    lines.insert(4, asterisks)
    lines.insert(3, asterisks)
    f = open(fname,'w')
    for line in lines:
        f.write(line)
    f.close()

def get_list_of_tutorials(relative_dirname):
    tutorial_tag = '.ipynb'
    tutorial_list = (
        [f for f in listdir(relative_dirname) if 
        isfile(join(relative_dirname,f)) & 
        (f[-len(tutorial_tag):] == tutorial_tag)]
        )
    return tutorial_list

def file_prepend_line(filename, line_to_prepend):
    f = fileinput.input(filename, inplace=1)
    for xline in f:
        if f.isfirstline():
            print line_to_prepend.rstrip('\r\n') + '\n' + xline,
        else:
            print xline,

def get_asterisks_line(header):
    asterisks=''
    for ii in range(len(header)):
        asterisks+='*'
    return asterisks+'\n'

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
    if enforce_pass is True:
        assert err==''

    # The script version of the .ipynb file 
    # is no longer necessary, so delete it
    system("rm -rf "+fname+".py")

    if err == '':
        return 'pass'
    else:
        print("error msg = \n"+err)
        return 'fail'


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
            fname_test = test_ipynb(fname, enforce_pass=False)

            if fname_test=='pass':
                # convert the notebook to rst for inclusion in the docs
                conversion_string = "ipython nbconvert --to rst "+fname
                c=system(conversion_string)

                rst_fname = fname[:-len('.ipynb')]+'.rst'
                add_asterisk_header(rst_fname)
                rewrite_first_line(rst_fname)
                system("rm "+fname)
                system("mv "+rst_fname+" "+tutorial_loc)

            else:
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
                print(failure+".ipynb\n")
            print("\n")

            
########################################################




############################
### Trigger
############################
if __name__ == '__main__':
    main()
