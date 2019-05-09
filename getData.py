import sys
import os
import glob
#test

def list_dir(verbose = True):
    '''
    function to list the contents of current working directory
    return:: list of branches
	     dict of current working directory {idx : branch}
    '''
    idx = []
    contents = []
    cwd = os.getcwd()
    tree = os.listdir(cwd)
    for i,j in enumerate(tree):
        idx.append(i)
        contents.append(j)
    if verbose:
        print("Working Dir: {}".format(cwd))
        print("Returning Contents of Working Directory..")
    return contents, dict(zip(idx, contents))

def fetch_data_path (folder = 'data'):
    '''
    function to string concat holistic path to data files w/ user input
    arguments:
        pass in string of data folder
    return :: string of concatenated path to data file
    
    '''
    cwd = os.getcwd()
    path = cwd + "\\" + folder
    print("Choose a file from data directory:")
    for idx, pat in enumerate(os.listdir(path)):
        print("{}) {}".format(idx, pat))
    i = input("Enter Number: ")
    try:
        if 0 <= int(i) <= len(os.listdir(path)):
            dataPath = os.listdir(path)[int(i)]
            print("Path to Data Stored: {}". format(path + "\\" + dataPath))
            return path + "\\" + dataPath
    except:
        print("Invalid Selection")
return None