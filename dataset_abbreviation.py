# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 15:40:36 2023, @author: Simon

Each datafile is specified by a tuple of numbers and/or strings, for instance: 
(year,month,day,sequence_number,shot_number).
These coordinates describe a tree structure: there can be several shots within 
a sequence, several sequences within a day, etc. When loading data, we often 
need to pick up certain branches of the tree, then load all the datafiles within.
To simplify the description of such a choice of branches, we choose to 
"""
import os
import numpy as np

def develop_abbreviation(abbreviation,start=()):
    """
    Examples:
        develop( [(2023,7,5,[2,6])] ) 
            -> [ (2023,7,5,2), (2023,7,5,6) ]
            
        develop( [(2023,7,range(5,8))] )
            -> [ (2023,7,5), (2023,7,6), (2023,7,8)]
            
        develop( [(2022,12,31),(2023,1,1)] ,root=('SeqBEC'))
            -> [ ('SeqBEC',2022,12,31), ('SeqBEC',2023,1,1) ]
            
        develop( [(2023,7,[5, (6,[1,2]) ])] )
            -> [ (2023,7,5), (2023,7,6,1), (2023,7,6,2) ]
    """
    list_of_coordinates = []
    for coordinates in abbreviation:
        if type(coordinates) is not tuple: 
            coordinates = (coordinates,)
        if type(coordinates[-1]) in [list,range,np.ndarray]:
            list_of_coordinates += develop_abbreviation(coordinates[-1],
                                              start = start + coordinates[:-1])
        else:
            list_of_coordinates += [start + coordinates]
    return list_of_coordinates

def all_filepaths_ending_with(extension,folder='.'):
    """ 
    Explores the content of the folder and returns the complete filepaths
    (folder\filename.extension) to the files with the given format extension.
    """
    filepaths,n = [],len(extension)
    for branch,_,files in os.walk(folder):
        for file in files:
            if file[-n:]==extension:
                filepaths.append(os.path.join(branch,file))
    return filepaths    

def folder_from_coordinates(root='',seq_name=None,year=None,month=None,
                            day=None,sequence=None):
    """ Specific to the experimental implementation. """
    coordinates = (root,)
    if seq_name is not None: coordinates += (seq_name,)
    if year is not None: coordinates += (f'{year}',)
    if month is not None: coordinates += (f'{month:02d}',)
    if day is not None: coordinates += (f'{day:02d}',)
    if sequence is not None: coordinates += (f'{sequence:04d}',)
    return os.path.join(*coordinates)
 
def filename_from_coordinates(root,seq_name,year,month,day,sequence,shot):
    """ Specific to the experimental implementation. """
    folder = folder_from_coordinates(root,seq_name,year,month,day,sequence)
    filename = f'{year}-{month:02d}-{day:02d}_{sequence:04d}_{seq_name}'
    nb_h5_files = len([x for x in os.listdir(folder) if x[-3:]=='.h5'])
    if nb_h5_files==1: fmt = '{:01d}'
    else: fmt = '{:0'+str(int(np.floor(np.log10(nb_h5_files-1))+1))+'d}'
    return filename + f'_{fmt.format(shot)}.h5'

def full_set_of_coordinates(coordinates):
    return type(coordinates) is tuple and len(coordinates)==6

def filepaths_from_coordinates(root,*coordinates):
    folder = folder_from_coordinates(root,*coordinates)
    if full_set_of_coordinates(coordinates):
        filename = filename_from_coordinates(root,*coordinates)
        return [os.path.join(folder,filename)]
    else:
        return all_filepaths_ending_with('.h5',folder)

def filepaths_from_abbreviation(abbreviation):
    list_of_coordinates,filepaths = develop_abbreviation(abbreviation), []
    for coordinates in list_of_coordinates:
        filepaths += filepaths_from_coordinates(coordinates)
    return filepaths