#! /usr/bin/env python

# Write EMAN2 box files

import csv
import numpy as np

def write_box(path, image, boxes,boxsize):
    with open(path, 'wb') as boxfile:
        boxwriter = csv.writer(boxfile, delimiter='\t',
                           quotechar='|', quoting=csv.QUOTE_NONE)
        for box in boxes:
            x_ll = np.int((box.x - boxsize / 2) * image.shape[1]) # lower left
            y_ll = np.int((box.y - boxsize / 2) * image.shape[0]) # lower right
            boxsize = np.int(boxsize * image.shape[1])
            boxwriter.writerow([x_ll,y_ll,boxsize,boxsize])
