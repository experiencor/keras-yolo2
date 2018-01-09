#! /usr/bin/env python

# Write EMAN2 box files

import csv

def write_box(path, image, boxes,boxsize):
    boxwriter = csv.writer(path, delimiter='\t',
                           quotechar='|', quoting=csv.QUOTE_NONE)
    for box in boxes:
        x_ll = int((box.x - boxsize / 2) * image.shape[1]) # lower left
        y_ll = int((box.y - boxsize / 2) * image.shape[0]) # lower right
        boxwriter.writerow([x_ll,y_ll,boxsize,boxsize])
