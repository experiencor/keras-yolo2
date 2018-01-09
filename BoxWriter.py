#! /usr/bin/env python

# Write EMAN2 box files

import csv

def write_box(path, image, boxes,boxsize):
    with open(path, 'wb') as boxfile:
        boxwriter = csv.writer(boxfile, delimiter='\t',
                           quotechar='|', quoting=csv.QUOTE_NONE)
        for box in boxes:
            x_ll = int((box.x - boxsize / 2) * image.shape[1]) # lower left
            y_ll = int(image.shape[0]-(box.y)*image.shape[0] + (boxsize/2)*image.shape[0] ) # lower right
            export_boxsize = int(boxsize * image.shape[1])
            boxwriter.writerow([x_ll,y_ll,export_boxsize,export_boxsize])
