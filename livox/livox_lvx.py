# Prof. Jason's attempt to read an lvx file using the lvx library.
# pip install livox-lvx

# from lvx import LvxFileReader, LvxFileWriter

# from lvx import clean_file
# from lvx import diff

from read_livox import *

INPUT = "/home/edonson/sokil_code/2025-09-08 11-05-53.lvx"

#clean_file(INPUT, "/home/edonson/sokil_code/livox/2025-09-03 11-21-00_clean.lvx")
#diff("/home/edonson/sokil_code/Test1-2R.lvx", 'output.diff', header_only=False)
#INPUT = '/home/edonson/sokil_code/Test1-2R.lvx'

with open(INPUT, 'rb') as fi, open(INPUT + '.other', 'wb') as fo:
    lvx_in = LvxFileReader(fi)
    LvxFileReader.make_csv(fi, "test", "test.csv", lvx_in, 2)
    # header = lvx_in.header
    # for it, i in enumerate(lvx_in):
    #     print(i)
    #     #print(i.packages)

        
