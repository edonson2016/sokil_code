# Prof. Jason's attempt to read a .lvx file with openpylivox.
# pip install git+https://github.com/ryan-brazeal-ufl/openpylivox.git
# note that this got forked and became https://github.com/Livox-SDK/openpylivox which seems behind.

import openpylivox as opl

# Open the LVX file
filepath = "../Test1-2R.lvx"
with opl.open_lvx_file(filepath) as f:

    # Iterate over frames
    for frame in f:
        # Access point cloud data
        points = frame.points() 

        # Do something with the points
        print(points)
        