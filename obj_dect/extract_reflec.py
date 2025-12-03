import numpy as np

def read_pcd_with_reflectivity(filename):
    """
    Takes .pcd file collected by Livox Avia saved at filename and extracts reflectivity and xyz-postion,
    returning a dictionary with keys "positions" and "intensity" and values of the list of xyz-positions
    and reflectivity values from the .pcd file respectively.
    
    Param:
    filename : String
        Relative Filepath to .pcd file collected by Livox Avia containing xyz-position and reflectivity data
    """

    with open(filename, 'rb') as f:
        header = []
        while True:
            line = f.readline().decode('utf-8').strip()
            header.append(line)
            if line.startswith('DATA'):
                data_header = line
                break

        # Parse header for field info
        fields = None
        for line in header:
            if line.startswith('FIELDS'):
                fields = line.split()[1:]
            if line.startswith('POINTS'):
                num_points = int(line.split()[1])

        # Check data format
        binary = 'binary' in data_header.lower()

        if binary:
            # Use numpy to read after header
            offset = f.tell()
            raw = np.fromfile(f, dtype=np.float32)
            raw = raw.reshape(-1, len(fields))
        else:
            raw = np.loadtxt(filename, skiprows=len(header))

    data = {field: raw[:, i] for i, field in enumerate(fields)}

    return data
