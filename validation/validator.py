import sys
import time
from itertools import zip

if len(sys.argv) != 3:
    print("Usage: python validator.py <vtk_file_1_path> <vtk_file_2_path")
    sys.exit(1)

vtk_1 = sys.argv[1]
vtk_2 = sys.argv[2]

with open(vtk_1) as file1:
    with open(vtk_2) as file2:
        for line1, line2 in zip(file1, file2):
            print(line1)
            print(line2)
            print("-------------")
            time.sleep(2)
