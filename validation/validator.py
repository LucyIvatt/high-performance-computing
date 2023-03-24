import sys

TOLERANCE = 0.02

if len(sys.argv) != 3:
    print("Usage: python3 validator.py <vtk_file_1_path> <vtk_file_2_path")
    sys.exit(1)
else:
    VTK_1 = sys.argv[1]
    VTK_2 = sys.argv[2]

def compare_lines(l1, l2):
    num_diff = 0

    print(l1)
    print(l2)

    l1 = [float(num) for num in l1.rstrip().split(" ")]
    l2 = [float(num) for num in l2.rstrip().split(" ")]

    for i in range(len(l1)):
        if l1[i] != l2[i]:
            num_diff += 1 

    print(num_diff, "/", len(l1))
    return num_diff, len(l1)


def compare_files():
    total_diff, total_values = 0, 0
    with open(VTK_1) as file1:
        with open(VTK_2) as file2:
            line_count = 0
            for line1, line2 in zip(file1, file2):
                line_count += 1
                if line_count > 15:
                    diff, total = compare_lines(line1, line2)
                    total_diff += diff
                    total_values += total
    print("Identical Values = ", str(total_diff), "/", str(total_values))

compare_files()
