import sys
import re

TOLERANCE = 0.02

def is_num_str(string):
    pattern = r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$'
    return bool(re.match(pattern, string))

def compare_lines(l1, l2, stats_dict):
    l1 = [val for val in l1.rstrip().split(" ")]
    l2 = [val for val in l2.rstrip().split(" ")]

    for i in range(len(l1)):
        # If the strings are numeric then compare them and add to totals 
        if is_num_str(l1[i]) and is_num_str(l2[i]):
            num1, num2 = float(l1[i]), float(l2[i])
            if num1 == num2:
                stats_dict["EXACT"] += 1
            elif abs(num1 - num2) <= TOLERANCE:
                stats_dict["CLOSE"] += 1
            else:
                stats_dict["WRONG"] += 1
            stats_dict["TOTAL"] += 1
        
        # If one of the files contains a number in this position and the other doesn't, raise an error.
        elif (is_num_str(l1[i]) and not is_num_str(l2[i])) or (is_num_str(l2[i]) and not is_num_str(l1[i])):
            raise ValueError("Files do not contain the same number of values, may be from running Vortex with different input parameters.")

    return stats_dict


def compare_files(f1, f2):
    stats_dict = {"WRONG":0, "CLOSE":0, "EXACT":0, "TOTAL":0}

    with open(f1) as file1:
        with open(f2) as file2:
            for line1, line2 in zip(file1, file2):
                compare_lines(line1, line2, stats_dict)

    print("-----------------------------------------------")
    print(f"Comparing original implementation ({VTK_1}) to parallel implementation ({VTK_2}):\n")
    for label in ["WRONG", "CLOSE", "EXACT"]:
        print(f"{label}: {stats_dict[label]}/{stats_dict['TOTAL']} - {100.0 * stats_dict[label]/stats_dict['TOTAL']:.4f}%")
    print(f"\nNote: Close values are determined using a tolerance value of {TOLERANCE}. Percentages are calculated to 4 decimal places.")
    print("-----------------------------------------------")

if len(sys.argv) == 3:
    VTK_1 = sys.argv[1]
    VTK_2 = sys.argv[2]
else:
    print("Usage: python3 validator.py <vtk_file_path_original> <vtk_file_path_parallel>")
    sys.exit(1)

compare_files(VTK_1, VTK_2)
