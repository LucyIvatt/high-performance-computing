import sys
import re
from math import sqrt

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

TOLERANCE = 0.2

def is_num_str(string):
    pattern = r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$'
    return bool(re.match(pattern, string))

def compare_lines(l1, l2, stats_dict, count):
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
                print(f"{num1}, {num2}, col_num={i}, line_num={count}")
            else:
                stats_dict["WRONG"] += 1
                print(f"{num1}, {num2}, col_num={i}, line_num={count}")

            stats_dict["TOTAL"] += 1
            stats_dict["DOT_PROD"] += num1 * num2
            stats_dict["SUM_SQUARES_A"] += num1 * num1
            stats_dict["SUM_SQUARES_B"] += num2 * num2

            
        
        # If one of the files contains a number in this position and the other doesn't, raise an error.
        elif (is_num_str(l1[i]) and not is_num_str(l2[i])) or (is_num_str(l2[i]) and not is_num_str(l1[i])):
            raise ValueError("Files do not contain the same number of values, may be from running Vortex with different input parameters.")

    return stats_dict


def compare_files(f1, f2):
    stats_dict = {"WRONG":0, "CLOSE":0, "EXACT":0, "TOTAL":0, "DOT_PROD":0, "SUM_SQUARES_A": 0, "SUM_SQUARES_B":0}

    with open(f1) as file1:
        with open(f2) as file2:
            count = 0
            for line1, line2 in zip(file1, file2):
                compare_lines(line1, line2, stats_dict, count)
                count += 1
    
    cosine_similarity = stats_dict["DOT_PROD"] / (sqrt(stats_dict["SUM_SQUARES_A"]) * sqrt(stats_dict["SUM_SQUARES_B"]))

    print("-----------------------------------------------")
    print(f"Comparing implementation ({VTK_1}) to parallel implementation ({VTK_2}):\n")
    for label in ["WRONG", "CLOSE", "EXACT"]:
        print(f"{label}: {stats_dict[label]}/{stats_dict['TOTAL']} - {100.0 * stats_dict[label]/stats_dict['TOTAL']:.4f}%")
    
    print(f"\nNote: Close values are determined using a tolerance value of {TOLERANCE}. Percentages are calculated to 4 decimal places.\n")

    print(f"Cosine Similarity: {cosine_similarity * 100}")

    if stats_dict["EXACT"] == stats_dict["TOTAL"]:
        print(f"{bcolors.OKGREEN}PASS: Both files are an exact match - successful parallel implementation. {bcolors.ENDC}")
    elif stats_dict["WRONG"] == 0:
        print(f"{bcolors.WARNING}WARNING: The file are similar but not an exact match - floating point error may be the cause.{bcolors.ENDC}")
    else:
        print(f"{bcolors.FAIL}ERROR: Files are not identical - check recent changes and retry. {bcolors.ENDC}")

    print("-----------------------------------------------")

if len(sys.argv) == 3:
    VTK_1 = sys.argv[1]
    VTK_2 = sys.argv[2]
else:
    print("Usage: python3 validator.py <vtk_file_path_original> <vtk_file_path_parallel>")
    sys.exit(1)

compare_files(VTK_1, VTK_2)
