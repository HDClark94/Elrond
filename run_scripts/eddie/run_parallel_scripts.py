import subprocess
import sys

mouse = sys.argv[1]
day = sys.argv[2]
sorter_name = sys.argv[3]

script_file_names = [
    f"stagein_M{mouse}_D{day}_0", 
    f"stagein_M{mouse}_D{day}_1", 
    f"stagein_M{mouse}_D{day}_2",
    f"M{mouse}_{day}_z_{sorter_name}.sh",
    f"M{mouse}_{day}_{sorter_name}.sh",
    f"M{mouse}_{day}_sspp_{sorter_name}.sh",
    f"M{mouse}_{day}_sspp_{sorter_name}.sh",
    f"M{mouse}_{day}_1dlc.sh",
    f"M{mouse}_{day}_2dlc.sh",
    f"M{mouse}_{day}_behave.sh"
]

for script_file_name in script_file_names:
    compute_string = "qsub " + script_file_name
    subprocess.run( compute_string.split() )