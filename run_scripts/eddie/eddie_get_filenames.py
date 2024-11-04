import os
import sys
from pathlib import Path

mouse = sys.argv[1]
day = sys.argv[2]
project_path = sys.argv[3]

if int(mouse) > 21:
    data_path = "/exports/cmvm/datastore/sbms/groups/CDBS_SIDB_storage/NolanLab/ActiveProjects/Harry/Cohort12_august2024/"
else:
    data_path = "/exports/cmvm/datastore/sbms/groups/CDBS_SIDB_storage/NolanLab/ActiveProjects/Harry/Cohort11_april2024/"


recording_paths = [data_path + "of/" + s for s in os.listdir(data_path + "of/") if str(mouse)+"_D"+str(day) in s] + [data_path + "vr/" + s for s in os.listdir(data_path + "vr/") if str(mouse)+"_D"+str(day) in s] 
data_path_on_eddie = project_path + "data/M" + mouse + "_D" + day + "/"
Path(data_path_on_eddie).mkdir(exist_ok=True)

with open(data_path_on_eddie + "data_folder_names.txt", "w") as f:
    for recording_path in recording_paths:
        f.write(recording_path + "\n")
