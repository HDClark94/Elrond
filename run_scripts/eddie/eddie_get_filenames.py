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

of_paths = [data_path + "of/" + s for s in os.listdir(data_path + "of/") if str(mouse)+"_D"+str(day) in s]
vr_paths = [data_path + "vr/" + s for s in os.listdir(data_path + "vr/") if str(mouse)+"_D"+str(day) in s] 
vr_multi_context_paths = [data_path + "vr_multi_context/" + s for s in os.listdir(data_path + "vr_multi_context/") if str(mouse)+"_D"+str(day) in s]
abo_paths = [data_path + "allen_brain_observatory_visual_coding/" + s for s in os.listdir(data_path + "allen_brain_observatory_visual_coding/") if str(mouse)+"_D"+str(day) in s]
vs_paths = [data_path + "allen_brain_observatory_visual_sequences/" + s for s in os.listdir(data_path + "allen_brain_observatory_visual_sequences/") if str(mouse)+"_D"+str(day) in s]
vs2_paths = [data_path + "allen_brain_observatory_visual_multi_sequences/" + s for s in os.listdir(data_path + "allen_brain_observatory_visual_multi_sequences/") if str(mouse)+"_D"+str(day) in s]
dvd_paths = [data_path + "dvd_waitscreen/" + s for s in os.listdir(data_path + "dvd_waitscreen/") if str(mouse)+"_D"+str(day) in s]


recording_paths = of_paths + vr_paths + vr_multi_context_paths + abo_paths + vs_paths + vs2_paths + dvd_paths
data_path_on_eddie = project_path + "data/M" + mouse + "_D" + day + "/"
Path(data_path_on_eddie).mkdir(exist_ok=True)

with open(data_path_on_eddie + "data_folder_names.txt", "w") as f:
    for recording_path in recording_paths:
        f.write(recording_path + "\n")
