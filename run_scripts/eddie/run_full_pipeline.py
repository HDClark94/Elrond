import sys
import os
from Elrond.Helpers.create_eddie_scripts import stagein_data, run_python_script, run_stageout_script
from pathlib import Path

mouse = sys.argv[1]
day = sys.argv[2]
sorter_name = sys.argv[3]
project_path = sys.argv[4]

import Elrond
elrond_path = Elrond.__path__[0]

data_path = project_path + f"data/M{mouse}_D{day}"
Path(data_path).mkdir(exist_ok=True)

# check if raw recordings are on eddie. If not, stage them
stagein_name = None
if len(os.listdir(data_path)) < 3:
    stagein_name = f"stagein_M{mouse}_D{day}"
    stagein_data(mouse, day, project_path, job_name = stagein_name)

# Now run full pipeline on eddie
run_python_script(elrond_path + "/../../run_scripts/eddie/run_pipeline_on_data.py " + mouse + " " + day + " " + sorter_name + " " + project_path, hold_jid=stagein_name)

run_stageout_script({
    project_path + "derivatives/M"+mouse+"/D"+day+"/": "/exports/cmvm/datastore/sbms/groups/CDBS_SIDB_storage/NolanLab/ActiveProjects/Chris/Cohort12/derivatives/M"+mouse+"/D"+day+"/"
    })
