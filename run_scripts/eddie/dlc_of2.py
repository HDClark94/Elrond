from ..run_pipeline import do_dlc_pipeline
import sys

mouse = sys.argv[1]
day = sys.argv[2]
sorter_name = sys.argv[3]
project_path = sys.argv[4]

do_dlc_pipeline(mouse, day, dlc_of_model_path = project_path + "derivatives/dlc_models/of_cohort12-krs-2024-10-30/", do_of1=False, do_vr=False)
