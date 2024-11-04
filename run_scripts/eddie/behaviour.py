from run_pipeline import do_behavioural_postprocessing
import sys

mouse = sys.argv[1]
day = sys.argv[2]
sorter_name = sys.argv[3]
project_path = sys.argv[4]

do_behavioural_postprocessing(mouse, day, sorter_name, project_path)