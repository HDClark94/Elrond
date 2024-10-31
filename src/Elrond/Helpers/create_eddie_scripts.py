import subprocess
from datetime import datetime
from pathlib import Path
import time

def make_run_python_script(python_arg, venv=None, cores=None, email=None, h_rt=None, h_vmem=None, hold_jid=None, job_name=None, staging=False):
    """
    Makes a python script, which will run
    >>  python python_arg

    If nothing else is supplied, this will run on the venv 'elrond' with 8 cores, 19GB of RAM per core, with a 
    hard runtime limit of 48 hours.    
    """

    if hold_jid is not None:
        hold_script = f" -hold_jid {hold_jid}"
    else:
        hold_script = ""
    if email is not None:
        email_script = f" -M {email} -m e"
    else:
        email_script = ""
    if venv is None:
        venv = "elrond"
    
    
    if cores is None:
        cores = 8

    if h_rt is None:
        h_rt = "47:59:59"
    if h_vmem is None:
        h_vmem=19
    if job_name is not None:
        name_script = f" -N {job_name}"
    else:
        name_script = ""
    if staging:
        staging_script = " -q staging"
        core_script = ""
        vmem_script = ""
    else:
        staging_script = ""
        core_script = f" -pe sharedmem {cores}"
        vmem_script = f",h_vmem={h_vmem}G"

    script_content = f"""#!/bin/bash
#$ -cwd{staging_script}{core_script} -l rl9=true{vmem_script},h_rt={h_rt}{hold_script}{email_script}{name_script}
source /etc/profile.d/modules.sh
module load anaconda
conda activate {venv}
python {python_arg}"""
    
    return script_content

def save_and_run_script(script_content, script_file_path):

    if script_file_path is None:
        script_file_path = f"run_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".sh"

    f = open(script_file_path, "w")
    f.write(script_content)
    f.close()

    compute_string = "qsub " + script_file_path
    subprocess.run( compute_string.split() )

def run_python_script(python_arg, venv=None, cores=None, email=None, h_rt=None, h_vmem=None, hold_jid=None, script_file_path=None, staging=False):

    script_content = make_run_python_script(python_arg, venv=venv, cores=cores, email=email, h_rt=h_rt, h_vmem=h_vmem, hold_jid=hold_jid, staging=staging)
    save_and_run_script(script_content, f"run_python_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".sh")

    return

def run_stageout_script(stageout_dict, script_file_path=None):
    """
    makes a stage out script from a stageout_dict of the form
    {'path/to/file/on/eddie': 'path/to/destination/on/datastore'}
    Note: let's NEVER stageout to the raw data file, to avoid risk of deletion
    """

    script_text="""#!/bin/sh
#$ -cwd
#$ -q staging
#$ -l h_rt=00:29:59"""

    for source, dest in stageout_dict.items():
        script_text = script_text + "\nrsync -r " + str(source) + " " + str(dest)

    save_and_run_script(script_text, script_file_path)

    return 

def run_stagein_script(stagein_dict, script_file_path=None, job_name = None):
    """
    makes a stage out script from a stageout_dict of the form
    {'path/to/file/on/datastore': 'path/to/destination/on/eddie'}
    """



    script_text="""#!/bin/sh
#$ -cwd
#$ -q staging
#$ -l h_rt=00:59:59"""

    if job_name is not None:
        script_text += "#$ -N " + job_name + "\n" 

    for source, dest in stagein_dict.items():
        script_text = script_text + "\nrsync -r " + str(source) + " " + str(dest)

    save_and_run_script(script_text, script_file_path)

    return 

def stagein_data(mouse, day, project_path, job_name=None):

    filenames_path = project_path + f"data/M{mouse}_D{day}/data_folder_names.txt"

    if Path(filenames_path).exists() is False:
        get_filepaths_on_datastore(mouse, day, project_path)

    while Path(filenames_path).exists() is False:
        time.sleep(5)

    with open(filenames_path) as f:
        paths_on_datastore = f.read().splitlines()

    folder_names = [path_on_datastore.split('/')[-1] for path_on_datastore in paths_on_datastore]
    dest_on_eddie = [project_path + f"data/M{mouse}_D{day}/" + folder_name for folder_name in folder_names]

    stagein_dict = dict(zip(paths_on_datastore, dest_on_eddie))

    run_stagein_script(stagein_dict, job_name)

    return

def get_filepaths_on_datastore(mouse, day, project_path):

    import Elrond
    elrond_path = Elrond.__path__[0]
    run_python_script(python_arg = f"{elrond_path}/../../run_scripts/eddie/eddie_get_filenames.py {mouse} {day} {project_path}", venv="elrond", staging=True, h_rt="0:29:59", cores=1)
    return 
