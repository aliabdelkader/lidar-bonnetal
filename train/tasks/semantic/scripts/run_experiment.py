import papermill as pm
import os
import subprocess
from datetime import datetime
import yaml
import sys

    
def commit_exp(experiment_parameters, commit_message, tag):
    experiment_name = experiment_parameters['experiment_name']
    log_dir = experiment_parameters["log"]
    subprocess.check_output(f"git add {log_dir}", shell=True)
    subprocess.check_output(f"git commit -m \"{commit_message}\"", shell=True)
    if tag:
        experiment_version = experiment_parameters['experiment_version']
        tag_name = f"{experiment_name}_{experiment_version}"
        subprocess.call(f"git tag -d {tag_name}", shell=True)
        subprocess.check_output(f"git tag {tag_name}", shell=True)
        
def run_experiment(template_path, experiment_parameters):
    # create experiment directory
    os.makedirs(experiment_parameters["log"], exist_ok=True)

    # set output notebook path
    output_notebook = os.path.join(experiment_parameters["log"], "run.ipynb")
    
    # prepare notebook
    pm.execute_notebook(
        template_path,
       output_notebook,
       parameters=experiment_parameters,
        prepare_only=True
    )
    
    #convert output notebook to html
    subprocess.check_output(['jupyter', 'nbconvert', output_notebook])

    # commit start of experiment
    commit_message_config =  experiment_parameters["commit_message"]
    
    commit_exp(experiment_parameters=experiment_parameters, 
               commit_message= f"START: {commit_message_config}", 
               tag=False)
    
    pm.execute_notebook(
        template_path,
       output_notebook,
       parameters=experiment_parameters,
       log_output=True,
        autosave_cell_every=5,
        stdout_file=sys.stdout,
        stderr_file=sys.stderr
    )
    
    #convert output notebook to html
    subprocess.check_output(['jupyter', 'nbconvert', output_notebook])
 
    # commit end of experiment
    commit_exp(experiment_parameters=experiment_parameters, 
               commit_message= f"END: {commit_message_config}", 
               tag=True)

if __name__ == '__main__':

    assert len(sys.argv) >=  3, "please provide path to run config, run template notebook"
    template_path = sys.argv[1]
    config_path = sys.argv[2]

    run_config = yaml.safe_load(open(config_path, 'r'))
    print(run_config)

    
    time_str = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    
    run_config['log'] = run_config['log'] + '/' + time_str

    run_experiment(template_path, run_config)

