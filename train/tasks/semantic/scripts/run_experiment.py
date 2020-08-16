import papermill as pm
import os
import subprocess
from datetime import datetime
import yaml
import sys

def run_experiment(template_path, experiment_parameters):
    # create experiment directory
    os.makedirs(experiment_parameters["log"], exist_ok=True)

    # set output notebook path
    output_notebook = os.path.join(experiment_parameters["log"], "run.ipynb")
    
    # run notebook
    pm.execute_notebook(
        template_path,
       output_notebook,
       parameters=experiment_parameters,
       log_output=True,
        autosave_cell_every=5,
        stdout_file=sys.stdout,
        stderr_file=sys.stderr
    )
    
    #optional: convert output notebook to html
    subprocess.check_output(['jupyter', 'nbconvert', output_notebook])


if __name__ == '__main__':

    assert len(sys.argv) >=  3, "please provide path to run config, run template notebook"
    template_path = sys.argv[1]
    config_path = sys.argv[2]

    run_config = yaml.safe_load(open(config_path, 'r'))
    print(run_config)

    
    time_str = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    
    run_config['log'] = run_config['log'] + '/' + time_str

    run_experiment(template_path, run_config)

