import optuna
import pandas as pd
from tasks.semantic.modules.trainer import Trainer
from tensorboard.plugins.hparams import api as hp
from optuna.integration.tensorboard import TensorBoardCallback
from pathlib import Path
class HPTrainer(Trainer):
    def __init__(self, ARCH, DATA, datadir, logdir, study_name, n_trials, path=None):
        super(HPTrainer, self).__init__(ARCH, DATA, datadir, logdir, path)
        self.log = logdir
        self.study = optuna.create_study(study_name=study_name, 
        storage='sqlite:///{}.db'.format(study_name + '_' + Path(logdir).name), 
        direction='maximize', load_if_exists=True)
        self.n_trials = n_trials

    def train(self):
        # hparams, metrics = self.define_experiment()
        self.create_logger()
        self.tensorboard_callback = TensorBoardCallback(self.log, metric_name="iou")
        self.study.optimize(self.objective, n_trials=self.n_trials, callbacks=[self.tensorboard_callback])
        df = self.study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
        df.to_csv(self.log + "/trials.csv")

    def objective(self, trial: optuna.Trial):
        
        self.ARCH["train"]["lr"] = trial.suggest_loguniform("learning_rate", 0.001, 0.1)
        self.ARCH["train"]["momentum"] = trial.suggest_float("momentum", 0.0, 1.0)
        self.ARCH["train"]["w_decay"] = trial.suggest_loguniform("w_decay", 0.0001 , 0.1)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        bn_d = trial.suggest_float("bn_d", 0.01, 0.1)

        if self.ARCH["backbone"]["train"] == True:
            self.ARCH["backbone"]["dropout"] = dropout
            self.ARCH["backbone"]["bn_d"] = bn_d

        if self.ARCH["decoder"]["train"] == True:
            self.ARCH["decoder"]["dropout"] = dropout
            self.ARCH["decoder"]["bn_d"] = bn_d

        if self.ARCH["head"]["train"] == True:
            self.ARCH["head"]["dropout"] = dropout

        (self
            .create_parser()
            .create_model()
            .move_model_to_gpu()
            .create_loss_fn()
            .create_optimzer_parameter_groups()
            .create_optimizer()
            .create_lr_schedular())
        
        best_val_iou = self._train()

        return best_val_iou