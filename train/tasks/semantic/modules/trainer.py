#!/usr/bin/env python3
from tasks.semantic.modules.BaseTrainer import BaseTrainer

class Trainer(BaseTrainer):
  def __init__(self, ARCH, DATA, datadir, logdir, path=None):
    super(Trainer, self).__init__( ARCH, DATA, datadir, logdir, path)
    (self.create_seed()
      .create_logger()
      .create_parser()
      .create_model()
      .move_model_to_gpu()
      .create_loss_fn()
      .create_optimzer_parameter_groups()
      .create_optimizer()
      .create_lr_schedular())


  def train(self):
    return self._train()
  
  def validate(self):
    return self._validate()
