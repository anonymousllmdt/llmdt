from utils import logger
from loss.loss import loss
from .trainer import Trainer

def _loss_factory(cfg, network):
    loss_cfg = cfg.loss_cfg
    logger.info(f'Making network loss @ {loss_cfg.name}')
    network_loss = loss.get(loss_cfg.name)(network, cfg, loss_cfg)
    return network_loss

def make_trainer(cfg, network):
    logger.info(f"Making trainer @ {cfg.trainer.name}")
    network = _loss_factory(cfg, network)
    return Trainer(network, cfg)
