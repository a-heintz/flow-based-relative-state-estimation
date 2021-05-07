import random, argparse, math, os

import torch, torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import DataLoader

# TensorboardX
from tensorboardX import SummaryWriter

# Ignite
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage
from ignite.handlers import Checkpoint

from annealer import Annealer
from shepardmetzler import ShepardMetzler
from partition import partition
from argparse import ArgumentParser
import GPUtil

import config as c
from data import get_data
import util

from model_cond_flow_DensityVector import get_model

def train():
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    print(device)
    # Random seeding
    torch.manual_seed(99)
    if cuda: torch.cuda.manual_seed(99)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_dataset, valid_dataset, train_loader, valid_loader = get_data()

    model, optimizer = get_model(device)

    print(model.combined_model)

    parser = ArgumentParser(description='New Architecture')
    parser.add_argument('--resume_training', type=bool, default=False)
    parser.add_argument('--previously_saved_epoch', type=int, default=0)
    args = parser.parse_args()

    ''' iteration / 3334 '''

    ''' python train_cond_flow.py --resume_training=True --previously_saved_epoch=157'''

    # Rate annealing schemes
    mu_scheme = Annealer(5 * 10 ** (-6), 5 * 10 ** (-6), 1.6 * 10 ** 5)

    def run_iter(batch, mode):

        x, v, d = batch
        x, v, d = x.to(device), v.to(device), d.to(device)

        x, v, x_q, v_q, d_c, d_q = partition(x, v, d)

        # Log likelihood
        if mode == 'train':
            nll, z_loss = model.combined_model(x, v, x_q, v_q, d_c, d_q, False)
        elif mode == 'val':
            nll, z_loss = model.combined_model(x, v, x_q, v_q, d_c, d_q, True)

        #z_loss *= 1e2
        loss = nll + z_loss
        #loss = z_loss # + nll

        return loss, nll, z_loss


    def step(engine, batch):
        model.combined_model.train()
        #model.combined_model.glow.zero_grad()

        model.optimizer.zero_grad()

        loss, nll, z_loss = run_iter(batch, 'train')


        loss.backward()
        #util.clip_grad_norm(model.optimizer, 10)
        #torch.nn.utils.clip_grad_value_(model.combined_model.glow.parameters(), 5)
        #grad_norm = torch.nn.utils.clip_grad_norm_(model.combined_model.glow.parameters(), 100)

        model.optimizer.step()

        #with torch.no_grad():
            # Anneal learning rate
            #mu = next(mu_scheme)
            #i = engine.state.iteration
            #for group in model.optimizer.param_groups:
            #    group["lr"] = mu * math.sqrt(1 - 0.999 ** i) / (1 - 0.9 ** i)

        return {"loss":loss.item(), "nll": nll.item(), "z_loss": z_loss.item()}


    # Trainer and metrics
    trainer = Engine(step)



    @trainer.on(Events.ITERATION_COMPLETED)
    def log_metrics(engine):
        for key, value in engine.state.metrics.items():
            writer.add_scalar("training/{}".format(key), value, engine.state.iteration)

    '''
    @trainer.on(Events.EPOCH_COMPLETED)
    def save_images(engine):
        with torch.no_grad():
            x, v = engine.state.batch
            x, v = x.to(device), v.to(device)
            x, v, x_q, v_q = partition(x, v)

            x_mu, _, _ = model.combined_model(x, v, v_q)

            # Send to CPU
            x_mu = x_mu.detach().cpu().float()

            #writer.add_image("reconstruction", make_grid(x_mu), engine.state.epoch)
    '''
    @trainer.on(Events.EPOCH_COMPLETED)
    def validate(engine):
        model.combined_model.eval()
        with torch.no_grad():
            batch = next(iter(valid_loader))

            loss, nll, z_loss = run_iter(batch, 'val')

            writer.add_scalar("validation/loss", loss.item(), engine.state.epoch)
            #writer.add_scalar("validation/elbo", elbo.item(), engine.state.epoch)

    @trainer.on(Events.EXCEPTION_RAISED)
    def handle_exception(engine, e):
        writer.close()
        engine.terminate()
        if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
            import warnings
            warnings.warn('KeyboardInterrupt caught. Exiting gracefully.')
            checkpoint_handler(engine, { 'model_exception': model })
        else: raise e

    metric_names = ["loss", "nll", "z_loss"]
    RunningAverage(output_transform=lambda x: x["loss"]).attach(trainer, "loss")
    RunningAverage(output_transform=lambda x: x["nll"]).attach(trainer, "nll")
    RunningAverage(output_transform=lambda x: x["z_loss"]).attach(trainer, "z_loss")
    ProgressBar().attach(trainer, metric_names=metric_names)

    # Model checkpointing
    to_save = {'trainer': trainer, 'model': model.combined_model, 'optimizer': model.optimizer, 'mu_scheme': mu_scheme}
    checkpoint_handler = ModelCheckpoint(c.pretrained_path, "checkpoint", n_saved=3, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), checkpoint_handler, to_save=to_save) #(every=2)

    timer = Timer(average=True).attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # Tensorbard writer
    writer = SummaryWriter(log_dir=c.log_dir)

    if args.resume_training == True:
        checkpoints_dir = os.listdir(c.pretrained_path)
        checkpoints_dir = [x for x in checkpoints_dir if 'checkpoint_checkpoint_' in x]
        #print(checkpoints_dir)
        resume_epoch = args.previously_saved_epoch#len(checkpoints_dir)

        checkpoint_path = os.path.join(c.pretrained_path, checkpoints_dir[-1])

        to_load = {'trainer': trainer, 'model': model.combined_model, 'optimizer': model.optimizer,
                   'mu_scheme': mu_scheme}

        #print(checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint)

        trainer.state.iteration = (resume_epoch) * len(train_loader)
        trainer.state.epoch = (resume_epoch)

        print('Resuming Training at Epoch ', trainer.state.epoch, '... Iteration ', trainer.state.iteration)


    random.seed(99)

    trainer.run(train_loader, c.n_epochs)
    writer.close()

if __name__ == '__main__':
    train()
