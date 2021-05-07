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

from annealer import Annealer
from shepardmetzler import ShepardMetzler
from partition import partition
from argparse import ArgumentParser
import GPUtil

import config as c
from data import get_data

from model_reverseNN import get_model

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
    args = parser.parse_args()

    # Rate annealing schemes
    sigma_scheme = Annealer(2.0, 0.7, 80000)
    mu_scheme = Annealer(5 * 10 ** (-6), 5 * 10 ** (-6), 1.6 * 10 ** 5)

    def run_iter(batch, mode):
        
        x, v = batch
        x, v = x.to(device), v.to(device)

        x, v, x_q, v_q = partition(x, v)

        #x_mu, x_mu_d, kl, nll = model.combined_model(x, v, x_q, v_q)
        # Log likelihood
        if mode == 'train':
            x_mu, kl, v_loss = model.combined_model(x, v, x_q, v_q, False)
            sigma = next(sigma_scheme)
        elif mode == 'val':
            x_mu, kl, v_loss = model.combined_model(x, v, x_q, v_q, True)
            sigma = sigma_scheme.recent
        ll = Normal(x_mu, sigma).log_prob(x_q)
        likelihood     = torch.mean(torch.sum(ll, dim=[1, 2, 3]))
        kl_divergence  = torch.mean(torch.sum(kl.view(c.batch_size, -1), dim=[1]))

        # Evidence lower bound
        elbo = likelihood - kl_divergence
        loss = -elbo #+ v_loss * c.loss_coeff
        
        return loss, elbo, v_loss
    

    def step(engine, batch):
        model.combined_model.train()
        #model.combined_model.glow.zero_grad()
        
        loss, elbo, v_loss = run_iter(batch, 'train')
        
        
        loss.backward()
        
        #torch.nn.utils.clip_grad_value_(model.combined_model.glow.parameters(), 5)
        #grad_norm = torch.nn.utils.clip_grad_norm_(model.combined_model.glow.parameters(), 100)

        model.optimizer.zero_grad()
        model.optimizer.step()
        
        with torch.no_grad():
            # Anneal learning rate
            mu = next(mu_scheme)
            i = engine.state.iteration
            for group in model.optimizer.param_groups:
                group["lr"] = mu * math.sqrt(1 - 0.999 ** i) / (1 - 0.9 ** i)

        return {"loss":loss.item(), "v_loss": v_loss.item(), "elbo": -elbo.item()}


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
            
            loss, elbo, v_loss = run_iter(batch, 'val')
            
            writer.add_scalar("validation/loss", loss.item(), engine.state.epoch)
            writer.add_scalar("validation/elbo", elbo.item(), engine.state.epoch)

    @trainer.on(Events.EXCEPTION_RAISED)
    def handle_exception(engine, e):
        writer.close()
        engine.terminate()
        if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
            import warnings
            warnings.warn('KeyboardInterrupt caught. Exiting gracefully.')
            checkpoint_handler(engine, { 'model_exception': model })
        else: raise e

    metric_names = ["loss", "elbo", "v_loss"]
    RunningAverage(output_transform=lambda x: x["loss"]).attach(trainer, "loss")
    RunningAverage(output_transform=lambda x: x["elbo"]).attach(trainer, "elbo")
    RunningAverage(output_transform=lambda x: x["v_loss"]).attach(trainer, "v_loss")
    ProgressBar().attach(trainer, metric_names=metric_names)

    # Model checkpointing
    to_save = {'trainer': trainer, 'model': model.combined_model, 'optimizer': model.optimizer, 'sigma_scheme': sigma_scheme, 'mu_scheme': mu_scheme}
    checkpoint_handler = ModelCheckpoint(c.pretrained_path, "checkpoint", n_saved=3, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=2), checkpoint_handler, to_save=to_save) #(every=2)

    timer = Timer(average=True).attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # Tensorbard writer
    writer = SummaryWriter(log_dir=c.log_dir)

    if args.resume_training == True:
        checkpoints_dir = os.listdir(c.pretrained_path)
        checkpoints_dir = [x for x in checkpoints_dir if 'checkpoint_checkpoint_' in x]
        #print(checkpoints_dir)
        resume_epoch = len(checkpoints_dir)

        checkpoint_path = os.path.join(c.pretrained_path, checkpoints_dir[-1])

        to_load = {'trainer': trainer, 'model': model, 'optimizer': model.optimizer,
                   'sigma_scheme': sigma_scheme, 'mu_scheme': mu_scheme}

        #print(checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint)

        trainer.state.iteration = (resume_epoch) * len(train_loader)
        trainer.state.epoch = (resume_epoch)

        print('Resuming Training at Epoch ', trainer.state.epoch, '... Iteration ', trainer.state.iteration)




    random.seed(99)

    trainer.run(train_loader, c.n_epochs)
    writer.close()

if __name__ == '__main__':
    train()
