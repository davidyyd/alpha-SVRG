import math
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from itertools import combinations
import utils
from svrg import variance_metrics
from utils import save_snapshot, load_snapshot

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None, schedules={},
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False, snapshot_pass_freq=None, 
                    use_svrg=False, compute_variance=False, snapshot_dir=None, model_without_ddp=None, use_cache_svrg=True, 
                    snapshot_model=None, use_optimal_coefficient=False):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    # get the current random state
    if use_svrg:
        optimizer.save_epoch_state()
        if compute_variance:
            # we save the snapshot for later variance computation
            save_snapshot(snapshot_dir, 0, model_without_ddp)
        if use_cache_svrg:
            # start snapshot phase
            optimizer.start_snapshot_phase()
            cache_svrg_snapshot_pass(model, model_without_ddp, optimizer, criterion, data_loader, device, mixup_fn, update_freq, num_training_steps_per_epoch)
            # end snapshot phase
            optimizer.end_snapshot_phase()
            optimizer.set_epoch_state()
    optimizer.zero_grad()
    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if data_iter_step % update_freq == 0:
            if lr_schedule_values is not None or wd_schedule_values is not None:
                for i, param_group in enumerate(optimizer.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                    if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                        param_group["weight_decay"] = wd_schedule_values[it]

        if compute_variance and it % 20 == 0:
            # save model
            save_snapshot(snapshot_dir, step, model_without_ddp)
        # svrg full batch gradient calculation
        if use_svrg and step % snapshot_pass_freq == 0 and data_iter_step % update_freq == 0 and not use_cache_svrg:
            # save the current random state
            optimizer.save_curr_state()
            if compute_variance:
                # we save the snapshot for later variance computation
                save_snapshot(snapshot_dir, step, model_without_ddp)
            optimizer.synchronize()
            # reset the random state to the beginning of the epoch
            vanilla_svrg_snapshot_update(model, model_without_ddp, optimizer, criterion, data_loader, device, mixup_fn, update_freq, num_training_steps_per_epoch)
            # resume the state to the beginning of snapshot calculation
            optimizer.set_curr_state()
        
        if use_svrg and use_optimal_coefficient and data_iter_step % update_freq == 0:
            # save the current random state
            optimizer.save_curr_state()
            # reset the random state to the beginning of the epoch
            optimizer.set_epoch_state()
            if use_cache_svrg:
                optimizer.start_model_phase()
                cache_svrg_snapshot_pass(model, model_without_ddp, optimizer, criterion, data_loader, device, mixup_fn, update_freq, num_training_steps_per_epoch)
                optimizer.end_model_phase()
            # resume the state to the beginning of snapshot calculation
            optimizer.set_curr_state()
        
        if use_svrg:
            optimizer.coefficient = schedules['svrg'][it]
            if use_optimal_coefficient:
                optimizer.compute_optimal_coefficient()
                coefs = []
                if log_writer is not None:
                    for i, coef in enumerate(optimizer.coefficient):
                        coefs.append(coef.reshape(-1))
                    coefs = torch.cat(coefs, axis=0)
                    log_writer.update(coefficient_distribution=coefs, head=f'svrg')
                    log_writer.update(coefficient=torch.mean(coefs), head=f'svrg')
            else:
                if log_writer is not None:
                    log_writer.update(coefficient=optimizer.coefficient, head=f'svrg')
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        if use_svrg and not use_cache_svrg:
            optimizer.save_curr_state()
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(samples)
                loss = criterion(output, targets)
        else: # full precision
            output = model(samples)
            loss = criterion(output, targets)
        loss_value = loss.item()

        if use_svrg and not use_cache_svrg:
            optimizer.set_curr_state()
            output2 = snapshot_model(samples)
            loss2 = criterion(output2, targets)

        if not math.isfinite(loss_value): # this could trigger if using AMP
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
        else: # full precision
            loss /= update_freq
            loss.backward()
            if use_svrg and not use_cache_svrg:
                loss2 /= update_freq
                loss2.backward()
            if (data_iter_step + 1) % update_freq == 0:
                if use_svrg:
                    # update grad use svrg
                    optimizer.update_grad()
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)

        if use_amp:
            metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if wandb_logger:
            wandb_logger._wandb.log({
                'Rank-0 Batch Wise/train_loss': loss_value,
                'Rank-0 Batch Wise/train_max_lr': max_lr,
                'Rank-0 Batch Wise/train_min_lr': min_lr
            }, commit=False)
            if class_acc:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_class_acc': class_acc}, commit=False)
            if use_amp:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_grad_norm': grad_norm}, commit=False)
            wandb_logger._wandb.log({'Rank-0 Batch Wise/global_train_step': it})
            
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def vanilla_svrg_snapshot_update(model, model_without_ddp, optimizer, criterion, data_loader, device, mixup_fn, 
                        update_freq, num_samples):
    optimizer.zero_grad()
    optimizer.synchronize()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Calculate mini-batch gradient for snapshot:"
    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        step = data_iter_step // update_freq
        if step >= num_samples:
            continue
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        output = model(samples)
        loss = criterion(output, targets)
        metric_logger.update(svrg_mini_batch_loss=loss)
        loss /= update_freq * num_samples
        loss.backward()
        torch.cuda.synchronize()
    optimizer.calculate_full_batch_grad()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def cache_svrg_snapshot_pass(model, model_without_ddp, optimizer, criterion, data_loader, device, mixup_fn, 
                        update_freq, num_samples):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Calculate mini-batch gradient for snapshot:"
    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        step = data_iter_step // update_freq
        if step >= num_samples:
            continue
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        output = model(samples)
        loss = criterion(output, targets)
        metric_logger.update(svrg_mini_batch_loss=loss)
        loss /= update_freq
        loss.backward()
        if (data_iter_step + 1) % update_freq == 0:
            # collect accumulated grad as one single mini-batch
            optimizer.collect_mini_batch_grad()
            optimizer.zero_grad()
        torch.cuda.synchronize()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def compute_variance(model, model_without_ddp, optimizer, criterion, data_loader, device, mixup_fn, 
                        snapshot_pass_freq, update_freq, epoch, num_training_steps_per_epoch, snapshot_dir, 
                        use_svrg, log_writer, schedules, snapshot_model=None, use_cache_svrg=True,
                        use_optimal_coefficient=None):
    metric_logger = utils.MetricLogger(delimiter="  ")
    print(f"Compute variance for epoch {epoch}")
    header = "Calculate mini-batch gradient for variance computation:"
    if use_svrg:
        optimizer.save_epoch_state()
    for data_iter_step in range(num_training_steps_per_epoch):
        step = data_iter_step // update_freq
        it = epoch*num_training_steps_per_epoch+step
        if use_svrg and step % snapshot_pass_freq == 0 and data_iter_step % update_freq == 0:
            optimizer.coefficient = schedules['svrg'][it]
            # load snapshot model
            load_snapshot(snapshot_dir, step, model_without_ddp)
            # reset the random state to the beginning of the epoch
            optimizer.set_epoch_state()
            if use_cache_svrg:
                # start snapshot phase
                optimizer.start_snapshot_phase()
                cache_svrg_snapshot_pass(model, model_without_ddp, optimizer, criterion, data_loader, device, mixup_fn, update_freq, num_training_steps_per_epoch)
                # end snapshot phase
                optimizer.end_snapshot_phase()
            else:
                optimizer.synchronize()
                # reset the random state to the beginning of the epoch
                vanilla_svrg_snapshot_update(model, model_without_ddp, optimizer, criterion, data_loader, device, mixup_fn, update_freq, num_training_steps_per_epoch)
            # resume the state to the beginning of snapshot calculation
            optimizer.set_epoch_state()
        if it % 80 != 0 or step == 0:
            continue
        # load checkpoint model
        load_snapshot(snapshot_dir, step, model_without_ddp)
        if use_svrg and use_optimal_coefficient and data_iter_step % update_freq == 0:
            # save the current random state
            # reset the random state to the beginning of the epoch
            optimizer.set_epoch_state()
            optimizer.start_model_phase()
            cache_svrg_snapshot_pass(model, model_without_ddp, optimizer, criterion, data_loader, device, mixup_fn, update_freq, num_training_steps_per_epoch)
            optimizer.end_model_phase()
            # resume the state to the beginning of snapshot calculation
            optimizer.set_epoch_state()
            optimizer.compute_optimal_coefficient()
        all_grads = []
        for inner_data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
            inner_step = inner_data_iter_step // update_freq
            optimizer.curr_step = inner_step
            if inner_step >= num_training_steps_per_epoch:
                continue
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if mixup_fn is not None:
                samples, targets = mixup_fn(samples, targets)

            if use_svrg:
                optimizer.save_curr_state()

            output = model(samples)
            loss = criterion(output, targets)
            loss /= update_freq
            loss.backward()

            if use_svrg and not use_cache_svrg:
                optimizer.set_curr_state()
                output2 = snapshot_model(samples)
                loss2 = criterion(output2, targets)
                loss2 /= update_freq
                loss2.backward()

            if (inner_data_iter_step + 1) % update_freq == 0:
                # calculate svrg grad
                if use_svrg:
                    optimizer.update_grad()
                # calculate grad
                grads = []
                for weight in model.module.parameters():
                    grads.append(weight.grad.reshape(-1))
                all_grads.append(torch.cat(grads))
                optimizer.zero_grad()
            torch.cuda.synchronize()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        all_grads = torch.stack(all_grads, dim = 0)

        cosine_dist, variance, spectral_norm = variance_metrics(all_grads)
        print(f'* Variance Stats: Cosine Distance {cosine_dist:.3f} Variance {variance:.3f} Spectral Norm {spectral_norm:.3f}')
        if log_writer is not None:
            log_writer.update(cosine_dist=cosine_dist, head = "var", step=it)
            log_writer.update(variance=variance, head = "var", step=it)
            log_writer.update(spectral_norm=spectral_norm, head = "var", step=it)
        

