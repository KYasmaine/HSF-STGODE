import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, OneCycleLR
import time
from tqdm import tqdm
from loguru import logger

from args import args
from model import ODEGCN, MultiStreamWrapper
from utils import generate_dataset, read_data, get_normalized_adj
from eval import masked_mae_np, masked_mape_np, masked_rmse_np


def compute_lead_time_loss(pred, target, threshold, lead_steps):
    if lead_steps <= 0:
        return pred.new_tensor(0.0)
    B, N, T = pred.shape
    device = pred.device
    timeline = torch.arange(T, device=device)

    true_cond = target <= threshold
    true_idx = torch.where(true_cond, timeline.view(1, 1, T), torch.full((1, 1, T), T, device=device))
    event_true = true_idx.min(dim=-1)[0]

    pred_cond = pred <= threshold
    pred_idx = torch.where(pred_cond, timeline.view(1, 1, T), torch.full((1, 1, T), T, device=device))
    event_pred = pred_idx.min(dim=-1)[0]

    mask = (event_true < T).float()
    if mask.sum() == 0:
        return pred.new_tensor(0.0)

    lead_margin = event_true - event_pred
    violation = torch.clamp(lead_steps - lead_margin, min=0.0)
    loss = (violation * mask).sum() / (mask.sum() + 1e-6)
    return loss


def train(loader, model, optimizer, criterion, device, physics_weight=0.0, delay_weight=0.0,
          lead_weight=0.0, lead_threshold=0.0, lead_steps=0, use_lead=False,
          use_multistream=False, num_streams=0):
    batch_loss = 0
    batch_phys = 0
    phys_steps = 0
    batch_delay = 0
    delay_steps = 0
    batch_lead = 0
    lead_counts = 0
    for idx, batch in enumerate(tqdm(loader)):
        model.train()
        optimizer.zero_grad()

        if hasattr(model, 'reset_delay_history'):
            model.reset_delay_history()
        if use_multistream:
            streams = [tensor.to(device) for tensor in batch[:num_streams]]
            targets = batch[-1].to(device)
            outputs = model(streams)
        else:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
        loss = criterion(outputs, targets)
        if physics_weight > 0 and hasattr(model, 'physics_regularizer'):
            physics_reg = model.physics_regularizer()
            if physics_reg is not None:
                loss = loss + physics_weight * physics_reg
                batch_phys += physics_reg.detach().cpu().item()
                phys_steps += 1
        if delay_weight > 0 and hasattr(model, 'delay_regularizer'):
            delay_reg = model.delay_regularizer()
            if delay_reg is not None:
                loss = loss + delay_weight * delay_reg
                batch_delay += delay_reg.detach().cpu().item()
                delay_steps += 1
        if use_lead and lead_weight > 0:
            lead_loss = compute_lead_time_loss(outputs, targets, lead_threshold, lead_steps)
            if lead_loss is not None:
                loss = loss + lead_weight * lead_loss
                batch_lead += lead_loss.detach().cpu().item()
                lead_counts += 1
        loss.backward()
        optimizer.step()

        batch_loss += loss.detach().cpu().item() 
    avg_phys = batch_phys / phys_steps if phys_steps > 0 else 0.0
    avg_delay = batch_delay / delay_steps if delay_steps > 0 else 0.0
    avg_lead = batch_lead / lead_counts if lead_counts > 0 else 0.0
    return batch_loss / (idx + 1), avg_phys, avg_delay, avg_lead


@torch.no_grad()
def eval(loader, model, std, mean, device, use_multistream=False, num_streams=0):
    batch_rmse_loss = 0  
    batch_mae_loss = 0
    batch_mape_loss = 0
    if hasattr(model, 'reset_delay_history'):
        model.reset_delay_history()
    for idx, batch in enumerate(tqdm(loader)):
        model.eval()

        if hasattr(model, 'reset_delay_history'):
            model.reset_delay_history()
        if use_multistream:
            streams = [tensor.to(device) for tensor in batch[:num_streams]]
            targets = batch[-1].to(device)
            output = model(streams)
        else:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
        
        out_unnorm = output.detach().cpu().numpy()*std + mean
        target_unnorm = targets.detach().cpu().numpy()*std + mean

        mae_loss = masked_mae_np(target_unnorm, out_unnorm, 0)
        rmse_loss = masked_rmse_np(target_unnorm, out_unnorm, 0)
        mape_loss = masked_mape_np(target_unnorm, out_unnorm, 0)
        batch_rmse_loss += rmse_loss
        batch_mae_loss += mae_loss
        batch_mape_loss += mape_loss

    return batch_rmse_loss / (idx + 1), batch_mae_loss / (idx + 1), batch_mape_loss / (idx + 1)


@torch.no_grad()
def eval_horizon(loader, model, std, mean, device, use_multistream=False, num_streams=0):
    """
    Evaluate model and return metrics for each prediction horizon.
    Returns: (overall_rmse, overall_mae, overall_mape, horizon_metrics)
    horizon_metrics is a list of (rmse, mae, mape) for each time step
    """
    batch_rmse_loss = 0  
    batch_mae_loss = 0
    batch_mape_loss = 0
    
    # Initialize lists to store per-horizon metrics
    num_horizons = None
    horizon_rmse = None
    horizon_mae = None
    horizon_mape = None
    
    if hasattr(model, 'reset_delay_history'):
        model.reset_delay_history()
    
    for idx, batch in enumerate(tqdm(loader)):
        model.eval()

        if hasattr(model, 'reset_delay_history'):
            model.reset_delay_history()
        if use_multistream:
            streams = [tensor.to(device) for tensor in batch[:num_streams]]
            targets = batch[-1].to(device)
            output = model(streams)
        else:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
        
        out_unnorm = output.detach().cpu().numpy()*std + mean
        target_unnorm = targets.detach().cpu().numpy()*std + mean

        # Overall metrics
        mae_loss = masked_mae_np(target_unnorm, out_unnorm, 0)
        rmse_loss = masked_rmse_np(target_unnorm, out_unnorm, 0)
        mape_loss = masked_mape_np(target_unnorm, out_unnorm, 0)
        batch_rmse_loss += rmse_loss
        batch_mae_loss += mae_loss
        batch_mape_loss += mape_loss
        
        # Per-horizon metrics
        # Assuming shape is (batch, nodes, time_steps) or (batch, time_steps, nodes)
        if num_horizons is None:
            # Determine the time dimension
            if len(target_unnorm.shape) == 3:
                # Try to find which dimension is time
                if target_unnorm.shape[2] <= 12:  # Assume last dim is time
                    num_horizons = target_unnorm.shape[2]
                else:  # Assume second dim is time
                    num_horizons = target_unnorm.shape[1]
            else:
                num_horizons = target_unnorm.shape[-1]
            
            horizon_rmse = np.zeros(num_horizons)
            horizon_mae = np.zeros(num_horizons)
            horizon_mape = np.zeros(num_horizons)
        
        # Calculate metrics for each horizon
        for t in range(num_horizons):
            if len(target_unnorm.shape) == 3 and target_unnorm.shape[2] <= 12:
                # Shape is (batch, nodes, time)
                target_t = target_unnorm[:, :, t]
                pred_t = out_unnorm[:, :, t]
            else:
                # Shape is (batch, time, nodes)
                target_t = target_unnorm[:, t, :]
                pred_t = out_unnorm[:, t, :]
            
            horizon_mae[t] += masked_mae_np(target_t, pred_t, 0)
            horizon_rmse[t] += masked_rmse_np(target_t, pred_t, 0)
            horizon_mape[t] += masked_mape_np(target_t, pred_t, 0)
    
    # Average over all batches
    num_batches = idx + 1
    overall_rmse = batch_rmse_loss / num_batches
    overall_mae = batch_mae_loss / num_batches
    overall_mape = batch_mape_loss / num_batches
    
    horizon_rmse = horizon_rmse / num_batches
    horizon_mae = horizon_mae / num_batches
    horizon_mape = horizon_mape / num_batches
    
    horizon_metrics = [(horizon_mae[t], horizon_rmse[t], horizon_mape[t]) for t in range(num_horizons)]
    
    return overall_rmse, overall_mae, overall_mape, horizon_metrics


def main(args):
    # random seed
    seed = 2
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    device = torch.device('cuda:'+str(args.num_gpu)) if torch.cuda.is_available() else torch.device('cpu')

    if args.log:
        logger.add('log_{time}.log')
    options = vars(args)
    if args.log:
        logger.info(options)
    else:
        print(options)

    data, mean, std, dtw_matrix, sp_matrix = read_data(args)
    train_loader, valid_loader, test_loader = generate_dataset(data, args)
    A_sp_wave = get_normalized_adj(sp_matrix).to(device)
    A_se_wave = get_normalized_adj(dtw_matrix).to(device)

    physics_config = {'max_scale': args.physics_max_scale}
    delay_config = {'horizon': args.delay_horizon, 'num_patterns': args.delay_patterns}
    net = ODEGCN(num_nodes=data.shape[1], 
                num_features=data.shape[2], 
                num_timesteps_input=args.his_length, 
                num_timesteps_output=args.pred_length, 
                A_sp_hat=A_sp_wave, 
                A_se_hat=A_se_wave,
                use_physics=args.use_physics,
                physics_config=physics_config,
                use_multiscale=args.use_multiscale,
                num_regions=args.num_regions,
                use_delay=args.use_delay,
                delay_config=delay_config,
                use_continuous_readout=args.use_continuous_readout)
    stream_count = 0
    if getattr(args, 'use_multistream_input', False):
        stream_names = getattr(args, 'multi_stream_names', None)
        if not stream_names:
            raise ValueError('Multi-stream input enabled but no streams were generated. Check dataset preparation.')
        stream_count = len(stream_names)
        net = MultiStreamWrapper(net, stream_names, feature_dim=data.shape[2])
    net = net.to(device)
    lr = args.lr
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss()

    best_valid_rmse = 1000
    best_epoch = 0
    best_valid_mae = 1000
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
    
    # Time tracking
    total_train_time = 0
    total_inference_time = 0
    start_total_time = time.time()

    for epoch in range(1, args.epochs+1):
        print("=====Epoch {}=====".format(epoch))
        print('Training...')
        
        # Track training time
        train_start = time.time()
        if hasattr(net, 'reset_delay_history'):
            net.reset_delay_history()
        loss, avg_phys, avg_delay, avg_lead = train(
            train_loader, net, optimizer, criterion, device,
            physics_weight=args.physics_weight if args.use_physics else 0.0,
            delay_weight=args.delay_weight if args.use_delay else 0.0,
            lead_weight=args.lead_weight if args.lead_weight > 0 and args.use_continuous_readout else 0.0,
            lead_threshold=args.lead_threshold,
            lead_steps=args.lead_steps,
            use_lead=args.use_continuous_readout and args.lead_steps > 0,
            use_multistream=args.use_multistream_input,
            num_streams=stream_count
        )
        train_time = time.time() - train_start
        total_train_time += train_time
        
        if args.use_physics and args.physics_weight > 0:
            print(f'Average physics regularizer: {avg_phys:.6f}')
        if args.use_delay and args.delay_weight > 0:
            print(f'Average delay regularizer: {avg_delay:.6f}')
        if args.use_continuous_readout and args.lead_weight > 0 and args.lead_steps > 0:
            print(f'Average lead-time loss: {avg_lead:.6f}')
        print('Evaluating...')
        
        # Track inference time
        inference_start = time.time()
        train_rmse, train_mae, train_mape = eval(train_loader, net, std, mean, device,
                                                 use_multistream=args.use_multistream_input,
                                                 num_streams=stream_count)
        valid_rmse, valid_mae, valid_mape = eval(valid_loader, net, std, mean, device,
                                                 use_multistream=args.use_multistream_input,
                                                 num_streams=stream_count)
        inference_time = time.time() - inference_start
        total_inference_time += inference_time

        if valid_rmse < best_valid_rmse:
            best_valid_rmse = valid_rmse
            best_valid_mae = valid_mae
            best_epoch = epoch
            print('New best results!')
            torch.save(net.state_dict(), f'net_params_{args.filename}_{args.num_gpu}.pkl')

        physics_msg = ''
        if args.use_physics and args.physics_weight > 0:
            physics_msg += f', physics reg: {avg_phys}'
        if args.use_delay and args.delay_weight > 0:
            physics_msg += f', delay reg: {avg_delay}'
        if args.use_continuous_readout and args.lead_weight > 0 and args.lead_steps > 0:
            physics_msg += f', lead loss: {avg_lead}'
        if args.log:
            logger.info(f'\n##on train data## loss: {loss}{physics_msg}, \n' +
                        f'##on train data## rmse loss: {train_rmse}, mae loss: {train_mae}, mape loss: {train_mape}\n' +
                        f'##on valid data## rmse loss: {valid_rmse}, mae loss: {valid_mae}, mape loss: {valid_mape}\n')
        else:
            print(f'\n##on train data## loss: {loss}{physics_msg}, \n' +
                  f'##on train data## rmse loss: {train_rmse}, mae loss: {train_mae}, mape loss: {train_mape}\n' +
                  f'##on valid data## rmse loss: {valid_rmse}, mae loss: {valid_mae}, mape loss: {valid_mape}\n')

        if args.use_multistream_input and hasattr(net, 'pop_gate_history'):
            fast_avg, slow_avg = net.pop_gate_history()
            if fast_avg is not None and slow_avg is not None:
                print('Gate fast avg:', fast_avg.numpy(), 'slow avg:', slow_avg.numpy())
        
        scheduler.step()

    # Calculate average times
    avg_train_time = total_train_time / args.epochs
    avg_inference_time = total_inference_time / args.epochs
    total_time = time.time() - start_total_time
    
    # Print summary
    print('\n' + '='*60)
    print(f'Average Training Time: {avg_train_time:.4f} secs/epoch\n')
    print(f'Average Inference Time: {avg_inference_time:.4f} secs\n')
    print('Training ends\n')
    print(f'The epoch of the best result: {best_epoch}\n')
    print(f'The valid loss of the best model {best_valid_mae:.4f}\n')
    
    # Load best model and evaluate on test set with per-horizon metrics
    net.load_state_dict(torch.load(f'net_params_{args.filename}_{args.num_gpu}.pkl'))
    test_rmse, test_mae, test_mape, horizon_metrics = eval_horizon(
        test_loader, net, std, mean, device,
        use_multistream=args.use_multistream_input,
        num_streams=stream_count
    )
    
    # Print per-horizon results
    for horizon, (mae, rmse, mape) in enumerate(horizon_metrics, 1):
        # Convert MAPE from percentage to decimal (divide by 100)
        mape_decimal = mape / 100.0
        print(f'Evaluate best model on test data for horizon {horizon}, Test MAE: {mae:.4f}, Test RMSE: {rmse:.4f}, Test MAPE: {mape_decimal:.4f},\n')
    
    # Print average results
    test_mape_decimal = test_mape / 100.0
    print(f'On average over {len(horizon_metrics)} horizons, Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}, Test MAPE: {test_mape_decimal:.4f}\n')
    print(f'Total time spent: {total_time:.4f}\n')
    print('='*60)


if __name__ == '__main__':
    main(args)
