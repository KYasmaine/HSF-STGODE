import os
import csv
from pathlib import Path
import numpy as np
from fastdtw import fastdtw
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
import torch


DATA_DIR = Path(__file__).resolve().parent / 'data'

files = {
    'pems03': ['PEMS03/pems03.npz', 'PEMS03/distance.csv'],
    'pems04': ['PEMS04/pems04.npz', 'PEMS04/distance.csv'],
    'pems07': ['PEMS07/pems07.npz', 'PEMS07/distance.csv'],
    'pems08': ['PEMS08/pems08.npz', 'PEMS08/distance.csv'],
    'pemsbay': ['PEMSBAY/pems_bay.npz', 'PEMSBAY/distance.csv'],
    'pemsD7M': ['PeMSD7M/PeMSD7M.npz', 'PeMSD7M/distance.csv'],
    'pemsD7L': ['PeMSD7L/PeMSD7L.npz', 'PeMSD7L/distance.csv']
}

def read_data(args):
    """read data, generate spatial adjacency matrix and semantic adjacency matrix by dtw

    Args:
        sigma1: float, default=0.1, sigma for the semantic matrix
        sigma2: float, default=10, sigma for the spatial matrix
        thres1: float, default=0.6, the threshold for the semantic matrix
        thres2: float, default=0.5, the threshold for the spatial matrix

    Returns:
        data: tensor, T * N * 1
        dtw_matrix: array, semantic adjacency matrix
        sp_matrix: array, spatial adjacency matrix
    """
    filename = args.filename
    file = files[filename]
    if args.remote:
        filepath = Path('/home/lantu.lqq/ftemp/data/')
    else:
        filepath = DATA_DIR
    with np.load((filepath / file[0]).as_posix(), allow_pickle=True) as npz:
        data = npz['data']
        sensor_ids = npz['sensor_ids'] if 'sensor_ids' in npz.files else None
    id_to_index = None
    if sensor_ids is not None:
        # map raw sensor identifiers (e.g. 317842) to the dense indices used in data
        sensor_ids = sensor_ids.astype(int)
        id_to_index = {int(sensor_id): idx for idx, sensor_id in enumerate(sensor_ids)}
    # PEMS04 == shape: (16992, 307, 3)    feature: flow,occupy,speed
    # PEMSD7M == shape: (12672, 228, 1)
    # PEMSD7L == shape: (12672, 1026, 1)
    num_node = data.shape[1]
    mean_value = np.mean(data, axis=(0, 1)).reshape(1, 1, -1)
    std_value = np.std(data, axis=(0, 1)).reshape(1, 1, -1)
    data = (data - mean_value) / std_value
    mean_value = mean_value.reshape(-1)[0]
    std_value = std_value.reshape(-1)[0]

    dtw_path = DATA_DIR / f'{filename}_dtw_distance.npy'
    if not os.path.exists(dtw_path):
        data_mean = np.mean([data[:, :, 0][24*12*i: 24*12*(i+1)] for i in range(data.shape[0]//(24*12))], axis=0)
        data_mean = data_mean.squeeze().T 
        dtw_distance = np.zeros((num_node, num_node))
        for i in tqdm(range(num_node)):
            for j in range(i, num_node):
                dtw_distance[i][j] = fastdtw(data_mean[i], data_mean[j], radius=6)[0]
        for i in range(num_node):
            for j in range(i):
                dtw_distance[i][j] = dtw_distance[j][i]
        np.save(dtw_path, dtw_distance)

    dist_matrix = np.load(dtw_path)

    mean = np.mean(dist_matrix)
    std = np.std(dist_matrix)
    dist_matrix = (dist_matrix - mean) / std
    sigma = args.sigma1
    dist_matrix = np.exp(-dist_matrix ** 2 / sigma ** 2)
    dtw_matrix = np.zeros_like(dist_matrix)
    dtw_matrix[dist_matrix > args.thres1] = 1

    # # use continuous semantic matrix
    # if not os.path.exists(f'data/{filename}_dtw_c_matrix.npy'):
    #     dist_matrix = np.load(f'data/{filename}_dtw_distance.npy')
    #     # normalization
    #     std = np.std(dist_matrix[dist_matrix != np.float('inf')])
    #     mean = np.mean(dist_matrix[dist_matrix != np.float('inf')])
    #     dist_matrix = (dist_matrix - mean) / std
    #     sigma = 0.1
    #     dtw_matrix = np.exp(- dist_matrix**2 / sigma**2)
    #     dtw_matrix[dtw_matrix < 0.5] = 0 
    #     np.save(f'data/{filename}_dtw_c_matrix.npy', dtw_matrix)
    # dtw_matrix = np.load(f'data/{filename}_dtw_c_matrix.npy')
    
    # use continuous spatial matrix
    spatial_path = DATA_DIR / f'{filename}_spatial_distance.npy'
    regenerate_spatial = True
    if spatial_path.exists():
        try:
            cached = np.load(spatial_path)
            if cached.shape == (num_node, num_node) and np.isfinite(cached).any():
                regenerate_spatial = False
        except Exception:
            regenerate_spatial = True
    if regenerate_spatial:
        with open((filepath / file[1]).as_posix(), 'r') as fp:
            dist_matrix = np.zeros((num_node, num_node)) + np.float('inf')
            reader = csv.reader(fp)
            next(reader, None)
            edges = []
            for line in reader:
                if len(line) < 3:
                    continue
                raw_start = int(float(line[0]))
                raw_end = int(float(line[1]))
                dist = float(line[2])
                edges.append((raw_start, raw_end, dist))
            if id_to_index is None:
                unique_ids = sorted({node for edge in edges for node in edge[:2]})
                if len(unique_ids) != num_node:
                    raise ValueError(f'Mismatch between data nodes ({num_node}) and distance.csv ids ({len(unique_ids)}).')
                id_to_index = {node_id: idx for idx, node_id in enumerate(unique_ids)}
            for raw_start, raw_end, dist in edges:
                start = id_to_index.get(raw_start)
                end = id_to_index.get(raw_end)
                if start is None or end is None:
                    continue
                dist_matrix[start][end] = dist
                dist_matrix[end][start] = dist
            np.save(spatial_path, dist_matrix)

    # use 0/1 spatial matrix
    # if not os.path.exists(f'data/{filename}_sp_matrix.npy'):
    #     dist_matrix = np.load(f'data/{filename}_spatial_distance.npy')
    #     sp_matrix = np.zeros((num_node, num_node))
    #     sp_matrix[dist_matrix != np.float('inf')] = 1
    #     np.save(f'data/{filename}_sp_matrix.npy', sp_matrix)
    # sp_matrix = np.load(f'data/{filename}_sp_matrix.npy')

    dist_matrix = np.load(spatial_path)
    # normalization
    std = np.std(dist_matrix[dist_matrix != np.float('inf')])
    mean = np.mean(dist_matrix[dist_matrix != np.float('inf')])
    dist_matrix = (dist_matrix - mean) / std
    sigma = args.sigma2
    sp_matrix = np.exp(- dist_matrix**2 / sigma**2)
    sp_matrix[sp_matrix < args.thres2] = 0 
    # np.save(f'data/{filename}_sp_c_matrix.npy', sp_matrix)
    # sp_matrix = np.load(f'data/{filename}_sp_c_matrix.npy')

    print(f'average degree of spatial graph is {np.sum(sp_matrix > 0)/2/num_node}')
    print(f'average degree of semantic graph is {np.sum(dtw_matrix > 0)/2/num_node}')
    return torch.from_numpy(data.astype(np.float32)), mean_value, std_value, dtw_matrix, sp_matrix


def get_normalized_adj(A):
    """
    Returns a tensor, the degree normalized adjacency matrix.
    """
    alpha = 0.8
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    A_reg = alpha / 2 * (np.eye(A.shape[0]) + A_wave)
    return torch.from_numpy(A_reg.astype(np.float32))


class MyDataset(Dataset):
    def __init__(self, data, split_start, split_end, his_length, pred_length):
        split_start = int(split_start)
        split_end = int(split_end)
        self.data = data[split_start: split_end]
        self.his_length = his_length
        self.pred_length = pred_length
    
    def __getitem__(self, index):
        x = self.data[index: index + self.his_length].permute(1, 0, 2)
        y = self.data[index + self.his_length: index + self.his_length + self.pred_length][:, :, 0].permute(1, 0)
        return torch.Tensor(x), torch.Tensor(y)
    def __len__(self):
        return self.data.shape[0] - self.his_length - self.pred_length + 1


def _build_stream_specs(args):
    specs = []
    pph = args.points_per_hour
    if getattr(args, 'num_hours_input', 0) > 0:
        offsets = [i * pph for i in range(args.num_hours_input)]
        specs.append({'name': 'hour', 'offsets': offsets})
    if getattr(args, 'num_days_input', 0) > 0:
        offsets = [(i + 1) * 24 * pph for i in range(args.num_days_input)]
        specs.append({'name': 'day', 'offsets': offsets})
    if getattr(args, 'num_weeks_input', 0) > 0:
        offsets = [(i + 1) * 7 * 24 * pph for i in range(args.num_weeks_input)]
        specs.append({'name': 'week', 'offsets': offsets})
    return specs


def _collect_stream_sample(data_np, label_idx, his_length, offsets):
    seqs = []
    for lag in offsets:
        start = label_idx - lag - his_length
        end = label_idx - lag
        if start < 0:
            return None
        seqs.append(data_np[start:end])
    stacked = np.stack(seqs, axis=0)
    return stacked.mean(axis=0)


def _build_multistream_samples(data_np, args, stream_specs):
    T = data_np.shape[0]
    pred_len = args.pred_length
    his_length = args.his_length
    max_required = max((max(spec['offsets']) if spec['offsets'] else 0) + his_length
                       for spec in stream_specs)
    samples = []
    for label_idx in range(max_required, T - pred_len):
        streams = {}
        valid = True
        for spec in stream_specs:
            sample = _collect_stream_sample(data_np, label_idx, his_length, spec['offsets'])
            if sample is None:
                valid = False
                break
            streams[spec['name']] = sample
        if not valid:
            continue
        target = data_np[label_idx: label_idx + pred_len][:, :, 0]
        samples.append((streams, target))
    return samples


def _streams_to_dataset(samples, stream_names, pred_length):
    def to_tensor(entry_stream):
        # entry_stream shape (T, N, F) -> (N, T, F)
        return torch.from_numpy(entry_stream.transpose(1, 0, 2)).float()
    tensors = []
    for stream_name in stream_names:
        tensors.append(torch.stack(
            [to_tensor(streams[stream_name]) for streams, _ in samples]
        ))
    targets = torch.stack([
        torch.from_numpy(target.transpose(1, 0)).float()
        for _, target in samples
    ])
    dataset = TensorDataset(*tensors, targets)
    return dataset


def generate_dataset(data, args):
    """
    Args:
        data: input dataset, shape like T * N
        batch_size: int 
        train_ratio: float, the ratio of the dataset for training
        his_length: the input length of time series for prediction
        pred_length: the target length of time series of prediction

    Returns:
        train_dataloader: torch tensor, shape like batch * N * his_length * features
        test_dataloader: torch tensor, shape like batch * N * pred_length * features
    """
    batch_size = args.batch_size
    train_ratio = args.train_ratio
    valid_ratio = args.valid_ratio
    his_length = args.his_length
    pred_length = args.pred_length

    if getattr(args, 'use_multistream_input', False):
        stream_specs = _build_stream_specs(args)
        if not stream_specs:
            raise ValueError('use_multistream_input is True but no stream specs were constructed.')
        data_np = data.numpy()
        samples = _build_multistream_samples(data_np, args, stream_specs)
        if len(samples) == 0:
            raise ValueError('No valid samples generated for multi-stream setting. Check lag parameters.')
        total = len(samples)
        split1 = int(total * train_ratio)
        split2 = int(total * (train_ratio + valid_ratio))
        train_samples = samples[:split1]
        val_samples = samples[split1:split2]
        test_samples = samples[split2:]
        stream_names = [spec['name'] for spec in stream_specs]
        setattr(args, 'multi_stream_names', stream_names)
        train_dataset = _streams_to_dataset(train_samples, stream_names, pred_length)
        val_dataset = _streams_to_dataset(val_samples, stream_names, pred_length)
        test_dataset = _streams_to_dataset(test_samples, stream_names, pred_length)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_dataloader, val_dataloader, test_dataloader

    train_dataset = MyDataset(data, 0, data.shape[0] * train_ratio, his_length, pred_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = MyDataset(data, data.shape[0]*train_ratio, data.shape[0]*(train_ratio+valid_ratio), his_length, pred_length)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = MyDataset(data, data.shape[0]*(train_ratio+valid_ratio), data.shape[0], his_length, pred_length)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, valid_dataloader, test_dataloader
