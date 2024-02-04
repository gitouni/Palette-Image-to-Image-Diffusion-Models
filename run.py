import argparse
import os
import torch
import torch.multiprocessing as mp

from core.logger import VisualWriter, InfoLogger
import core.praser as Praser
import core.util as Util
from data import define_dataloader
from models import create_model, define_network, define_loss, define_metric
from models.model import Palette
from data.dataset import PatchInapintDataset


def main_worker(gpu, ngpus_per_node, opt):
    """  threads running on each GPU """
    if 'local_rank' not in opt:
        opt['local_rank'] = opt['global_rank'] = gpu
    if opt['distributed']:
        torch.cuda.set_device(int(opt['local_rank']))
        print('using GPU {} for training'.format(int(opt['local_rank'])))
        print("npgus_per_node:{}".format(ngpus_per_node))
        torch.distributed.init_process_group(backend = 'nccl', 
            init_method = opt['init_method'],
            world_size = opt['world_size'], 
            rank = opt['global_rank'],
            group_name='mtorch'
        )
    '''set seed and and cuDNN environment '''
    torch.backends.cudnn.enabled = True
    if opt['phase'] == 'semi':
        semi_mode = True
        opt['phase'] = 'train'
    else:
        semi_mode = False
    # warnings.warn('You have chosen to use cudnn for accleration. torch.backends.cudnn.enabled=True')
    Util.set_seed(opt['seed'])

    ''' set logger '''
    phase_logger = InfoLogger(opt)
    phase_writer = VisualWriter(opt, phase_logger)  
    phase_logger.info('Create the log file in directory {}.\n'.format(opt['path']['experiments_root']))

    '''set networks and dataset'''
    phase_loader, val_loader = define_dataloader(phase_logger, opt) # val_loader is None if phase is test.
    networks = [define_network(phase_logger, opt, item_opt) for item_opt in opt['model']['which_networks']]

    ''' set metrics, loss, optimizer and  schedulers '''
    metrics = [define_metric(phase_logger, item_opt) for item_opt in opt['model']['which_metrics']]
    losses = [define_loss(phase_logger, item_opt) for item_opt in opt['model']['which_losses']]

    model:Palette = create_model(
        opt = opt,
        networks = networks,
        phase_loader = phase_loader,
        val_loader = val_loader,
        losses = losses,
        metrics = metrics,
        logger = phase_logger,
        writer = phase_writer
    )

    phase_logger.info('Begin model {}.'.format(opt['phase']))
    if isinstance(phase_loader.dataset, PatchInapintDataset):
        setattr(model, 'patch_idx', phase_loader.dataset.patch_idx)
        setattr(model, 'patch_num', phase_loader.dataset.patch_num)
        setattr(model, 'target_img_size', phase_loader.dataset.image_size)
    try:
        if opt['phase'] == 'train':
            model.train()
            if semi_mode and (not opt['no_test']):
                opt['phase'] = 'test'
                model.update_loader(opt)
                model.iter = 0
                model.epoch = 0
                if isinstance(model.phase_loader.dataset, PatchInapintDataset):
                    setattr(model, 'patch_idx', model.phase_loader.dataset.patch_idx)
                    setattr(model, 'patch_num', model.phase_loader.dataset.patch_num)
                    setattr(model, 'target_img_size', model.phase_loader.dataset.image_size)
                    model.patch_test()
                else:
                    model.test()
        elif opt['phase'] == 'test':
            if isinstance(phase_loader.dataset, PatchInapintDataset):
                model.patch_test()
            else:
                model.test()
        else:
            raise NotImplementedError("phase {} not implemented.".format(opt['phase']))
    finally:
        phase_writer.close()
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/nmk_rand.json', help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train','test','semi'], help='Run train, test or semi', default='train')
    parser.add_argument('-nt','--no_test', action="store_true")
    parser.add_argument('-b', '--batch', type=int, default=None, help='Batch size in every gpu')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-P', '--port', default='21012', type=str)
    parser.add_argument("-rr","--recursive_refine",action="store_true")

    ''' parser configs '''
    args = parser.parse_args()
    opt = Praser.parse(args)
    opt['no_test'] = args.no_test
    ''' cuda devices '''
    gpu_str = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    print('export CUDA_VISIBLE_DEVICES={}'.format(gpu_str))

    ''' use DistributedDataParallel(DDP) and multiprocessing for multi-gpu training'''
    # [Todo]: multi GPU on multi machine
    if opt['distributed']:
        ngpus_per_node = len(opt['gpu_ids']) # or torch.cuda.device_count()
        opt['world_size'] = ngpus_per_node
        opt['init_method'] = 'tcp://127.0.0.1:'+ args.port 
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        opt['world_size'] = 1 
        main_worker(0, 1, opt)