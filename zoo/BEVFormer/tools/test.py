# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import argparse
from genericpath import isdir
import os
import sys
sys.path.append("./")
sys.path.append("../../")
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed, single_gpu_test
from corruption_test import custom_multi_gpu_corruption_test
from projects.mmdet3d_plugin.bevformer.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp

from tools.analysis_tools.parse_results import collect_average_metric, collect_metric, Logging_str


def get_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--corruption_test',
        action='store_true',
        default=False,
        help='Whether to test images under corruptions')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')

    # perturbation
    parser.add_argument('--hue', default=0.1, type=float,
        help='range should be in [-PI, PI], while 0 means no shift'
        'control: [ - defautl * PI, defautl * PI]')
    parser.add_argument('--saturation', default=0.2, type=float,
        help='range should be in [0, 2], while 1 means no shift'
        'control: [ 1 - defautl, 1 + defautl]')
    parser.add_argument('--contrast', default=0.0, type=float,
        help='range should be in [0, 2], while 1 means no shift'
        'control: [ 1 - defautl, 1 + defautl]')
    parser.add_argument('--bright', default=0.0, type=float,
        help='range should be in [0, 1], while 0 means no shift'
        'control: TBD')
    # DIRECT
    parser.add_argument('--max-evaluation', default=500, type=int)
    parser.add_argument('--max-deep', default=10, type=int)
    parser.add_argument('--po-set', action='store_true')
    parser.add_argument('--po-set-size', default=1, type=int)
    parser.add_argument('--max-iteration', default=50, type=int)
    parser.add_argument('--tolerance', default=1e-5, type=float)

    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args

def main():

    args = get_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)
    print(f'Batch size per GPU: {samples_per_gpu}')

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    
    # for debug purpose
    test = dataset[0]

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    # if not distributed:
    #     assert False, "Only support distributed test right now"
    #     # model = MMDataParallel(model, device_ids=[0])
    #     # outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    # else:
    #     model = MMDistributedDataParallel(
    #         model.cuda(),
    #         device_ids=[torch.cuda.current_device()],
    #         broadcast_buffers=False)

    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    config_file = os.path.basename(args.config)
    logging_path = os.path.join('log', f'{config_file[:-3]}.log')
    if not os.path.isdir('log'): os.makedirs('log')
    logging = Logging_str(logging_path)

    import time
    logging.write(f'Time: {time.asctime(time.localtime(time.time()))}\n')

    if args.corruption_test:

        for corruption in cfg.corruptions:

            results_dict_list = []

            logging.write(f'### Evaluating {corruption}\n')

            for severity in [2, 4, 5]:

                if not corruption == 'Clean':
                    logging.write(f'#### Severity-{severity}\n')
                    outputs = custom_multi_gpu_corruption_test(model, data_loader, corruption, severity, 
                                                                    args.tmpdir, args.gpu_collect)
                else:
                    outputs = custom_multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)

                rank, _ = get_dist_info()
                if rank == 0:
                    if args.out:
                        print(f'\nwriting results to {args.out}')
                        assert False
                        #mmcv.dump(outputs['bbox_results'], args.out)
                    kwargs = {} if args.eval_options is None else args.eval_options
                    kwargs['jsonfile_prefix'] = osp.join('test', cfg.model.type, corruption, f'severity_{severity}')
                    if args.format_only:
                        dataset.format_results(outputs, **kwargs)

                    if args.eval:
                        eval_kwargs = cfg.get('evaluation', {}).copy()
                        # hard-code way to remove EvalHook args
                        for key in [
                                'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                                'rule'
                        ]:
                            eval_kwargs.pop(key, None)
                        eval_kwargs.update(dict(metric=args.eval, **kwargs))

                        results_dict = dataset.evaluate(outputs, **eval_kwargs)
                        results_dict['corruption'] = corruption
                        results_dict['severity'] = severity
                        results_dict_list.append(results_dict)

                    collect_metric(results_dict, logging)

                # break severity loop is test clean image
                if corruption == 'Clean':
                    break
            if rank == 0:
                if not corruption == 'Clean':
                    logging.write(f'#### Average\n')
                    collect_average_metric(results_dict_list, logging)

    else:
        # outputs = custom_multi_gpu_test(model, data_loader, args.tmpdir,
        #                                 args.gpu_collect)
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
        rank, _ = get_dist_info()
        if rank == 0:
            if args.out:
                print(f'\nwriting results to {args.out}')
                assert False
                #mmcv.dump(outputs['bbox_results'], args.out)
            kwargs = {} if args.eval_options is None else args.eval_options
            kwargs['jsonfile_prefix'] = osp.join('test', args.config.split(
                '/')[-1].split('.')[-2], time.ctime().replace(' ', '_').replace(':', '_'))
            if args.format_only:
                dataset.format_results(outputs, **kwargs)

            if args.eval:
                eval_kwargs = cfg.get('evaluation', {}).copy()
                # hard-code way to remove EvalHook args
                for key in [
                        'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                        'rule'
                ]:
                    eval_kwargs.pop(key, None)
                eval_kwargs.update(dict(metric=args.eval, **kwargs))

                print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    main()
