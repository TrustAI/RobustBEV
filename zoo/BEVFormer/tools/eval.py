# ---------------------------------------------
# ---------------------------------------------
#  Basd on ./test.py 
# ---------------------------------------------
#  Modified by Fu Wang (fw377@exeter.ac.uk) 
# ---------------------------------------------

import argparse
import os
import sys
sys.path.append("./")
sys.path.append("../../")
from verify_utils import (LowBoundedDIRECT_potential, 
                          CentreDistance,
                          get_task_bound)

import copy
import time
import warnings
import pickle
import numpy as np
import torch

from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor

from torchvision.utils import save_image


def get_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
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
    parser.add_argument('--perturb-type', 
                        choices=["optical", "geometry", "motion"],
                        default="optical",
                        help="define the perturbation type")
    # optical
    parser.add_argument('--hue', default=0.0, type=float,
        help='range should be in [-PI, PI], while 0 means no shift '
        'control: [ - hue * PI, hue * PI]')
    parser.add_argument('--saturation', default=0.0, type=float,
        help='range should be in [0, 2], while 1 means no shift '
        'control: [ 1 - saturation, 1 + saturation]')
    parser.add_argument('--contrast', default=0.0, type=float,
        help='range should be in [0, 2], while 1 means no shift '
        'control: [ 1 - contrast, 1 + contrast]')
    parser.add_argument('--bright', default=0.0, type=float,
        help='range should be in [-1, 1], while 0 means no shift '
        'control: [-bright, bright')
    # geometry
    parser.add_argument('--shift', default=0, type=float,
           help='range should be in [-1, 1], while 0 means no shift '
            'control: [ -shift, shift]')
    parser.add_argument('--scale', default=0.0, type=float,
           help='range should be in [0,2], while 1 means no shift '
            'control: [1-scale, 1+scale]')
    # motion blur
    parser.add_argument('--kernal-size', default=0, type=int,
           help='kernal size of motion blur')

    # OPTIMISATION
    parser.add_argument('--max-evaluation', default=500, type=int)
    parser.add_argument('--max-deep', default=10, type=int)
    parser.add_argument('--quant', default=3, type=int)
    parser.add_argument('--max-iteration', default=50, type=int)
    parser.add_argument('--tolerance', default=1e-4, type=float)
    parser.add_argument('--distance-threshold', default=1.5, type=float)
    parser.add_argument('--overshoot', default=1.5, type=float)
    parser.add_argument('--save-result', action='store_true', 
                        help='save predicions on perturbed frames')
    parser.add_argument('--save-name', 
                        type=str, help="naming the saved result")
    parser.add_argument('--debug', action='store_true', 
                        help='save predicions on perturbed frames')
    parser.add_argument('--resume', action='store_true', 
                        help='save predicions on perturbed frames')

    parser.add_argument(
        '--launcher',
        choices=['none'],
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

def get_frame(save_name):
    parts = save_name.split('_')
    start_fm = parts[-2]
    end_fm = parts[-1]
    return int(start_fm), int(end_fm)

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

    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    if 'vovnet' in args.config:
        using_vov = True
    else: using_vov = False

    if args.resume:
        start_frame, end_frame = get_frame(args.save_name)
    else: start_frame, end_frame = 1, 81

###################################################
#               Evaluation
###################################################
    bound, verificaiton_problem = get_task_bound(args)
    query_model = copy.deepcopy(model)
    pertubed_results = []
    pertubation = []
    metric = CentreDistance(dataset)

    sample_token = None
    counter = 0
    for i, data in enumerate(data_loader):
        scene_token = data['img_metas'][0].data[0][0]['scene_token']
        if sample_token != scene_token:
            counter = 0
            sample_token = scene_token
        else:
            counter += 1

        with torch.no_grad():
            ori_result = model(return_loss=False, rescale=True, **data)
        ori_matches = metric.box_match(ori_result, i, 
                                    dist_th=args.distance_threshold)
        # Move to the next frame if no match
        if len(ori_matches) == 0: 
            print(f"Frame {i+1} -- No match, continue")
            continue

        ori_dist = metric.query(ori_result, i, 
                                dist_th=args.distance_threshold, 
                                overshoot=1.,
                                negative=False)

        if counter in [0, 20]:
            # Define  problem
            task = verificaiton_problem(query_model,
                                i,
                                data,
                                ori_matches,
                                args.distance_threshold,
                                args.overshoot,
                                metric.query,
                                using_vov)
            object_func = task.set_problem()

            # Define optimiser
            direct_solver = LowBoundedDIRECT_potential(
                                        object_func,len(bound),
                                        bound, 
                                        args.max_iteration, 
                                        args.max_deep, 
                                        args.max_evaluation, 
                                        args.tolerance,
                                        args.quant,
                                        debug=False)
            
            # Verification start
            start_time = time.time()
            direct_solver.solve()
            end_time = time.time()
            time_cost = (end_time-start_time)/60 # time unit: min

            pertubation.append(direct_solver.optimal_result())
            cur_ptb = direct_solver.optimal_result()
        else:
            cur_ptb = pertubation[-1]

        # Re-implement the found  perturbation
        if args.perturb_type == 'motion':
            optimal = np.reshape(
                            cur_ptb, (task.nb_camera, -1))
            tmp_img = task.functional_perturb(
                                task.rgb_images, optimal)
        else:
            optimal = np.reshape(cur_ptb, 
                        (-1,len(bound)//task.nb_camera))
            
            tmp_img = torch.zeros_like(task.rgb_images)
            for kk in range(task.nb_camera):
                min_max_v = task.min_max_vaules[kk]
                tmp_img[kk] = task.functional_perturb(
                                    task.rgb_images[kk],
                                    optimal[kk],min_max_v)

        tmp_data = task.replace_img_tensor(copy.deepcopy(data), tmp_img.unsqueeze(0))

        # Check and record the model's prediction on the perturbed frame
        with torch.no_grad():
            ptd_result = model(return_loss=False, rescale=True, **tmp_data)
        dist = metric.query(ptd_result, i,
                            dist_th=args.distance_threshold, 
                            ori_match=ori_matches,
                            overshoot=1.,
                            negative=False)

        matches = metric.box_match(ptd_result, i, 
                                   dist_th=args.distance_threshold)

        # Report
        nb_gt_boxes = len(metric.gt_boxes[dataset.data_infos[i]['token']])
        print(f"Frame {i+1} -- match ({len(ori_matches)} -> {len(matches)})/{nb_gt_boxes}", 
                end='')
        print(f", distance: {ori_dist:.3f} -> {dist:.3f}, time: {time_cost:.2f} min")
        pertubed_results.extend(ptd_result)

        if args.save_result:
            with open(f"results/{args.save_name}_result.pkl", "wb") as f:
                pickle.dump(pertubed_results, f)
            with open(f"results/{args.save_name}_ptb.pkl", "wb") as f:
                pickle.dump(pertubation, f)

    ###################################################
    # CHECK
    kwargs = {} if args.eval_options is None else args.eval_options
    eval_kwargs = cfg.get('evaluation', {}).copy()
    # hard-code way to remove EvalHook args
    for key in [
            'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
            'rule'
    ]:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric=args.eval, **kwargs))
    print(dataset.evaluate(pertubed_results, **eval_kwargs))

    ###################################################





if __name__ == '__main__':
    main()