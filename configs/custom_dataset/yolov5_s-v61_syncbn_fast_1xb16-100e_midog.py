_base_ = '../yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'

max_epochs = 200
# annotations = 'Cannotations'
# img_dir = 'CtoAimages/'
# data_root = '/data1/shitianlei/openmmlab/mmyolo/data/NormalToAdata/vahadane/'
# data_root = '/root/workspace/mmyolo/data/cat/'  # Docker

annotations = 'annotations'
img_dir = 'images/'
data_root = 'data/MIDOG-A/'




work_dir = 'MIDOG/work_dirs/yolov5_s_MIDOG_A_unnormal'
# load_from ='MIDOG/work_dirs/yolov5_s-v61_syncbn_fast_1*b32-100e_midog/best_coco/bbox_mAP_epoch_62.pth'
load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth'  # noqa

train_batch_size_per_gpu = 16
train_num_workers = 2

save_epoch_intervals = 2

# base_lr_default * (your_bs / default_bs)
base_lr = _base_.base_lr / 8

anchors = [[[11, 47], [62, 63], [107, 163]], 
            [[193, 213], [211, 265], [449, 352]], 
            [[384, 501], [429, 551], [610, 666]]
            ]


class_name = ('mitotic figure', 'not mitotic figure' )
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(220, 20, 60),(60,20,220)])

train_cfg = dict(
    max_epochs=max_epochs, val_begin=20, val_interval=save_epoch_intervals)

model = dict(
    bbox_head=dict(
        head_module=dict(num_classes=num_classes),
        prior_generator=dict(base_sizes=anchors),
        loss_cls=dict(loss_weight=0.5 *
                      (num_classes / 80 * 3 / _base_.num_det_layers))))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        _delete_=True,
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=_base_.dataset_type,
            data_root=data_root,
            metainfo=metainfo,
            ann_file='annotations/patch_train.json',
            data_prefix=dict(img='images/'),
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            pipeline=_base_.train_pipeline)))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file=f'{annotations}/patch_val.json',
        data_prefix=dict(img=img_dir)))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=f'{data_root}/{annotations}/patch_val.json')
test_evaluator = val_evaluator

optim_wrapper = dict(optimizer=dict(lr=base_lr))

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_epoch_intervals,
        max_keep_ckpts=5,
        save_best='auto'),
    param_scheduler=dict(max_epochs=max_epochs),
    logger=dict(type='LoggerHook', interval=1))

#visualizer = dict(vis_backends=[dict(type='LocalVisBackend'), dict(type='WandbVisBackend')])