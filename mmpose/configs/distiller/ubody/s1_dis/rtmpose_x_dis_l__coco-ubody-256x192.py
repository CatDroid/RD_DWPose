_base_ = ['../../../wholebody_2d_keypoint/rtmpose/ubody/rtmpose-l_8xb64-270e_coco-ubody-wholebody-256x192.py']

# model settings
find_unused_parameters = False

# 配置是否使用这个loss
# config settings
fea = True
logit = True

# method details
model = dict(
    _delete_ = True,
    type='PoseEstimatorDistiller',
    teacher_pretrained = 'work_dirs/rtmpose-x_8xb64-270e_coco-ubody-wholebody-256x192/rtm-x_ucoco.pth',
    teacher_cfg = 'configs/wholebody_2d_keypoint/rtmpose/ubody/rtmpose-x_8xb64-270e_coco-ubody-wholebody-256x192.py',
    # 这个配置类原始的loss
    student_cfg = 'configs/wholebody_2d_keypoint/rtmpose/ubody/rtmpose-l_8xb64-270e_coco-ubody-wholebody-256x192.py',
    distill_cfg = [dict(methods=[dict(type='FeaLoss',
                                       # type是Loss的python类型 
                                       # FeaLoss 是最后输出的loss
                                       name='loss_fea',
                                       # 是否使用loss
                                       use_this = fea,
                                       student_channels = 1024,
                                       teacher_channels = 1280,
                                       # alpha因子  调节loss的数量级
                                       alpha_fea=0.00007,
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='KDLoss',
                                       # KDLoss 是最后输出的loss
                                       name='loss_logit',
                                       use_this = logit,
                                       # beta因子  KL散度 ?
                                       weight = 0.1,
                                       )
                                ]
                        ),
                    ],
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
)
optim_wrapper = dict(
    clip_grad=dict(max_norm=1., norm_type=2))