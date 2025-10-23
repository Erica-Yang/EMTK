import os
from detectron2.utils import comm
from detectron2.engine import launch
from detectron2.data import MetadataCatalog
from detectron2.checkpoint import DetectionCheckpointer
from defrcn.config import get_cfg, set_global_cfg
from defrcn.evaluation import DatasetEvaluators, verify_results
from defrcn.engine import Trainer, TwoSteamTrainer, default_argument_parser, default_setup


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_global_cfg(cfg)
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    if cfg.DATASETS.TWO_STREAM:
        trainer = TwoSteamTrainer(cfg)
    else:
        trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    #查看模型参数：
    # 计算可学习参数数量
    # trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    
    # # 计算模型存储大小（包括parameters和buffers）
    # param_size = sum(p.numel() * p.element_size() for p in trainer.model.parameters())
    # buffer_size = sum(b.numel() * b.element_size() for b in trainer.model.buffers())
    # model_size_mb = (param_size + buffer_size) / (1024 ** 2)  # 转换为MB
    # print('trainable pamrams: {:.2f} M'.format(trainable_params/(1024 ** 2)))
    # print('model pamrams: {:.2f} M'.format(param_size/(1024 ** 2) ))
    
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    # args.config_file = "/root/emtk_seeds/configs/coco/base.yaml"
    # args.config_file = "/root/emtk_seeds/configs/coco/emtk_gfsod_novel_30shot_seed0.yaml"
    # args.config_file = "/root/emtk_seeds/configs/coco/emtk_gfsod_novel_1shot_seed0.yaml"
    args.config_file = "configs/coal/emtk_gfsod_novel_5shot_seed0.yaml"
    args.num_gpus = 1
    # args.num_gpus = 2
    args.eval_only = True
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
