import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)
import os, sys
sys.path.append(os.path.join(root))
import numpy as np
np.bool = np.bool_  # fix bug for mxnet 1.9.1
np.object = np.object_

import pandas as pd
from models import get_model
from aligners import get_aligner
from evaluations import get_evaluator_by_name
from lightning.fabric.loggers import CSVLogger
from pipelines import pipeline_from_name
from lightning.pytorch.loggers import WandbLogger
from general_utils.config_utils import load_config
from evaluations import summary
from lightning.fabric import Fabric
from functools import partial
from fabric.fabric import setup_dataloader_from_dataset

import lovely_tensors as lt
lt.monkey_patch()


def get_runname_and_task(ckpt_dir):
    if 'pretrained_models' in ckpt_dir:
        runname = ckpt_dir.split('/')[-1]
        code_task = os.path.abspath(__file__).split('/')[-2]
        save_dir_task = 'pretrained_models'
    else:
        runname = ckpt_dir.split('/')[-3]
        code_task = os.path.abspath(__file__).split('/')[-2]
        save_dir_task = code_task
    return runname, save_dir_task, code_task



if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpu', type=int, default=1)
    parser.add_argument('--precision', type=str, default='32-true')
    parser.add_argument('--eval_config_name', type=str, default='quick')
    parser.add_argument('--pipeline_name', type=str, default='default')
    parser.add_argument('--ckpt_dir', type=str, default="../../../../pretrained_models/recognition/adaface_ir101_webface12m")
    args = parser.parse_args()

    # setup output dir
    runname, save_dir_task, task = get_runname_and_task(args.ckpt_dir)
    eval_config = load_config(f'evaluations/configs/{args.eval_config_name}.yaml')
    output_dir = os.path.join(root, 'research/recognition/experiments', save_dir_task, 'eval_' + runname)
    os.makedirs(output_dir, exist_ok=True)

    # load model
    model_config = load_config(os.path.join(args.ckpt_dir, 'model.yaml'))
    model = get_model(model_config, task)
    model.load_state_dict_from_path(os.path.join(args.ckpt_dir, 'model.pt'))
    train_transform = model.make_train_transform()
    test_transform = model.make_test_transform()

    # maybe load aligner
    if os.path.exists(os.path.join(args.ckpt_dir, 'aligner.yaml')):
        aligner_config = load_config(os.path.join(args.ckpt_dir, 'aligner.yaml'))
        aligner_config.start_from = os.path.join(args.ckpt_dir, 'aligner.pt')
        aligner = get_aligner(aligner_config)
    else:
        aligner_config = load_config(os.path.join(root, 'research/recognition/code/', task, f'aligners/configs/none.yaml'))
        aligner = get_aligner(aligner_config)


    # load pipeline
    if args.pipeline_name == 'default':
        full_config_path = os.path.join(args.ckpt_dir, 'config.yaml')
        assert os.path.isfile(full_config_path), f"config.yaml not found at {full_config_path}, try with pipeline name"
        pipeline_name = load_config(full_config_path).pipelines.eval_pipeline_name
    else:
        pipeline_name = args.pipeline_name

    # launch fabric with enhanced W&B logging
    csv_logger = CSVLogger(root_dir=output_dir, flush_logs_every_n_steps=1)
    wandb_logger = WandbLogger(project=task,
                               entity=os.getenv('WANDB_TEAM'),
                               save_dir=output_dir,
                               name=f"{runname}_{args.eval_config_name}",  # Better run names
                               log_model=False,
                               tags=[runname, args.eval_config_name, "face-recognition", "ijbc-benchmark"])
    fabric = Fabric(precision=args.precision,  # Use actual precision arg
                    accelerator="auto",
                    strategy="ddp",
                    devices=args.num_gpu,
                    loggers=[csv_logger, wandb_logger],
                    )

    if args.num_gpu == 1:
        fabric.launch()
    print(f"Fabric launched with {args.num_gpu} GPUS and {args.precision}")
    fabric.setup_dataloader_from_dataset = partial(setup_dataloader_from_dataset, fabric=fabric, seed=2048)

    # Log model configuration and metadata
    if fabric.local_rank == 0:
        import time
        eval_start_time = time.time()
        
        # Log model metadata (with safe config access)
        try:
            model_arch = model_config.model.name if hasattr(model_config, 'model') and hasattr(model_config.model, 'name') else 'unknown'
        except:
            model_arch = 'unknown'
            
        fabric.log_dict({
            "model/name": runname,
            "model/architecture": model_arch,
            "model/precision": args.precision,
            "eval/dataset": args.eval_config_name,
            "eval/pipeline": pipeline_name,
            "system/num_gpus": args.num_gpu,
            "eval/start_timestamp": eval_start_time
        })
        
        # Log model config as artifact (with safe serialization)
        if hasattr(wandb_logger.experiment, 'config'):
            try:
                model_config_dict = dict(model_config) if hasattr(model_config, '__iter__') else str(model_config)
                eval_config_dict = dict(eval_config) if hasattr(eval_config, '__iter__') else str(eval_config)
            except:
                model_config_dict = str(model_config)
                eval_config_dict = str(eval_config)
                
            wandb_logger.experiment.config.update({
                "model_config": model_config_dict,
                "eval_config": eval_config_dict,
                "checkpoint_path": args.ckpt_dir
            })

    # prepare accelerator
    model = fabric.setup(model)
    if aligner.has_trainable_params():
        aligner = fabric.setup(aligner)

    # make inference pipe (after accelerator setup)
    eval_pipeline = pipeline_from_name(pipeline_name, model, aligner)
    eval_pipeline.integrity_check(dataset_color_space='RGB')

    # evaluation callbacks
    evaluators = []
    for name, info in eval_config.per_epoch_evaluations.items():
        eval_data_path = os.path.join(eval_config.data_root, info.path)
        eval_type = info.evaluation_type
        eval_batch_size = info.batch_size
        eval_num_workers = info.num_workers
        evaluator = get_evaluator_by_name(eval_type=eval_type, name=name, eval_data_path=eval_data_path,
                                          transform=eval_pipeline.make_test_transform(),
                                          fabric=fabric, batch_size=eval_batch_size, num_workers=eval_num_workers)
        evaluator.integrity_check(info.color_space, eval_pipeline.color_space)
        evaluators.append(evaluator)

    # Evaluation with progress tracking
    print('Evaluation Started')
    all_result = {}
    
    for eval_idx, evaluator in enumerate(evaluators):
        if fabric.local_rank == 0:
            print(f"Evaluating {evaluator.name}")
            eval_step_start = time.time()
            
            # Log evaluation progress
            fabric.log_dict({
                "eval/current_evaluator": evaluator.name,
                "eval/progress": eval_idx / len(evaluators),
                "eval/step": eval_idx
            })
            
        result = evaluator.evaluate(eval_pipeline, epoch=0, step=eval_idx, n_images_seen=0)
        
        if fabric.local_rank == 0:
            eval_step_time = time.time() - eval_step_start
            print(f"{evaluator.name}")
            print(result)
            
            # Log individual evaluator results and timing
            fabric.log_dict({
                f"eval/{evaluator.name}/duration_seconds": eval_step_time,
                **{f"eval/{evaluator.name}/{k}": v for k, v in result.items() if isinstance(v, (int, float))}
            })
            
        all_result.update({evaluator.name + "/" + k: v for k, v in result.items()})

    if fabric.local_rank == 0:
        # Calculate total evaluation time
        total_eval_time = time.time() - eval_start_time
        
        os.makedirs(os.path.join(output_dir, 'result'), exist_ok=True)
        save_result = pd.DataFrame(pd.Series(all_result), columns=['val'])
        save_result.to_csv(os.path.join(output_dir, f'result/eval_final.csv'))
        mean, summary_dict = summary(save_result, epoch=0, step=len(evaluators), n_images_seen=0)
        
        # Enhanced final logging with timing and model info
        final_log_dict = {
            **summary_dict,
            "eval/total_duration_seconds": total_eval_time,
            "eval/total_duration_minutes": total_eval_time / 60,
            "eval/completed_evaluators": len(evaluators),
            "eval/end_timestamp": time.time(),
            "model/checkpoint_dir": os.path.basename(args.ckpt_dir)
        }
        
        fabric.log_dict(final_log_dict)
        summary_result = pd.DataFrame(pd.Series(summary_dict), columns=['val'])
        summary_result.to_csv(os.path.join(output_dir, f'result/eval_summary_final.csv'))
        
        print(f"Evaluation completed in {total_eval_time:.1f}s ({total_eval_time/60:.1f} min)")

    print('Evaluation Finished')
