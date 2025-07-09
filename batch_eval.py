# Fixed parallel_benchmark_a100.py
import os
import subprocess
import pandas as pd
import time
import psutil

def evaluate_model(model_name):
    """Evaluate single model with A100 optimizations"""
    start_time = time.time()
    ckpt_dir = f"/mnt/data/CVLface/cvlface/pretrained_models/recognition/{model_name}/pretrained_model"
    
    if not os.path.exists(ckpt_dir):
        return {'model': model_name, 'status': 'MISSING', 'time': 0, 'tpr': None}
    
    # Change to eval directory (CRITICAL FIX)
    original_dir = os.getcwd()
    eval_dir = "/mnt/data/CVLface/cvlface/research/recognition/code/run_v1"
    
    try:
        os.chdir(eval_dir)
        
        # Make ckpt_dir relative to eval directory
        relative_ckpt = os.path.relpath(ckpt_dir, eval_dir)
        
        cmd = [
            'uv', 'run', 'python', 'eval.py',
            '--num_gpu', '1',
            '--precision', '16-mixed',  # A100 optimization
            '--eval_config_name', 'ijbc',  # Fixed: use existing config
            '--pipeline_name', 'default',
            '--ckpt_dir', relative_ckpt
        ]
        
        # A100-optimized + PyTorch 2.6 fix
        env = os.environ.copy()
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:2048,expandable_segments:True'
        env['CUDA_VISIBLE_DEVICES'] = '0'
        env['OMP_NUM_THREADS'] = '6'
        
        print(f"  📂 Model path: {relative_ckpt}")
        print(f"  🔧 Running evaluation...")
        
        subprocess.run(cmd, check=True, env=env)
        
        # FIXED: Use eval_final.csv which contains detailed IJB-C metrics
        result_path = f"/mnt/data/CVLface/cvlface/research/recognition/experiments/pretrained_models/eval_pretrained_model/result/eval_final.csv"
        
        print(f"  📊 Looking for results at: {result_path}")
        
        if os.path.exists(result_path):
            df = pd.read_csv(result_path, index_col=0)
            
            # Look for IJB-C TPR@FPR=1e-4 metric in detailed results
            target_metric = 'IJBC_gt_aligned/Norm:True_Det:True_tpr_at_fpr_0.0001'
            if target_metric in df.index:
                tpr = df.loc[target_metric, 'val']  # Use the specific metric
                elapsed = time.time() - start_time
                return {'model': model_name, 'status': 'SUCCESS', 'time': elapsed, 'tpr': float(tpr)}
            else:
                # Fallback: look for any IJB metric with tpr_at_fpr_0.0001
                ijbc_keys = [k for k in df.index if 'IJB' in k and 'tpr_at_fpr_0.0001' in k]
                if ijbc_keys:
                    tpr = df.loc[ijbc_keys[0], 'val']  # Use first match
                    elapsed = time.time() - start_time
                    return {'model': model_name, 'status': 'SUCCESS', 'time': elapsed, 'tpr': float(tpr)}
                else:
                    return {'model': model_name, 'status': 'NO_IJB_METRIC', 'time': time.time() - start_time, 'tpr': None}
        else:
            return {'model': model_name, 'status': 'NO_RESULTS', 'time': time.time() - start_time, 'tpr': None}
            
    except subprocess.CalledProcessError as e:
        return {'model': model_name, 'status': 'FAILED', 'time': time.time() - start_time, 'tpr': None}
    finally:
        os.chdir(original_dir)  # Always restore directory

def main():
    # Complete AdaFace model evaluation suite for A100 (ResNet + ViT)
    models = [
        "cvlface_adaface_ir101_webface12m",
        "cvlface_adaface_ir101_ms1mv2",
        "cvlface_adaface_ir101_ms1mv3",
        "cvlface_adaface_ir101_webface4m",
        "cvlface_adaface_vit_base_kprpe_webface12m",
        "cvlface_adaface_vit_base_kprpe_webface4m",
        "cvlface_adaface_vit_base_webface4m",
    ]
    
    print(f"🚀 Starting A100 AdaFace IJB-C Benchmark")
    print(f"📊 Models: {len(models)}")
    print(f"💾 Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    print(f"🎮 GPU Memory: 40GB A100")
    print("=" * 80)
    
    results = []
    for i, model in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] 🚀 Evaluating {model}...")
        result = evaluate_model(model)
        results.append(result)
        
        status_emoji = "✅" if result['status'] == 'SUCCESS' else "❌"
        tpr_display = f"{result['tpr']:.4f}" if result['tpr'] is not None else "N/A"
        print(f"  {status_emoji} Status: {result['status']}")
        print(f"  ⏱️  Time: {result['time']:.1f}s")
        print(f"  📈 TPR@FPR=1e-4: {tpr_display}")
        
        # A100 memory optimization
        if result['status'] == 'SUCCESS':
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
    
    # Fixed: Pandas compatibility
    results_df = pd.DataFrame(results)
    try:
        results_df = results_df.sort_values('tpr', ascending=False, na_last=True)
    except TypeError:
        # Fallback for older pandas
        results_df['tpr_fillna'] = results_df['tpr'].fillna(-1)
        results_df = results_df.sort_values('tpr_fillna', ascending=False)
        results_df = results_df.drop('tpr_fillna', axis=1)
    
    print(f"\n{'='*80}")
    print("🏆 A100 ADAFACE IJB-C BENCHMARK RESULTS")
    print(f"{'='*80}")
    print(results_df.to_string(index=False))
    
    # Save with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(f'a100_adaface_ijbc_benchmark_{timestamp}.csv', index=False)
    
    # Performance summary
    successful = results_df[results_df['status'] == 'SUCCESS']
    if len(successful) > 0:
        avg_time = successful['time'].mean()
        best_model = successful.iloc[0]
        print(f"\n📊 PERFORMANCE SUMMARY:")
        print(f"🥇 Best Model: {best_model['model']} (TPR: {best_model['tpr']:.4f})")
        print(f"⏱️  Average Eval Time: {avg_time:.1f}s")
        print(f"✅ Success Rate: {len(successful)}/{len(models)} ({100*len(successful)/len(models):.1f}%)")
    else:
        print(f"\n⚠️  No successful evaluations. Check:")
        print("   1. Models downloaded to cvlface/pretrained_models/recognition/")
        print("   2. IJB dataset at $DATA_ROOT/facerec_val/IJBC_gt_aligned/")
        print("   3. Environment variables set correctly")

if __name__ == "__main__":
    main()
