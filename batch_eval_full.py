#!/usr/bin/env python3
"""
Comprehensive AdaFace Batch Evaluation with Enhanced Metrics
A100 optimized evaluation across all AdaFace models with comprehensive metrics:
- 4.1 Identification Accuracy (Top-1, Top-5)
- 4.2 Verification Metrics (ROC AUC, EER, F1, Precision, Recall)
- 4.3 Speed & Throughput (FPS, Real-time capability)
- 4.4 Memory & Resource Usage (GPU VRAM, Model Size, CPU)
"""

import os
import subprocess
import pandas as pd
import time
import psutil
import numpy as np
import logging
from datetime import datetime

def setup_logging():
    """Setup logging to capture all output to datetime-stamped log file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.path.dirname(__file__), '.logs')
    log_file = os.path.join(log_dir, f"batch_eval_full_{timestamp}.log")
    
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Batch evaluation logging initialized. Log file: {log_file}")
    return logger, log_file

def evaluate_model_comprehensive(model_name):
    """Evaluate single model with comprehensive metrics using enhanced eval script"""
    start_time = time.time()
    ckpt_dir = f"/mnt/data/CVLface/cvlface/pretrained_models/recognition/{model_name}/pretrained_model"
    
    if not os.path.exists(ckpt_dir):
        return {
            'model': model_name,
            'status': 'MISSING',
            'time': 0,
            'ijbc_tpr': None,
            'roc_auc': None,
            'eer': None,
            'f1_score': None,
            'fps': None,
            'gpu_memory_gb': None,
            'model_size_mb': None,
            'model_size_gb': None,
            'top_10_accuracy': None
        }
    
    # Change to eval directory (CRITICAL for CVLface)
    original_dir = os.getcwd()
    eval_dir = "/mnt/data/CVLface/cvlface/research/recognition/code/run_v1"
    
    try:
        os.chdir(eval_dir)
        
        # Use absolute path directly (fixes path resolution issues)
        absolute_ckpt = os.path.abspath(ckpt_dir)
        
        cmd = [
            'uv', 'run', 'python', 'eval_full.py',  # Use our enhanced script (now in correct directory)
            '--num_gpu', '1',
            '--precision', '16-mixed',  # A100 optimization
            '--eval_config_name', 'ijbc',
            '--pipeline_name', 'default',
            '--ckpt_dir', absolute_ckpt
        ]
        
        # A100-optimized environment
        env = os.environ.copy()
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:2048,expandable_segments:True'
        env['CUDA_VISIBLE_DEVICES'] = '0'
        env['OMP_NUM_THREADS'] = '6'
        env['TORCH_CUDNN_V8_API_ENABLED'] = '1'  # A100 optimization
        
        logger = logging.getLogger(__name__)
        logger.info(f"  📂 Model path: {absolute_ckpt}")
        logger.info(f"  🔧 Running comprehensive evaluation...")
        
        subprocess.run(cmd, check=True, env=env)
        
        # Parse comprehensive results
        result_path = f"/mnt/data/CVLface/cvlface/research/recognition/experiments/pretrained_models/eval_pretrained_model/result/eval_full.csv"
        
        logger.info(f"  📊 Looking for comprehensive results at: {result_path}")
        
        if os.path.exists(result_path):
            df = pd.read_csv(result_path, index_col=0)
            
            # Extract comprehensive metrics
            result = {
                'model': model_name,
                'status': 'SUCCESS',
                'time': time.time() - start_time
            }
            
            # Original IJB-C metrics
            ijbc_keys = [k for k in df.index if 'IJBC' in k and 'tpr_at_fpr_0.0001' in k]
            if ijbc_keys:
                result['ijbc_tpr'] = float(df.loc[ijbc_keys[0], 'val'])
            else:
                result['ijbc_tpr'] = None
            
            # Enhanced verification metrics
            for metric in ['roc_auc', 'eer', 'f1_score', 'precision', 'recall']:
                key = f'comprehensive/verification/{metric}'
                if key in df.index:
                    result[metric] = float(df.loc[key, 'val'])
                else:
                    result[metric] = None
            
            # Speed metrics
            for metric in ['fps', 'ms_per_image', 'is_realtime']:
                key = f'comprehensive/speed/{metric}'
                if key in df.index:
                    result[metric] = float(df.loc[key, 'val']) if metric != 'is_realtime' else bool(df.loc[key, 'val'])
                else:
                    result[metric] = None
            
            # Resource metrics
            for metric in ['gpu_memory_peak_gb', 'cpu_usage_avg_percent', 'total_duration_minutes']:
                key = f'comprehensive/resources/{metric}'
                if key in df.index:
                    result[metric] = float(df.loc[key, 'val'])
                else:
                    result[metric] = None
                    
            # Identification metrics (including missing top_10_accuracy)
            for metric in ['top_1_accuracy', 'top_5_accuracy', 'top_10_accuracy']:
                key = f'comprehensive/identification/{metric}'
                if key in df.index:
                    result[metric] = float(df.loc[key, 'val'])
                else:
                    result[metric] = None
            
            # Model size metrics (including missing model_size_gb)
            model_size_mb_key = [k for k in df.index if 'model_size_mb' in k]
            if model_size_mb_key:
                result['model_size_mb'] = float(df.loc[model_size_mb_key[0], 'val'])
            else:
                result['model_size_mb'] = None
                
            model_size_gb_key = [k for k in df.index if 'model_size_gb' in k]
            if model_size_gb_key:
                result['model_size_gb'] = float(df.loc[model_size_gb_key[0], 'val'])
            else:
                result['model_size_gb'] = None
                
            return result
            
        else:
            return {
                'model': model_name,
                'status': 'NO_RESULTS',
                'time': time.time() - start_time,
                **{k: None for k in ['ijbc_tpr', 'roc_auc', 'eer', 'f1_score', 'fps', 'gpu_memory_gb', 'model_size_mb', 'model_size_gb', 'top_10_accuracy']}
            }
            
    except subprocess.CalledProcessError as e:
        return {
            'model': model_name,
            'status': 'FAILED',
            'time': time.time() - start_time,
            **{k: None for k in ['ijbc_tpr', 'roc_auc', 'eer', 'f1_score', 'fps', 'gpu_memory_gb', 'model_size_mb', 'model_size_gb', 'top_10_accuracy']}
        }
    finally:
        os.chdir(original_dir)


def main():
    """Run comprehensive benchmark across all AdaFace models"""
    # Initialize logging
    logger, log_file = setup_logging()
    
    # Complete AdaFace model suite (same as original batch_eval.py)
    models = [
        "cvlface_adaface_ir101_webface12m",
        "cvlface_adaface_ir101_ms1mv2",
        "cvlface_adaface_ir101_ms1mv3",
        "cvlface_adaface_ir101_webface4m",
        "cvlface_adaface_vit_base_kprpe_webface12m",
        "cvlface_adaface_vit_base_kprpe_webface4m",
        "cvlface_adaface_vit_base_webface4m",
    ]
    
    logger.info(f"🚀 COMPREHENSIVE ADAFACE BENCHMARK - A100 OPTIMIZED")
    logger.info(f"📊 Models: {len(models)}")
    logger.info(f"💾 Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    logger.info(f"🎮 GPU: 40GB A100")
    logger.info(f"📈 Metrics: IJB-C + ROC AUC + EER + F1 + FPS + Resources")
    logger.info(f"📝 Log file: {log_file}")
    logger.info("=" * 80)
    
    results = []
    benchmark_start = time.time()
    
    for i, model in enumerate(models, 1):
        logger.info(f"\n[{i}/{len(models)}] 🚀 Comprehensive Evaluation: {model}")
        logger.info("-" * 60)
        
        result = evaluate_model_comprehensive(model)
        results.append(result)
        
        # Display comprehensive results
        status_emoji = "✅" if result['status'] == 'SUCCESS' else "❌"
        logger.info(f"  {status_emoji} Status: {result['status']}")
        logger.info(f"  ⏱️  Time: {result['time']:.1f}s")
        
        if result['status'] == 'SUCCESS':
            ijbc_tpr = result.get('ijbc_tpr', 0)
            roc_auc = result.get('roc_auc', 0)
            eer = result.get('eer', 0)
            f1_score = result.get('f1_score', 0)
            fps = result.get('fps', 0)
            gpu_mem = result.get('gpu_memory_peak_gb', 0)
            model_size = result.get('model_size_mb', 0)
            top1_acc = result.get('top_1_accuracy', 0)
            
            logger.info(f"  📊 IJB-C TPR@FPR=1e-4: {ijbc_tpr:.3f}%" if ijbc_tpr else "  📊 IJB-C TPR: N/A")
            logger.info(f"  📈 ROC AUC: {roc_auc:.4f}" if roc_auc else "  📈 ROC AUC: N/A")
            logger.info(f"  ⚖️  EER: {eer:.2f}%" if eer else "  ⚖️  EER: N/A")
            logger.info(f"  🎯 F1 Score: {f1_score:.2f}%" if f1_score else "  🎯 F1 Score: N/A")
            logger.info(f"  🏆 Top-1 Acc: {top1_acc:.2f}%" if top1_acc else "  🏆 Top-1 Acc: N/A")
            logger.info(f"  ⚡ FPS: {fps:.1f}" if fps else "  ⚡ FPS: N/A")
            logger.info(f"  🚀 Real-time: {'✅' if result.get('is_realtime') else '❌'}")
            logger.info(f"  💾 GPU Memory: {gpu_mem:.2f} GB" if gpu_mem else "  💾 GPU Memory: N/A")
            logger.info(f"  📦 Model Size: {model_size:.1f} MB" if model_size else "  📦 Model Size: N/A")
        
        # A100 memory optimization between models
        if result['status'] == 'SUCCESS':
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
        # Progress indicator
        remaining = len(models) - i
        if remaining > 0:
            elapsed = time.time() - benchmark_start
            avg_time = elapsed / i
            estimated_remaining = avg_time * remaining
            logger.info(f"  📅 Progress: {i}/{len(models)} | ETA: {estimated_remaining/60:.1f} min")
    
    # Create comprehensive results DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by IJB-C performance
    try:
        results_df = results_df.sort_values('ijbc_tpr', ascending=False, na_last=True)
    except:
        # Fallback sorting
        results_df['tpr_sort'] = results_df['ijbc_tpr'].fillna(-1)
        results_df = results_df.sort_values('tpr_sort', ascending=False)
        results_df = results_df.drop('tpr_sort', axis=1)
    
    total_benchmark_time = time.time() - benchmark_start
    
    logger.info(f"\n{'='*100}")
    logger.info("🏆 COMPREHENSIVE ADAFACE BENCHMARK RESULTS")
    logger.info(f"{'='*100}")
    
    # Display summary table
    display_df = results_df.copy()
    
    # Format for display
    for col in ['ijbc_tpr', 'roc_auc', 'eer', 'f1_score', 'top_1_accuracy']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "N/A")
    
    for col in ['fps', 'gpu_memory_peak_gb', 'model_size_mb']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else "N/A")
    
    # Show key columns
    key_columns = ['model', 'status', 'ijbc_tpr', 'roc_auc', 'eer', 'f1_score', 'fps', 'gpu_memory_peak_gb']
    available_columns = [col for col in key_columns if col in display_df.columns]
    
    logger.info("\n" + display_df[available_columns].to_string(index=False))
    
    # Save comprehensive results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_filename = f'comprehensive_adaface_benchmark_{timestamp}.csv'
    results_df.to_csv(results_filename, index=False)
    
    # Performance analysis
    successful = results_df[results_df['status'] == 'SUCCESS']
    
    logger.info(f"\n{'='*80}")
    logger.info("📊 COMPREHENSIVE PERFORMANCE ANALYSIS")
    logger.info(f"{'='*80}")
    
    if len(successful) > 0:
        logger.info(f"✅ Success Rate: {len(successful)}/{len(models)} ({100*len(successful)/len(models):.1f}%)")
        logger.info(f"⏱️  Total Benchmark Time: {total_benchmark_time/60:.1f} minutes")
        logger.info(f"📈 Average Eval Time: {successful['time'].mean():.1f}s per model")
        
        # Best performers
        if 'ijbc_tpr' in successful.columns and successful['ijbc_tpr'].notna().any():
            best_accuracy = successful.loc[successful['ijbc_tpr'].idxmax()]
            logger.info(f"\n🥇 BEST ACCURACY: {best_accuracy['model']}")
            logger.info(f"   📊 IJB-C TPR@FPR=1e-4: {best_accuracy['ijbc_tpr']:.3f}%")
            
        if 'fps' in successful.columns and successful['fps'].notna().any():
            best_speed = successful.loc[successful['fps'].idxmax()]
            logger.info(f"\n⚡ FASTEST MODEL: {best_speed['model']}")
            logger.info(f"   🚀 FPS: {best_speed['fps']:.1f}")
            
        if 'roc_auc' in successful.columns and successful['roc_auc'].notna().any():
            best_verification = successful.loc[successful['roc_auc'].idxmax()]
            logger.info(f"\n🎯 BEST VERIFICATION: {best_verification['model']}")
            logger.info(f"   📈 ROC AUC: {best_verification['roc_auc']:.4f}")
        
        # Performance ranges
        logger.info(f"\n📊 PERFORMANCE RANGES:")
        if 'ijbc_tpr' in successful.columns:
            tpr_values = successful['ijbc_tpr'].dropna()
            if len(tpr_values) > 0:
                logger.info(f"   IJB-C TPR: {tpr_values.min():.3f}% - {tpr_values.max():.3f}%")
                
        if 'fps' in successful.columns:
            fps_values = successful['fps'].dropna()
            if len(fps_values) > 0:
                logger.info(f"   FPS Range: {fps_values.min():.1f} - {fps_values.max():.1f}")
                logger.info(f"   Real-time Models: {successful['is_realtime'].sum() if 'is_realtime' in successful.columns else 0}")
                
        if 'gpu_memory_peak_gb' in successful.columns:
            gpu_values = successful['gpu_memory_peak_gb'].dropna()
            if len(gpu_values) > 0:
                logger.info(f"   GPU Memory: {gpu_values.min():.1f}GB - {gpu_values.max():.1f}GB")
                
    else:
        logger.info(f"❌ No successful evaluations")
        logger.info(f"   Check model paths and IJB dataset availability")
    
    logger.info(f"\n💾 Results saved to: {results_filename}")
    logger.info(f"🎉 Comprehensive benchmark completed!")
    
    return results_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Comprehensive AdaFace Benchmark')
    parser.add_argument('--models', nargs='+', help='Specific models to evaluate')
    parser.add_argument('--max_models', type=int, help='Maximum number of models to evaluate')
    args = parser.parse_args()
    
    # Run comprehensive benchmark
    results_df = main()
