"""
generate_unified_visualizations.py
===================================

Creates publication-quality visualizations combining results from:
- Disease models: train_comparative_models.py
- Diet models: train_diet_models.py  
- Lifestyle models: train_lifestyle_models.py

Generates:
1. Performance comparison bar charts (all models, all tasks)
2. Dataset composition pie charts
3. Model-task performance heatmaps
4. Training efficiency analysis
5. Summary tables for IEEE paper
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import os

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Configuration
DISEASE_RESULTS_FILE = 'models/comparative_results.json'
DIET_RESULTS_FILE = 'models/diet_results.json'
LIFESTYLE_RESULTS_FILE = 'models/lifestyle_results.json'
DATA_PATH = 'data/processed/healthcare_cleaned.parquet'
OUTPUT_DIR = 'figures/ieee_paper'

# Create output directory
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD ALL RESULTS
# ============================================================

def load_all_results():
    """Load results from all three training scripts"""
    
    print("\n" + "="*70)
    print("LOADING RESULTS FROM ALL TRAINING SCRIPTS")
    print("="*70)
    
    results = {
        'disease': {'models': {}},
        'diet': {'models': {}},
        'lifestyle': {'models': {}}
    }
    
    # Load disease results (from train_comparative_models.py)
    if os.path.exists(DISEASE_RESULTS_FILE):
        with open(DISEASE_RESULTS_FILE, 'r') as f:
            disease_data = json.load(f)
            results['disease'] = disease_data.get('disease', {'models': {}})
            print(f"✅ Loaded disease results: {len(results['disease']['models'])} models")
    else:
        print(f"⚠️  Disease results not found: {DISEASE_RESULTS_FILE}")
        print("   Run: python scripts/train_comparative_models.py")
    
    # Load diet results (from train_diet_models.py)
    if os.path.exists(DIET_RESULTS_FILE):
        with open(DIET_RESULTS_FILE, 'r') as f:
            diet_data = json.load(f)
            # Convert to same format as comparative results
            for model_name, metrics in diet_data.items():
                results['diet']['models'][model_name] = {
                    'accuracy': metrics['accuracy'],
                    'f1_score': metrics['f1'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'training_time_seconds': metrics['training_time']
                }
            print(f"✅ Loaded diet results: {len(results['diet']['models'])} models")
    else:
        print(f"⚠️  Diet results not found: {DIET_RESULTS_FILE}")
        print("   Run: python scripts/train_diet_models.py")
    
    # Load lifestyle results (from train_lifestyle_models.py)
    if os.path.exists(LIFESTYLE_RESULTS_FILE):
        with open(LIFESTYLE_RESULTS_FILE, 'r') as f:
            lifestyle_data = json.load(f)
            # Convert to same format
            for model_name, metrics in lifestyle_data.items():
                results['lifestyle']['models'][model_name] = {
                    'accuracy': metrics['accuracy'],
                    'f1_score': metrics['f1'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'training_time_seconds': metrics['training_time']
                }
            print(f"✅ Loaded lifestyle results: {len(results['lifestyle']['models'])} models")
    else:
        print(f"⚠️  Lifestyle results not found: {LIFESTYLE_RESULTS_FILE}")
        print("   Run: python scripts/train_lifestyle_models.py")
    
    return results


# ============================================================
# DATASET ANALYSIS
# ============================================================

def create_dataset_visualizations():
    """Create pie charts showing dataset composition"""
    
    print("\n📊 Generating dataset composition visualizations...")
    
    if not os.path.exists(DATA_PATH):
        print(f"⚠️  Data file not found: {DATA_PATH}")
        return
    
    df = pd.read_parquet(DATA_PATH)
    
    # Create figure with 3 pie charts
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Dataset Composition Analysis', fontsize=16, fontweight='bold')
    
    # 1. Disease Distribution
    disease_counts = df['disease'].value_counts()
    colors_disease = plt.cm.Set3(range(len(disease_counts)))
    
    axes[0].pie(disease_counts.values, labels=disease_counts.index, autopct='%1.1f%%',
                startangle=90, colors=colors_disease)
    axes[0].set_title('Disease Categories\n(11 classes)', fontweight='bold', pad=20)
    
    # 2. Diet Plan Distribution
    diet_counts = df['diet_plan'].value_counts()
    colors_diet = plt.cm.Pastel1(range(len(diet_counts)))
    
    axes[1].pie(diet_counts.values, labels=diet_counts.index, autopct='%1.1f%%',
                startangle=90, colors=colors_diet)
    axes[1].set_title('Diet Plan Recommendations\n(8 classes)', fontweight='bold', pad=20)
    
    # 3. Lifestyle Plan Distribution
    lifestyle_counts = df['lifestyle_plan'].value_counts()
    colors_lifestyle = plt.cm.Pastel2(range(len(lifestyle_counts)))
    
    axes[2].pie(lifestyle_counts.values, labels=lifestyle_counts.index, autopct='%1.1f%%',
                startangle=90, colors=colors_lifestyle)
    axes[2].set_title('Lifestyle Interventions\n(8 classes)', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig1_dataset_composition.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Figure 1: Dataset composition pie charts saved")
    
    # Print statistics
    print(f"\n📈 Dataset Statistics:")
    print(f"   Total records: {len(df):,}")
    print(f"   Disease classes: {df['disease'].nunique()}")
    print(f"   Diet plans: {df['diet_plan'].nunique()}")
    print(f"   Lifestyle plans: {df['lifestyle_plan'].nunique()}")


# ============================================================
# PERFORMANCE COMPARISON BAR CHARTS
# ============================================================

def create_performance_comparison_bars(results):
    """Create grouped bar charts comparing all models across metrics"""
    
    print("\n📊 Generating performance comparison bar charts...")
    
    targets = ['disease', 'diet', 'lifestyle']
    metrics = ['accuracy', 'f1_score', 'precision', 'recall']
    
    # Get all unique model names across all tasks
    all_models = set()
    for target in targets:
        all_models.update(results[target]['models'].keys())
    model_names = sorted(list(all_models))
    
    if not model_names:
        print("⚠️  No models found in results")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comparative Performance Analysis: Disease, Diet & Lifestyle Prediction', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        x = np.arange(len(model_names))
        width = 0.25
        
        # Plot bars for each target
        for i, target in enumerate(targets):
            values = []
            for model in model_names:
                if model in results[target]['models']:
                    values.append(results[target]['models'][model][metric])
                else:
                    values.append(0)  # Model not trained for this task
            
            ax.bar(x + i*width, values, width, 
                  label=target.capitalize(), alpha=0.8)
        
        # Formatting
        ax.set_xlabel('Algorithm', fontweight='bold')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontweight='bold')
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison', 
                    fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend(loc='lower right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Dynamic y-axis limits
        all_values = [v for v in values if v > 0]
        if all_values:
            y_min = max(0, min(all_values) - 0.1)
            y_max = min(1.0, max(all_values) + 0.05)
            ax.set_ylim([y_min, y_max])
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3, fontsize=7)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig2_performance_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Figure 2: Performance comparison bars saved")


# ============================================================
# HEATMAP VISUALIZATION
# ============================================================

def create_heatmap_visualization(results):
    """Create heatmap showing model performance across tasks"""
    
    print("\n📊 Generating performance heatmaps...")
    
    targets = ['disease', 'diet', 'lifestyle']
    
    # Get all unique models
    all_models = set()
    for target in targets:
        all_models.update(results[target]['models'].keys())
    model_names = sorted(list(all_models))
    
    if not model_names:
        print("⚠️  No models found")
        return
    
    # Create matrices for accuracy and F1
    accuracy_matrix = np.zeros((len(model_names), len(targets)))
    f1_matrix = np.zeros((len(model_names), len(targets)))
    
    for i, model in enumerate(model_names):
        for j, target in enumerate(targets):
            if model in results[target]['models']:
                accuracy_matrix[i, j] = results[target]['models'][model]['accuracy']
                f1_matrix[i, j] = results[target]['models'][model]['f1_score']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Model-Task Performance Heatmaps', 
                 fontsize=14, fontweight='bold')
    
    # Heatmap 1: Accuracy
    sns.heatmap(accuracy_matrix, annot=True, fmt='.4f', cmap='YlOrRd',
                xticklabels=[t.capitalize() for t in targets],
                yticklabels=model_names, ax=ax1, 
                cbar_kws={'label': 'Accuracy'},
                vmin=0, vmax=1.0)
    ax1.set_title('Test Accuracy Scores', fontweight='bold')
    ax1.set_xlabel('Prediction Task', fontweight='bold')
    ax1.set_ylabel('Algorithm', fontweight='bold')
    
    # Heatmap 2: F1-Score
    sns.heatmap(f1_matrix, annot=True, fmt='.4f', cmap='YlGnBu',
                xticklabels=[t.capitalize() for t in targets],
                yticklabels=model_names, ax=ax2, 
                cbar_kws={'label': 'F1-Score'},
                vmin=0, vmax=1.0)
    ax2.set_title('F1-Scores', fontweight='bold')
    ax2.set_xlabel('Prediction Task', fontweight='bold')
    ax2.set_ylabel('Algorithm', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig3_performance_heatmaps.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Figure 3: Performance heatmaps saved")


# ============================================================
# TRAINING EFFICIENCY ANALYSIS
# ============================================================

def create_efficiency_analysis(results):
    """Create training time vs accuracy scatter plots"""
    
    print("\n📊 Generating efficiency analysis...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Training Efficiency: Time vs. Accuracy Trade-off', 
                 fontsize=14, fontweight='bold')
    
    targets = ['disease', 'diet', 'lifestyle']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for idx, target in enumerate(targets):
        ax = axes[idx]
        
        models_data = results[target]['models']
        if not models_data:
            continue
        
        model_names = list(models_data.keys())
        times = [models_data[m]['training_time_seconds'] for m in model_names]
        accuracies = [models_data[m]['accuracy'] for m in model_names]
        
        # Scatter plot
        ax.scatter(times, accuracies, s=250, alpha=0.6, 
                  c=colors[idx], edgecolors='black', linewidth=2)
        
        # Add labels
        for i, model in enumerate(model_names):
            ax.annotate(model, (times[i], accuracies[i]), 
                       textcoords="offset points", xytext=(0,12), 
                       ha='center', fontsize=8, fontweight='bold')
        
        ax.set_xlabel('Training Time (seconds)', fontweight='bold', fontsize=11)
        ax.set_ylabel('Test Accuracy', fontweight='bold', fontsize=11)
        ax.set_title(f'{target.capitalize()} Prediction', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add reference lines
        if accuracies:
            avg_acc = np.mean(accuracies)
            ax.axhline(y=avg_acc, color='green', linestyle='--', 
                      alpha=0.5, label=f'Mean Accuracy: {avg_acc:.3f}')
            ax.legend(loc='lower right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig4_training_efficiency.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Figure 4: Training efficiency analysis saved")


# ============================================================
# BEST MODELS SUMMARY
# ============================================================

def create_best_models_summary(results):
    """Create visualization showing best model per task"""
    
    print("\n📊 Generating best models summary...")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    targets = ['Disease\nPrediction', 'Diet Plan\nRecommendation', 
              'Lifestyle\nIntervention']
    target_keys = ['disease', 'diet', 'lifestyle']
    
    best_models = []
    best_accuracies = []
    best_f1s = []
    best_times = []
    
    for target_key in target_keys:
        models = results[target_key]['models']
        if models:
            best_model = max(models.items(), key=lambda x: x[1]['accuracy'])
            best_models.append(best_model[0])
            best_accuracies.append(best_model[1]['accuracy'])
            best_f1s.append(best_model[1]['f1_score'])
            best_times.append(best_model[1]['training_time_seconds'])
        else:
            best_models.append('N/A')
            best_accuracies.append(0)
            best_f1s.append(0)
            best_times.append(0)
    
    x = np.arange(len(targets))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, best_accuracies, width, 
                   label='Accuracy', alpha=0.8, color='#FF6B6B')
    bars2 = ax.bar(x + width/2, best_f1s, width, 
                   label='F1-Score', alpha=0.8, color='#4ECDC4')
    
    ax.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax.set_xlabel('Prediction Task', fontweight='bold', fontsize=12)
    ax.set_title('Best Performing Model per Task', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(targets, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.0])
    
    # Add value labels
    ax.bar_label(bars1, fmt='%.4f', padding=3, fontsize=9)
    ax.bar_label(bars2, fmt='%.4f', padding=3, fontsize=9)
    
    # Add model names and training time
    for i, (model, target, t) in enumerate(zip(best_models, targets, best_times)):
        ax.text(i, 0.05, f'🏆 {model}\n⏱️  {t:.1f}s', ha='center', 
               fontsize=9, fontweight='bold', color='darkgreen',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig5_best_models_summary.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Figure 5: Best models summary saved")


# ============================================================
# COMPREHENSIVE SUMMARY TABLE
# ============================================================

def create_summary_table(results):
    """Create comprehensive summary table"""
    
    print("\n📊 Generating comprehensive summary table...")
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')
    
    targets = ['disease', 'diet', 'lifestyle']
    
    # Collect all data
    table_data = []
    headers = ['Task', 'Algorithm', 'Accuracy', 'F1-Score', 
              'Precision', 'Recall', 'Time (s)']
    
    for target in targets:
        models = results[target]['models']
        if not models:
            continue
        
        # Sort by accuracy
        sorted_models = sorted(models.items(), 
                             key=lambda x: x[1]['accuracy'], 
                             reverse=True)
        
        for model_name, metrics in sorted_models:
            row = [
                target.capitalize(),
                model_name,
                f"{metrics['accuracy']:.4f}",
                f"{metrics['f1_score']:.4f}",
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['training_time_seconds']:.2f}"
            ]
            table_data.append(row)
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.12, 0.18, 0.12, 0.12, 0.12, 0.12, 0.10])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white', size=10)
    
    # Alternate row colors and highlight best
    current_task = None
    task_start = 1
    
    for i, row in enumerate(table_data, 1):
        # Alternate row colors
        if i % 2 == 0:
            for j in range(len(headers)):
                table[(i, j)].set_facecolor('#F5F5F5')
        
        # Highlight best model per task (first row of each task)
        if row[0] != current_task:
            current_task = row[0]
            for j in range(len(headers)):
                table[(i, j)].set_facecolor('#90EE90')
                table[(i, j)].set_text_props(weight='bold')
    
    plt.title('Comprehensive Model Performance Summary', 
             fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig(f'{OUTPUT_DIR}/table1_comprehensive_summary.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Table 1: Comprehensive summary saved")


# ============================================================
# GENERATE LATEX TABLE CODE
# ============================================================

def generate_latex_table(results):
    """Generate LaTeX code for IEEE paper"""
    
    print("\n📄 Generating LaTeX table code...")
    
    latex_code = """
\\begin{table*}[t]
\\centering
\\caption{Comprehensive Performance Analysis: Disease, Diet and Lifestyle Prediction Models}
\\label{tab:model_performance}
\\begin{tabular}{llcccccc}
\\toprule
\\textbf{Task} & \\textbf{Algorithm} & \\textbf{Accuracy} & \\textbf{F1} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{Time (s)} \\\\
\\midrule
"""
    
    targets = ['Disease', 'Diet', 'Lifestyle']
    target_keys = ['disease', 'diet', 'lifestyle']
    
    for target, target_key in zip(targets, target_keys):
        models = results[target_key]['models']
        if not models:
            continue
        
        sorted_models = sorted(models.items(), 
                             key=lambda x: x[1]['accuracy'], 
                             reverse=True)
        
        for i, (model_name, metrics) in enumerate(sorted_models):
            if i == 0:
                latex_code += f"{target} & "
            else:
                latex_code += " & "
            
            latex_code += f"{model_name} & "
            latex_code += f"{metrics['accuracy']:.4f} & "
            latex_code += f"{metrics['f1_score']:.4f} & "
            latex_code += f"{metrics['precision']:.4f} & "
            latex_code += f"{metrics['recall']:.4f} & "
            latex_code += f"{metrics['training_time_seconds']:.2f} \\\\\n"
        
        latex_code += "\\midrule\n"
    
    latex_code += """\\bottomrule
\\end{tabular}
\\end{table*}
"""
    
    # Save to file
    with open(f'{OUTPUT_DIR}/table_latex_code.txt', 'w') as f:
        f.write(latex_code)
    
    print("✅ LaTeX table code saved")


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Generate all visualizations"""
    
    print("\n" + "="*70)
    print("UNIFIED VISUALIZATION GENERATION FOR IEEE PAPER")
    print("="*70)
    print("\nCombining results from:")
    print("  • Disease models (train_comparative_models.py)")
    print("  • Diet models (train_diet_models.py)")
    print("  • Lifestyle models (train_lifestyle_models.py)")
    print("="*70)
    
    try:
        # Load all results
        results = load_all_results()
        
        # Check if we have any results
        has_results = False
        for target in ['disease', 'diet', 'lifestyle']:
            if results[target]['models']:
                has_results = True
                break
        
        if not has_results:
            print("\n⚠️  ERROR: No results found!")
            print("\nPlease run the training scripts first:")
            print("  1. python scripts/train_comparative_models.py")
            print("  2. python scripts/train_diet_models.py")
            print("  3. python scripts/train_lifestyle_models.py")
            return
        
        # Generate all visualizations
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        
        create_dataset_visualizations()
        create_performance_comparison_bars(results)
        create_heatmap_visualization(results)
        create_efficiency_analysis(results)
        create_best_models_summary(results)
        create_summary_table(results)
        generate_latex_table(results)
        
        print("\n" + "="*70)
        print("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
        print("="*70)
        print(f"\n📁 Figures saved in: {OUTPUT_DIR}/")
        print("\nGenerated files:")
        print("  • fig1_dataset_composition.png")
        print("  • fig2_performance_comparison.png")
        print("  • fig3_performance_heatmaps.png")
        print("  • fig4_training_efficiency.png")
        print("  • fig5_best_models_summary.png")
        print("  • table1_comprehensive_summary.png")
        print("  • table_latex_code.txt")
        print("\n📄 Ready for IEEE paper submission!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()