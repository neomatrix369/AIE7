"""
Fixed Performance Analysis from LangSmith Evaluate Results

This script properly handles the actual structure returned by LangSmith's evaluate() function.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from collections import defaultdict
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path="../.env")

class EvaluateResultsAnalyzer:
    """Analyzes performance using LangSmith evaluate() results directly"""
    
    def __init__(self):
        """Initialize the analyzer"""
        self.evaluation_results = {}
        self.analysis_summary = {}
        self.results_df = None
    
    def add_evaluation_result(self, retriever_name: str, evaluate_result):
        """Add an evaluation result from LangSmith evaluate() function
        
        Args:
            retriever_name: Name of the retriever (e.g., 'naive_retrieval_chain')
            evaluate_result: Result object returned by langsmith.evaluation.evaluate()
        """
        
        print(f"Adding evaluation result for: {retriever_name}")
        
        try:
            # Get the actual results from the _results attribute
            if not hasattr(evaluate_result, '_results'):
                print(f"❌ No _results attribute found for {retriever_name}")
                return
            
            detailed_results = evaluate_result._results
            
            if not detailed_results:
                print(f"❌ No detailed results found for {retriever_name}")
                return
            
            # Initialize tracking variables
            all_metrics = defaultdict(list)
            run_ids = []
            example_ids = []
            detailed_breakdown = []
            total_examples = len(detailed_results)
            total_cost = 0.0
            total_latency = 0.0
            
            print(f"Processing {total_examples} evaluation results...")
            
            # Process each result
            for i, result_item in enumerate(detailed_results):
                try:
                    # Extract run information
                    run = result_item['run']
                    example = result_item['example']
                    evaluation_results = result_item['evaluation_results']['results']
                    
                    run_ids.append(str(run.id))
                    example_ids.append(str(example.id))
                    
                    # Extract cost and latency if available
                    run_cost = 0.0
                    run_latency = 0.0
                    
                    # Try to get cost from run
                    if hasattr(run, 'total_cost') and run.total_cost:
                        run_cost = float(run.total_cost)
                    
                    # Try to get latency from run timing
                    if hasattr(run, 'end_time') and hasattr(run, 'start_time') and run.end_time and run.start_time:
                        if hasattr(run.end_time, 'timestamp') and hasattr(run.start_time, 'timestamp'):
                            run_latency = run.end_time.timestamp() - run.start_time.timestamp()
                        elif isinstance(run.end_time, datetime) and isinstance(run.start_time, datetime):
                            run_latency = (run.end_time - run.start_time).total_seconds()
                    
                    total_cost += run_cost
                    total_latency += run_latency
                    
                    # Extract metrics for this example
                    example_metrics = {}
                    for eval_result in evaluation_results:
                        metric_name = eval_result.key
                        metric_score = eval_result.score
                        metric_value = eval_result.value
                        
                        # Store for aggregation
                        all_metrics[metric_name].append(metric_score)
                        
                        # Store detailed info
                        example_metrics[metric_name] = {
                            'score': metric_score,
                            'value': metric_value,
                            'evaluator_run_id': str(eval_result.source_run_id)
                        }
                    
                    # Build detailed breakdown
                    detailed_breakdown.append({
                        'example_index': i,
                        'example_id': str(example.id),
                        'example_link': getattr(example, 'link', 'N/A'),
                        'run_id': str(run.id),
                        'run_name': getattr(run, 'name', 'N/A'),
                        'run_type': getattr(run, 'run_type', 'N/A'),
                        'cost': run_cost,
                        'latency': run_latency,
                        'metrics': example_metrics
                    })
                    
                except Exception as e:
                    print(f"Error processing result {i}: {e}")
                    continue
            
            # Calculate aggregate metrics
            aggregate_metrics = {}
            for metric_name, scores in all_metrics.items():
                if scores:
                    aggregate_metrics[f'{metric_name}_mean'] = np.mean(scores)
                    aggregate_metrics[f'{metric_name}_std'] = np.std(scores)
                    aggregate_metrics[f'{metric_name}_total'] = sum(scores)
                    aggregate_metrics[f'{metric_name}_count'] = len(scores)
                    aggregate_metrics[f'{metric_name}_success_rate'] = sum(1 for s in scores if s > 0) / len(scores)
                    aggregate_metrics[f'{metric_name}_min'] = min(scores)
                    aggregate_metrics[f'{metric_name}_max'] = max(scores)
            
            # Store the results
            results_data = {
                'retriever_name': retriever_name,
                'total_runs': total_examples,
                'total_cost': round(total_cost, 6),
                'avg_cost_per_run': round(total_cost / max(total_examples, 1), 6),
                'total_latency': round(total_latency, 2),
                'avg_latency': round(total_latency / max(total_examples, 1), 2),
                'run_ids': run_ids,
                'example_ids': example_ids,
                'aggregate_metrics': aggregate_metrics,
                'detailed_results': detailed_breakdown,
                'raw_scores': dict(all_metrics)
            }
            
            self.evaluation_results[retriever_name] = results_data
            
            print(f"✅ Processed {total_examples} runs for {retriever_name}")
            print(f"   - Total cost: ${total_cost:.6f}")
            print(f"   - Avg cost per run: ${results_data['avg_cost_per_run']:.6f}")
            print(f"   - Avg latency: {results_data['avg_latency']:.2f}s")
            print(f"   - Metrics tracked: {list(all_metrics.keys())}")
            
            # Print aggregate metrics
            for metric_name in all_metrics.keys():
                mean_score = aggregate_metrics[f'{metric_name}_mean']
                success_rate = aggregate_metrics[f'{metric_name}_success_rate']
                print(f"   - {metric_name}: {mean_score:.3f} avg, {success_rate:.1%} success rate")
            
        except Exception as e:
            print(f"❌ Error processing evaluation result for {retriever_name}: {e}")
            import traceback
            traceback.print_exc()
    
    def create_analysis_dataframe(self) -> pd.DataFrame:
        """Create a DataFrame from all evaluation results"""
        
        if not self.evaluation_results:
            print("No evaluation results to analyze")
            return pd.DataFrame()
        
        analysis_data = []
        
        for retriever_name, results in self.evaluation_results.items():
            # Create base record
            record = {
                'retriever_name': retriever_name,
                'total_runs': results['total_runs'],
                'total_cost': results['total_cost'],
                'avg_cost_per_run': results['avg_cost_per_run'],
                'avg_latency': results['avg_latency'],
                'total_latency': results['total_latency']
            }
            
            # Add aggregate metrics
            for metric_name, metric_value in results['aggregate_metrics'].items():
                record[metric_name] = metric_value
            
            analysis_data.append(record)
        
        self.results_df = pd.DataFrame(analysis_data)
        return self.results_df
    
    def create_performance_summary(self) -> Dict[str, Any]:
        """Create a comprehensive performance summary"""
        
        if not self.evaluation_results:
            return {"error": "No data available for analysis"}
        
        summary = {}
        
        for retriever_name, results in self.evaluation_results.items():
            summary[retriever_name] = {
                'total_runs': results['total_runs'],
                'avg_cost_per_run': results['avg_cost_per_run'],
                'total_cost': results['total_cost'],
                'avg_latency': results['avg_latency'],
            }
            
            # Add performance metrics
            for metric_name, metric_value in results['aggregate_metrics'].items():
                if metric_name.endswith('_mean'):
                    base_name = metric_name.replace('_mean', '')
                    summary[retriever_name][f'{base_name}_score'] = metric_value
                elif metric_name.endswith('_success_rate'):
                    summary[retriever_name][metric_name] = metric_value
        
        self.analysis_summary = summary
        return summary
    
    def rank_retrievers(self) -> Dict[str, List]:
        """Rank retrievers by different criteria"""
        
        if not self.analysis_summary:
            self.create_performance_summary()
        
        if not self.analysis_summary:
            return {'by_cost': [], 'by_latency': [], 'by_correctness': [], 'by_helpfulness': [], 'by_overall': []}
        
        # Extract data for ranking
        retriever_data = []
        for retriever, metrics in self.analysis_summary.items():
            retriever_data.append({
                'name': retriever,
                'cost': metrics.get('avg_cost_per_run', float('inf')),
                'latency': metrics.get('avg_latency', float('inf')),
                'correctness': metrics.get('correctness_score', 0),
                'helpfulness': metrics.get('helpfulness_score', 0),
                'empathy': metrics.get('empathy_score', 0)
            })
        
        if not retriever_data:
            return {'by_cost': [], 'by_latency': [], 'by_correctness': [], 'by_helpfulness': [], 'by_overall': []}
        
        df = pd.DataFrame(retriever_data)
        
        rankings = {
            'by_cost': df.nsmallest(len(df), 'cost')[['name', 'cost']].to_dict('records'),
            'by_latency': df.nsmallest(len(df), 'latency')[['name', 'latency']].to_dict('records'),
            'by_correctness': df.nlargest(len(df), 'correctness')[['name', 'correctness']].to_dict('records'),
            'by_helpfulness': df.nlargest(len(df), 'helpfulness')[['name', 'helpfulness']].to_dict('records')
        }
        
        # Calculate overall score
        if len(df) > 1 and df['cost'].max() > df['cost'].min():
            # Normalize scores (0-1 range)
            df['cost_norm'] = 1 - (df['cost'] - df['cost'].min()) / (df['cost'].max() - df['cost'].min())
            df['latency_norm'] = 1 - (df['latency'] - df['latency'].min()) / (df['latency'].max() - df['latency'].min() + 1e-8)
            df['correctness_norm'] = df['correctness']  # Already 0-1
            df['helpfulness_norm'] = df['helpfulness']  # Already 0-1
            
            # Weighted overall score (adjust weights as needed)
            df['overall_score'] = (
                0.4 * df['correctness_norm'] + 
                0.2 * df['helpfulness_norm'] + 
                0.2 * df['cost_norm'] + 
                0.2 * df['latency_norm']
            )
            rankings['by_overall'] = df.nlargest(len(df), 'overall_score')[['name', 'overall_score']].to_dict('records')
        else:
            rankings['by_overall'] = []
        
        return rankings
    
    def generate_analysis_report(self) -> str:
        """Generate a comprehensive analysis report"""
        
        if not self.evaluation_results:
            return "❌ No evaluation results available. Please add evaluation results first using add_evaluation_result()."
        
        # Ensure we have the latest analysis
        self.create_analysis_dataframe()
        summary = self.create_performance_summary()
        rankings = self.rank_retrievers()
        
        report = []
        report.append("# 📊 Retriever Performance Analysis Report")
        report.append("*Generated from LangSmith evaluate() results*")
        report.append("=" * 60)
        report.append("")
        
        # Executive Summary
        report.append("## 📋 Executive Summary")
        report.append("")
        total_runs = sum(results['total_runs'] for results in self.evaluation_results.values())
        total_cost = sum(results['total_cost'] for results in self.evaluation_results.values())
        
        report.append(f"Analyzed **{len(summary)}** different retriever strategies across **{total_runs}** total evaluation runs.")
        report.append(f"Total evaluation cost: **${total_cost:.6f}**")
        report.append("")
        
        # Best performers
        if rankings['by_correctness']:
            best_correctness = rankings['by_correctness'][0]['name']
            best_score = rankings['by_correctness'][0]['correctness']
            report.append(f"🏆 **Best Correctness**: {best_correctness} ({best_score:.3f})")
        
        if rankings['by_cost']:
            best_cost = rankings['by_cost'][0]['name']
            cost_value = rankings['by_cost'][0]['cost']
            report.append(f"💰 **Most Cost-Effective**: {best_cost} (${cost_value:.6f} per run)")
        
        if rankings['by_latency']:
            best_latency = rankings['by_latency'][0]['name']
            latency_value = rankings['by_latency'][0]['latency']
            report.append(f"⚡ **Fastest**: {best_latency} ({latency_value:.2f}s avg)")
        
        if rankings['by_overall']:
            best_overall = rankings['by_overall'][0]['name']
            overall_score = rankings['by_overall'][0]['overall_score']
            report.append(f"🎯 **Best Overall**: {best_overall} ({overall_score:.3f} weighted score)")
        
        report.append("")
        
        # Detailed Analysis
        report.append("## 📈 Detailed Performance Analysis")
        report.append("")
        
        for retriever, metrics in summary.items():
            clean_name = retriever.replace('_retrieval_chain', '').replace('_', ' ').title()
            report.append(f"### {clean_name}")
            report.append("")
            report.append(f"- **Total Runs**: {metrics.get('total_runs', 0)}")
            report.append(f"- **Average Cost per Run**: ${metrics.get('avg_cost_per_run', 0):.6f}")
            report.append(f"- **Total Cost**: ${metrics.get('total_cost', 0):.6f}")
            report.append(f"- **Average Latency**: {metrics.get('avg_latency', 0):.2f}s")
            
            # Add performance scores
            for score_type in ['correctness_score', 'helpfulness_score', 'empathy_score']:
                if score_type in metrics:
                    clean_score_name = score_type.replace('_score', '').title()
                    score_value = metrics[score_type]
                    success_rate_key = score_type.replace('_score', '_success_rate')
                    success_rate = metrics.get(success_rate_key, 0)
                    report.append(f"- **{clean_score_name} Score**: {score_value:.3f} (Success Rate: {success_rate:.1%})")
            
            report.append("")
        
        # Rankings
        report.append("## 🏆 Rankings")
        report.append("")
        
        for rank_type, title in [
            ('by_correctness', '📈 By Correctness (Higher is Better)'),
            ('by_helpfulness', '🤝 By Helpfulness (Higher is Better)'),
            ('by_cost', '💰 By Cost (Lower is Better)'),
            ('by_latency', '⚡ By Latency (Lower is Better)'),
            ('by_overall', '🎯 Overall Ranking (Weighted Score)')
        ]:
            if rankings[rank_type]:
                report.append(f"### {title}")
                for i, item in enumerate(rankings[rank_type], 1):
                    name = item['name'].replace('_retrieval_chain', '').replace('_', ' ').title()
                    value = item.get('cost', item.get('latency', item.get('correctness', item.get('helpfulness', item.get('overall_score', 0)))))
                    
                    if rank_type == 'by_cost':
                        report.append(f"{i}. {name}: ${value:.6f}")
                    elif rank_type == 'by_latency':
                        report.append(f"{i}. {name}: {value:.2f}s")
                    else:
                        report.append(f"{i}. {name}: {value:.3f}")
                report.append("")
        
        # Recommendations
        report.append("## 💡 Recommendations")
        report.append("")
        
        if rankings['by_overall'] and rankings['by_overall'][0]:
            best = rankings['by_overall'][0]['name'].replace('_retrieval_chain', '').replace('_', ' ').title()
            score = rankings['by_overall'][0]['overall_score']
            report.append(f"**🎯 Primary Recommendation**: {best} (Score: {score:.3f})")
            report.append("Based on weighted analysis (40% correctness, 20% helpfulness, 20% cost, 20% latency).")
            report.append("")
        
        report.append("**📋 Use Case Specific Recommendations**:")
        report.append("")
        
        if rankings['by_correctness']:
            name = rankings['by_correctness'][0]['name'].replace('_retrieval_chain', '').replace('_', ' ').title()
            score = rankings['by_correctness'][0]['correctness']
            report.append(f"- **For Maximum Accuracy**: {name} ({score:.3f} correctness)")
        
        if rankings['by_cost']:
            name = rankings['by_cost'][0]['name'].replace('_retrieval_chain', '').replace('_', ' ').title()
            cost = rankings['by_cost'][0]['cost']
            report.append(f"- **For Cost Optimization**: {name} (${cost:.6f} per run)")
        
        if rankings['by_latency']:
            name = rankings['by_latency'][0]['name'].replace('_retrieval_chain', '').replace('_', ' ').title()
            latency = rankings['by_latency'][0]['latency']
            report.append(f"- **For Speed Critical Applications**: {name} ({latency:.2f}s avg)")
        
        report.append("")
        report.append("---")
        report.append(f"📅 Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return "\n".join(report)
    
    def save_analysis(self, filename: str = "retriever_analysis.md"):
        """Save analysis results to multiple formats"""
        
        print(f"\n💾 Saving analysis results...")
        
        # Save main report
        report = self.generate_analysis_report()
        with open(filename, 'w') as f:
            f.write(report)
        print(f"📄 Report saved: {filename}")
        
        # Save raw data as CSV
        if self.results_df is not None and not self.results_df.empty:
            csv_filename = filename.replace('.md', '_data.csv')
            self.results_df.to_csv(csv_filename, index=False)
            print(f"📊 Data saved: {csv_filename}")
        
        # Save summary as JSON
        if self.analysis_summary:
            json_filename = filename.replace('.md', '_summary.json')
            with open(json_filename, 'w') as f:
                json.dump(self.analysis_summary, f, indent=2, default=str)
            print(f"📋 Summary saved: {json_filename}")
        
        # Save detailed evaluation results
        eval_results_filename = filename.replace('.md', '_evaluation_results.json')
        with open(eval_results_filename, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2, default=str)
        print(f"🔍 Detailed results saved: {eval_results_filename}")
        
        # Save reloadable evaluation data
        self.save_evaluation_data(filename)
    
    def save_evaluation_data(self, base_filename: str):
        """Save evaluation data in a format that can be reloaded"""
        
        eval_data_filename = base_filename.replace('.md', '_evaluation_data.pkl')
        
        try:
            import pickle
            
            # Create comprehensive data structure for reloading
            save_data = {
                'evaluation_results': self.evaluation_results,
                'analysis_summary': self.analysis_summary,
                'results_df': self.results_df.to_dict('records') if self.results_df is not None else None,
                'metadata': {
                    'save_time': datetime.now().isoformat(),
                    'retriever_count': len(self.evaluation_results),
                    'total_runs': sum(result['total_runs'] for result in self.evaluation_results.values()),
                    'total_cost': sum(result['total_cost'] for result in self.evaluation_results.values())
                }
            }
            
            with open(eval_data_filename, 'wb') as f:
                pickle.dump(save_data, f)
            print(f"💾 Evaluation data saved for reloading: {eval_data_filename}")
            
            # Also save as JSON for cross-platform compatibility
            json_data_filename = base_filename.replace('.md', '_evaluation_data.json')
            with open(json_data_filename, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
            print(f"📋 Evaluation data (JSON) saved: {json_data_filename}")
            
        except Exception as e:
            print(f"⚠️ Could not save evaluation data: {e}")
    
    def load_evaluation_data(self, base_filename: str) -> bool:
        """Load previously saved evaluation data"""
        
        eval_data_filename = base_filename.replace('.md', '_evaluation_data.pkl')
        json_data_filename = base_filename.replace('.md', '_evaluation_data.json')
        
        # Try pickle first, then JSON
        for filename, loader in [(eval_data_filename, self._load_pickle), (json_data_filename, self._load_json)]:
            if os.path.exists(filename):
                try:
                    save_data = loader(filename)
                    
                    # Restore data
                    self.evaluation_results = save_data['evaluation_results']
                    self.analysis_summary = save_data['analysis_summary']
                    
                    if save_data['results_df']:
                        self.results_df = pd.DataFrame(save_data['results_df'])
                    
                    metadata = save_data.get('metadata', {})
                    print(f"✅ Loaded evaluation data from: {filename}")
                    print(f"   - Save time: {metadata.get('save_time', 'Unknown')}")
                    print(f"   - Retrievers: {metadata.get('retriever_count', len(self.evaluation_results))}")
                    print(f"   - Total runs: {metadata.get('total_runs', 'Unknown')}")
                    print(f"   - Total cost: ${metadata.get('total_cost', 0):.6f}")
                    
                    return True
                    
                except Exception as e:
                    print(f"⚠️ Could not load from {filename}: {e}")
                    continue
        
        print(f"❌ No evaluation data found for: {base_filename}")
        return False
    
    def _load_pickle(self, filename: str):
        """Load data from pickle file"""
        import pickle
        with open(filename, 'rb') as f:
            return pickle.load(f)
    
    def _load_json(self, filename: str):
        """Load data from JSON file"""
        with open(filename, 'r') as f:
            return json.load(f)
    
    def create_visualization_plots(self, save_plots: bool = True, base_filename: str = "retriever_analysis"):
        """Create visualization plots for the analysis"""
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if not self.analysis_summary:
                print("❌ No analysis data available for plotting")
                return
            
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Retriever Performance Analysis', fontsize=16, fontweight='bold')
            
            # Prepare data
            retrievers = list(self.analysis_summary.keys())
            retriever_names = [name.replace('_retrieval_chain', '').replace('_', ' ').title() for name in retrievers]
            
            # 1. Correctness Scores
            correctness_scores = [self.analysis_summary[r].get('correctness_score', 0) for r in retrievers]
            axes[0, 0].bar(retriever_names, correctness_scores, color='skyblue', alpha=0.7)
            axes[0, 0].set_title('Correctness Scores by Retriever')
            axes[0, 0].set_ylabel('Correctness Score')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(axis='y', alpha=0.3)
            
            # 2. Cost per Run
            costs = [self.analysis_summary[r].get('avg_cost_per_run', 0) for r in retrievers]
            axes[0, 1].bar(retriever_names, costs, color='lightcoral', alpha=0.7)
            axes[0, 1].set_title('Average Cost per Run')
            axes[0, 1].set_ylabel('Cost ($)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(axis='y', alpha=0.3)
            
            # 3. Latency
            latencies = [self.analysis_summary[r].get('avg_latency', 0) for r in retrievers]
            axes[1, 0].bar(retriever_names, latencies, color='lightgreen', alpha=0.7)
            axes[1, 0].set_title('Average Latency')
            axes[1, 0].set_ylabel('Latency (seconds)')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(axis='y', alpha=0.3)
            
            # 4. Multi-metric comparison
            metrics_to_plot = ['correctness_score', 'helpfulness_score', 'empathy_score']
            x = np.arange(len(retriever_names))
            width = 0.25
            
            for i, metric in enumerate(metrics_to_plot):
                values = [self.analysis_summary[r].get(metric, 0) for r in retrievers]
                offset = (i - 1) * width
                axes[1, 1].bar(x + offset, values, width, label=metric.replace('_score', '').title(), alpha=0.7)
            
            axes[1, 1].set_title('Multi-Metric Comparison')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(retriever_names, rotation=45)
            axes[1, 1].legend()
            axes[1, 1].grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            
            if save_plots:
                plot_filename = f"{base_filename}_plots.png"
                plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                print(f"📊 Plots saved: {plot_filename}")
            
            plt.show()
            
        except ImportError:
            print("⚠️ Matplotlib/Seaborn not available. Install with: pip install matplotlib seaborn")
        except Exception as e:
            print(f"⚠️ Error creating plots: {e}")
    
    def export_for_corpus_quality_demo(self, base_filename: str = "corpus_quality_demo_data"):
        """Export data specifically formatted for corpus quality improvement demos"""
        
        if not self.evaluation_results:
            print("❌ No evaluation results available")
            return
        
        demo_data = {
            'summary': {},
            'before_after_comparison': {},
            'failed_examples_for_improvement': {},
            'cost_benefit_analysis': {},
            'improvement_recommendations': []
        }
        
        # Create summary for demo
        for retriever_name, results in self.evaluation_results.items():
            clean_name = retriever_name.replace('_retrieval_chain', '').replace('_', ' ').title()
            
            demo_data['summary'][clean_name] = {
                'correctness_score': results['aggregate_metrics'].get('correctness_mean', 0),
                'helpfulness_score': results['aggregate_metrics'].get('helpfulness_mean', 0),
                'success_rate': results['aggregate_metrics'].get('correctness_success_rate', 0),
                'avg_cost_per_run': results['avg_cost_per_run'],
                'total_runs': results['total_runs'],
                'failed_examples_count': len([ex for ex in results['detailed_results'] 
                                            if ex['metrics'].get('correctness', {}).get('score', 1) == 0])
            }
        
        # Create before/after comparison (assuming first is baseline)
        retriever_names = list(self.evaluation_results.keys())
        if len(retriever_names) >= 2:
            baseline = retriever_names[0]
            best_performer = max(retriever_names, 
                               key=lambda x: self.evaluation_results[x]['aggregate_metrics'].get('correctness_mean', 0))
            
            baseline_score = self.evaluation_results[baseline]['aggregate_metrics'].get('correctness_mean', 0)
            best_score = self.evaluation_results[best_performer]['aggregate_metrics'].get('correctness_mean', 0)
            
            demo_data['before_after_comparison'] = {
                'baseline': {
                    'name': baseline,
                    'correctness': baseline_score,
                    'cost': self.evaluation_results[baseline]['avg_cost_per_run']
                },
                'improved': {
                    'name': best_performer,
                    'correctness': best_score,
                    'cost': self.evaluation_results[best_performer]['avg_cost_per_run']
                },
                'improvement': {
                    'correctness_delta': best_score - baseline_score,
                    'correctness_percent_change': ((best_score - baseline_score) / max(baseline_score, 0.001)) * 100,
                    'cost_delta': self.evaluation_results[best_performer]['avg_cost_per_run'] - self.evaluation_results[baseline]['avg_cost_per_run']
                }
            }
        
        # Get failed examples for corpus improvement
        for retriever_name, results in self.evaluation_results.items():
            failed_examples = []
            for example in results['detailed_results']:
                if example['metrics'].get('correctness', {}).get('score', 1) == 0:
                    failed_examples.append({
                        'example_id': example['example_id'],
                        'example_link': example['example_link'],
                        'run_id': example['run_id']
                    })
            
            if failed_examples:
                demo_data['failed_examples_for_improvement'][retriever_name] = failed_examples[:5]  # Top 5 failures
        
        # Cost-benefit analysis
        total_cost = sum(results['total_cost'] for results in self.evaluation_results.values())
        total_runs = sum(results['total_runs'] for results in self.evaluation_results.values())
        
        demo_data['cost_benefit_analysis'] = {
            'total_evaluation_cost': total_cost,
            'total_runs': total_runs,
            'avg_cost_per_run': total_cost / max(total_runs, 1),
            'cost_to_find_quality_issues': total_cost,
            'estimated_savings_from_proactive_assessment': total_cost * 10  # Assume 10x ROI
        }
        
        # Generate improvement recommendations
        if demo_data['before_after_comparison']:
            improvement = demo_data['before_after_comparison']['improvement']
            demo_data['improvement_recommendations'] = [
                f"Improve correctness by {improvement['correctness_percent_change']:.1f}% through targeted corpus enhancement",
                f"Focus on {len(demo_data['failed_examples_for_improvement'].get(baseline, []))} specific content gaps",
                f"Proactive assessment cost: ${total_cost:.4f} vs reactive fixes (estimated 10x higher)",
                "Implement continuous corpus quality monitoring"
            ]
        
        # Save demo data
        demo_filename = f"{base_filename}.json"
        with open(demo_filename, 'w') as f:
            json.dump(demo_data, f, indent=2, default=str)
        
        print(f"🎯 Demo data exported: {demo_filename}")
        print("📋 Demo highlights:")
        print(f"   - Analyzed {len(self.evaluation_results)} retriever strategies")
        print(f"   - Total cost: ${total_cost:.6f}")
        if demo_data['before_after_comparison']:
            improvement = demo_data['before_after_comparison']['improvement']
            print(f"   - Correctness improvement: {improvement['correctness_percent_change']:.1f}%")
        
        return demo_data
    
    def print_quick_summary(self):
        """Print a quick summary of all retrievers"""
        
        if not self.evaluation_results:
            print("❌ No evaluation results available")
            return
        
        print("\n" + "="*80)
        print("QUICK PERFORMANCE SUMMARY")
        print("="*80)
        
        for retriever_name, results in self.evaluation_results.items():
            clean_name = retriever_name.replace('_retrieval_chain', '').replace('_', ' ').title()
            print(f"\n📊 {clean_name}")
            print(f"   Runs: {results['total_runs']}")
            print(f"   Cost: ${results['total_cost']:.6f} (${results['avg_cost_per_run']:.6f}/run)")
            print(f"   Latency: {results['avg_latency']:.2f}s avg")
            
            # Print key metrics
            for metric_name, scores in results['raw_scores'].items():
                if scores:
                    avg_score = np.mean(scores)
                    success_rate = sum(1 for s in scores if s > 0) / len(scores)
                    print(f"   {metric_name.title()}: {avg_score:.3f} avg, {success_rate:.1%} success")
    
    def get_failed_examples(self, retriever_name: str, metric_name: str = 'correctness') -> List[Dict]:
        """Get examples where a specific metric failed"""
        
        if retriever_name not in self.evaluation_results:
            return []
        
        results = self.evaluation_results[retriever_name]
        failed_examples = []
        
        for example in results['detailed_results']:
            if metric_name in example['metrics']:
                if example['metrics'][metric_name]['score'] == 0:
                    failed_examples.append({
                        'example_id': example['example_id'],
                        'example_link': example['example_link'],
                        'run_id': example['run_id'],
                        'metric_value': example['metrics'][metric_name]['value'],
                        'cost': example['cost'],
                        'latency': example['latency']
                    })
        
        return failed_examples

# Enhanced usage functions with caching and smart loading capabilities
def smart_analyze_retrievers(evaluation_results: Dict[str, Any] = None, 
                           cache_filename: str = "retriever_performance_analysis.md", 
                           force_rerun: bool = False) -> EvaluateResultsAnalyzer:
    """Smart analysis that loads cached results if available, otherwise runs fresh analysis
    
    Args:
        evaluation_results: Dictionary of evaluate() results (only needed if no cache or force_rerun=True)
        cache_filename: Filename for saving/loading cached results
        force_rerun: If True, ignore cached results and rerun analysis
        
    Returns:
        EvaluateResultsAnalyzer with complete analysis
    """
    
    analyzer = EvaluateResultsAnalyzer()
    
    # Try to load existing results first (unless forced to rerun)
    if not force_rerun and analyzer.load_evaluation_data(cache_filename):
        print("✅ Using cached evaluation results")
        print(f"📊 Found {len(analyzer.evaluation_results)} retriever strategies")
        
        # Generate quick summary
        analyzer.print_quick_summary()
        return analyzer
    
    # If no cached results or forced rerun, we need fresh evaluation results
    if evaluation_results is None:
        print("❌ No cached results found and no evaluation_results provided")
        print("Please either:")
        print("1. Provide evaluation_results parameter, or")
        print("2. Run evaluations first to generate cached results")
        return analyzer
    
    # Run fresh analysis
    return analyze_from_evaluate_results(evaluation_results, cache_filename, force_rerun=True)

def analyze_from_evaluate_results(evaluation_results: Dict[str, Any], 
                                cache_filename: str = "retriever_performance_analysis.md",
                                force_rerun: bool = False) -> EvaluateResultsAnalyzer:
    """Analyze performance from a dictionary of evaluate() results with enhanced features"""
    
    analyzer = EvaluateResultsAnalyzer()
    
    # Check for existing cache unless forced to rerun
    if not force_rerun and analyzer.load_evaluation_data(cache_filename):
        print("✅ Loaded existing evaluation results from cache")
        print(f"📊 Found {len(analyzer.evaluation_results)} retriever strategies")
        analyzer.print_quick_summary()
        return analyzer
    
    print("🚀 Starting fresh analysis from evaluate() results...")
    print(f"📊 Analyzing {len(evaluation_results)} retriever strategies")
    print("-" * 50)
    
    # Add each evaluation result
    for retriever_name, evaluate_result in evaluation_results.items():
        analyzer.add_evaluation_result(retriever_name, evaluate_result)
    
    # Generate analysis
    analyzer.create_analysis_dataframe()
    analyzer.create_performance_summary()
    
    # Print quick summary
    analyzer.print_quick_summary()
    
    # Generate and print full report
    print("\n📊 Generating detailed analysis report...")
    report = analyzer.generate_analysis_report()
    print("\n" + report)
    
    # Save results with all formats
    print(f"\n💾 Saving comprehensive results...")
    analyzer.save_analysis(cache_filename)
    
    # Create visualizations
    try:
        analyzer.create_visualization_plots(save_plots=True, base_filename=cache_filename.replace('.md', ''))
    except Exception as e:
        print(f"⚠️ Could not create plots: {e}")
    
    # Export demo-specific data
    try:
        demo_data = analyzer.export_for_corpus_quality_demo(cache_filename.replace('.md', '_demo'))
        print(f"🎯 Demo data exported successfully")
    except Exception as e:
        print(f"⚠️ Could not export demo data: {e}")
    
    print("\n🎉 Analysis complete!")
    return analyzer

def create_corpus_quality_demo_summary(analyzer: EvaluateResultsAnalyzer) -> str:
    """Create a focused summary for corpus quality improvement demos"""
    
    if not analyzer.evaluation_results:
        return "❌ No evaluation results available"
    
    # Find baseline and best performer
    retrievers = list(analyzer.evaluation_results.keys())
    baseline = retrievers[0]  # Assume first is baseline
    
    best_performer = max(retrievers, 
                        key=lambda x: analyzer.evaluation_results[x]['aggregate_metrics'].get('correctness_mean', 0))
    
    baseline_data = analyzer.evaluation_results[baseline]
    best_data = analyzer.evaluation_results[best_performer]
    
    baseline_score = baseline_data['aggregate_metrics'].get('correctness_mean', 0)
    best_score = best_data['aggregate_metrics'].get('correctness_mean', 0)
    
    improvement = ((best_score - baseline_score) / max(baseline_score, 0.001)) * 100
    
    summary = f"""
🎯 CORPUS QUALITY IMPROVEMENT DEMO SUMMARY
============================================

📊 BEFORE vs AFTER Analysis:
• Baseline ({baseline}): {baseline_score:.3f} correctness
• Optimized ({best_performer}): {best_score:.3f} correctness
• Improvement: {improvement:+.1f}%

💰 Cost Analysis:
• Baseline cost/run: ${baseline_data['avg_cost_per_run']:.6f}
• Optimized cost/run: ${best_data['avg_cost_per_run']:.6f}
• Total evaluation cost: ${sum(r['total_cost'] for r in analyzer.evaluation_results.values()):.6f}

🔍 Quality Issues Identified:
• Failed examples in baseline: {len([ex for ex in baseline_data['detailed_results'] if ex['metrics'].get('correctness', {}).get('score', 1) == 0])}
• Success rate improvement: {(best_data['aggregate_metrics'].get('correctness_success_rate', 0) - baseline_data['aggregate_metrics'].get('correctness_success_rate', 0)) * 100:+.1f}%

🚀 Demo Impact:
✅ Proactive quality assessment prevents poor user experience
✅ Quantified improvement metrics show clear ROI
✅ Specific failed examples guide corpus enhancement
✅ Cost-effective compared to reactive fixes (estimated 10x savings)
"""
    
    return summary

# Example usage documentation
def create_comprehensive_example():
    """Create comprehensive example usage"""
    
    example_code = '''
# === COMPLETE WORKFLOW EXAMPLE ===

# 1. Run your evaluations (replace with your actual evaluation calls)
evaluation_results = {}
for retriever_name in ['naive', 'bm25', 'semantic', 'ensemble']:
    result = evaluate(
        your_chain.invoke,
        data="your_dataset", 
        evaluators=[correctness_evaluator, helpfulness_evaluator],
        experiment_prefix=f"{retriever_name}_retrieval_chain"
    )
    evaluation_results[f"{retriever_name}_retrieval_chain"] = result

# 2. Smart analysis with caching
analyzer = smart_analyze_retrievers(evaluation_results, 
                                  cache_filename="my_corpus_quality_analysis.md")

# 3. Get demo-ready summary
demo_summary = create_corpus_quality_demo_summary(analyzer)
print(demo_summary)

# 4. Access specific data for your demo
rankings = analyzer.rank_retrievers()
best_retriever = rankings['by_overall'][0]['name']
print(f"🏆 Best overall: {best_retriever}")

# 5. Get failed examples to guide corpus improvement
failed_examples = analyzer.get_failed_examples('naive_retrieval_chain', 'correctness')
print(f"🔍 Found {len(failed_examples)} examples to improve")

# 6. Export for presentation
demo_data = analyzer.export_for_corpus_quality_demo("presentation_data")

# 7. Create visualizations
analyzer.create_visualization_plots(save_plots=True)

# === SUBSEQUENT RUNS (USES CACHE) ===
# Just load from cache - much faster!
analyzer = smart_analyze_retrievers()  # Automatically uses cache
demo_summary = create_corpus_quality_demo_summary(analyzer)
'''
    
    return example_code

# Example usage
if __name__ == "__main__":
    print("Fixed Performance Analysis for LangSmith Evaluate Results")
    print("=" * 60)
    print("\nThis script properly handles the _results structure from evaluate() calls.")
    print("\nExample usage:")
    print("""
# After running your evaluations:
evaluation_results = {
    'naive_retrieval_chain': naive_result,
    'bm25_retrieval_chain': bm25_result,
    # ... other results
}

# Analyze all results
analyzer = analyze_from_evaluate_results(evaluation_results)

# Get specific insights
failed_examples = analyzer.get_failed_examples('naive_retrieval_chain', 'correctness')
rankings = analyzer.rank_retrievers()
""")