import pandas as pd
import numpy as np
from typing import Dict, Any, List

class SimpleRetrieverAnalyzer:
    """Simplified analyzer for LangSmith evaluate() results"""
    
    def __init__(self):
        self.results_data = []

    
    def get_value(self, obj, key):
        """Simple helper to get value from either object attribute or dict key"""
        if hasattr(obj, key):
            return getattr(obj, key)
        elif isinstance(obj, dict) and key in obj:
            return obj[key]
        return None

    def extract_token_costs(self, run):
        """Extract token counts - handles both objects and dictionaries"""
        
        try:
            # Navigate: run -> outputs -> response -> response_metadata -> token_usage
            outputs = self.get_value(run, 'outputs')
            if not outputs:
                return 0, 0
            
            response = self.get_value(outputs, 'response')
            if not response:
                return 0, 0
            
            response_metadata = self.get_value(response, 'response_metadata')
            if not response_metadata:
                return 0, 0
            
            token_usage = self.get_value(response_metadata, 'token_usage')
            if not token_usage:
                return 0, 0
            
            # Get token counts
            input_tokens = self.get_value(token_usage, 'prompt_tokens') or 0
            output_tokens = self.get_value(token_usage, 'completion_tokens') or 0
            
            # Add child run tokens (only LLM runs)
            child_runs = self.get_value(run, 'child_runs')
            if child_runs:
                for child_run in child_runs:
                    run_type = self.get_value(child_run, 'run_type') or ''
                    if 'llm' in str(run_type).lower():
                        child_input, child_output = self.extract_token_costs(child_run)
                        input_tokens += child_input
                        output_tokens += child_output
            
            return input_tokens, output_tokens
            
        except Exception as e:
            print(f"Error extracting tokens: {e}")
            return 0, 0

    def get_model_name(self, run):
        """Extract model name - handles both objects and dictionaries"""
        try:
            outputs = self.get_value(run, 'outputs')
            if outputs:
                response = self.get_value(outputs, 'response')
                if response:
                    response_metadata = self.get_value(response, 'response_metadata')
                    if response_metadata:
                        model_name = self.get_value(response_metadata, 'model_name')
                        if model_name:
                            return model_name
            return "gpt-4"
        except:
            return "gpt-4"

    def calculate_estimated_cost(self, input_tokens, output_tokens, model_name="gpt-4"):
        """Calculate estimated cost based on token counts and model pricing"""
        
        # Example pricing (you'd adjust these for actual model rates)
        PER_MILLION = 1_000_000
        pricing = {
            "gpt-4": {"input": 0.03/PER_MILLION, "output": 0.06/PER_MILLION},  # per token
            "gpt-3.5-turbo": {"input": 0.001/PER_MILLION, "output": 0.002/PER_MILLION},
            "claude": {"input": 0.008/PER_MILLION, "output": 0.024/PER_MILLION}
        }
        
        rates = pricing.get(model_name, pricing["gpt-4"])
        
        input_cost = input_tokens * rates["input"]
        output_cost = output_tokens * rates["output"]
        total_cost = input_cost + output_cost
        
        return input_cost, output_cost, total_cost        

    def get_nested_value(self, obj, *keys):
        """Get nested value using multiple keys"""
        current = obj
        for key in keys:
            current = self.get_value(current, key)
            if current is None:
                return None
        return current

    def add_evaluation_result(self, retriever_name: str, evaluate_result):
        """Extract key metrics from evaluate() result"""
        
        try:
            # Get results from _results attribute
            # detailed_results = evaluate_result._results
            detailed_results = self.get_value(evaluate_result, '_results')
            
            if not detailed_results:
                print(f"No results found for {retriever_name}")
                return
            
            # Initialize metrics tracking
            metrics = {
                'retriever_name': retriever_name,
                'total_runs': len(detailed_results),
                'total_cost': 0.0,
                'total_latency': 0.0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                'qa_scores': [],
                'helpfulness_scores': [],
                'empathy_scores': [],
                'correctness_scores': []
            }
            
            # Process each result
            for result_item in detailed_results:
                run = self.get_value(result_item, 'run') 
                evaluation_results = self.get_nested_value(result_item, 'evaluation_results', 'results') or []
                
                # Extract cost (if available)
                model_name = self.get_model_name(run)
                input_tokens, output_tokens = self.extract_token_costs(run)
                input_cost, output_cost, total_cost = self.calculate_estimated_cost(input_tokens, output_tokens, model_name)
                run_cost = total_cost

                metrics['total_cost'] += run_cost
                metrics['total_input_tokens'] += input_tokens  
                metrics['total_output_tokens'] += output_tokens
                
                # Extract latency
                start_time = self.get_value(run, 'start_time')
                end_time = self.get_value(run, 'end_time')
                
                if start_time and end_time:
                    if hasattr(end_time, 'timestamp') and hasattr(start_time, 'timestamp'):
                        latency = end_time.timestamp() - start_time.timestamp()
                    elif hasattr(end_time, 'total_seconds'):
                        latency = (end_time - start_time).total_seconds()
                    else:
                        latency = 0
                    metrics['total_latency'] += latency
                
                # Extract evaluation scores
                for eval_result in evaluation_results:
                    metric_key = self.get_value(eval_result, 'key')
                    metric_score = self.get_value(eval_result, 'score')
                    
                    if not metric_key or metric_score is None:
                        continue
                        
                    metric_name = metric_key.lower()
                    score = metric_score if metric_score is not None else 0
                    
                    if 'qa' in metric_name or 'question' in metric_name:
                        metrics['qa_scores'].append(score)
                    elif 'helpful' in metric_name:
                        metrics['helpfulness_scores'].append(score)
                    elif 'empathy' in metric_name:
                        metrics['empathy_scores'].append(score)
                    elif 'correct' in metric_name:
                        metrics['correctness_scores'].append(score)
            
            # Calculate summary statistics
            summary = {
                'Retriever': retriever_name.replace('_retrieval_chain', '').replace('_', ' ').title(),
                'Total_Runs': metrics['total_runs'],
                'Avg_Cost_Per_Run': round(metrics['total_cost'] / max(metrics['total_runs'], 1), 6),
                'Total_Cost': round(metrics['total_cost'], 6),
                'Total_Input_Tokens': metrics['total_input_tokens'],
                'Total_Output_Tokens': metrics['total_output_tokens'],
                'Avg_Input_Tokens_Per_Run': round(metrics['total_input_tokens'] / max(metrics['total_runs'], 1), 6),
                'Avg_Output_Tokens_Per_Run': round(metrics['total_output_tokens'] / max(metrics['total_runs'], 1), 6),
                'Avg_Latency_Sec': round(metrics['total_latency'] / max(metrics['total_runs'], 1), 2),
                'Total_Latency_Sec': round(metrics['total_latency'], 2)
            }
            
            # Add performance metrics
            for metric_type, scores in [
                ('QA', metrics['qa_scores']),
                ('Helpfulness', metrics['helpfulness_scores']),
                ('Empathy', metrics['empathy_scores']),
                ('Correctness', metrics['correctness_scores'])
            ]:
                if scores:
                    summary[f'{metric_type}_Avg_Score'] = round(np.mean(scores), 3)
                    summary[f'{metric_type}_Success_Rate'] = round(sum(1 for s in scores if s > 0) / len(scores), 3)
                    summary[f'{metric_type}_Min'] = round(min(scores), 3)
                    summary[f'{metric_type}_Max'] = round(max(scores), 3)
                else:
                    summary[f'{metric_type}_Avg_Score'] = 0.0
                    summary[f'{metric_type}_Success_Rate'] = 0.0
                    summary[f'{metric_type}_Min'] = 0.0
                    summary[f'{metric_type}_Max'] = 0.0
            
            self.results_data.append(summary)
            print(f"✅ Processed {metrics['total_runs']} runs for {retriever_name}")
            
        except Exception as e:
            print(f"❌ Error processing {retriever_name}: {e}")
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Get results as pandas DataFrame"""
        if not self.results_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.results_data)
        
        # Sort by best overall performance (you can adjust this)
        if 'Correctness_Avg_Score' in df.columns:
            df = df.sort_values('Correctness_Avg_Score', ascending=False)
        
        return df
    
    def get_rankings(self) -> Dict[str, pd.DataFrame]:
        """Get rankings by different criteria"""
        df = self.get_results_dataframe()
        
        if df.empty:
            return {}
        
        rankings = {}
        
        # Rank by cost (lower is better)
        if 'Avg_Cost_Per_Run' in df.columns:
            rankings['By_Cost'] = df.nsmallest(len(df), 'Avg_Cost_Per_Run')[['Retriever', 'Avg_Cost_Per_Run']]
        
        # Rank by latency (lower is better)
        if 'Avg_Latency_Sec' in df.columns:
            rankings['By_Latency'] = df.nsmallest(len(df), 'Avg_Latency_Sec')[['Retriever', 'Avg_Latency_Sec']]
        
        # Rank by performance metrics (higher is better)
        for metric in ['Correctness_Avg_Score', 'Helpfulness_Avg_Score', 'Empathy_Avg_Score', 'QA_Avg_Score']:
            if metric in df.columns:
                clean_name = metric.replace('_Avg_Score', '')
                rankings[f'By_{clean_name}'] = df.nlargest(len(df), metric)[['Retriever', metric]]
        
        return rankings
    
    def print_summary(self):
        """Print a quick summary"""
        df = self.get_results_dataframe()
        
        if df.empty:
            print("No data available")
            return
        
        print("\n" + "="*80)
        print("RETRIEVER PERFORMANCE SUMMARY")
        print("="*80)
        
        # Display full dataframe
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        
        print("\n📊 COMPLETE RESULTS:")
        print(df.to_string(index=False))
        
        # Show rankings
        rankings = self.get_rankings()
        
        print("\n🏆 RANKINGS:")
        print("-" * 40)
        
        for rank_name, rank_df in rankings.items():
            if not rank_df.empty:
                print(f"\n{rank_name.replace('_', ' ')}:")
                for i, row in rank_df.iterrows():
                    print(f"  {rank_df.index.get_loc(i) + 1}. {row['Retriever']}: {row.iloc[1]}")
        
        # Best performers summary
        print("\n🎯 BEST PERFORMERS:")
        print("-" * 40)
        
        for metric in ['Correctness_Avg_Score', 'Helpfulness_Avg_Score', 'Avg_Cost_Per_Run', 'Avg_Latency_Sec']:
            if metric in df.columns:
                if 'Cost' in metric or 'Latency' in metric:
                    best_row = df.loc[df[metric].idxmin()]
                    print(f"Best {metric.replace('_', ' ')}: {best_row['Retriever']} ({best_row[metric]})")
                else:
                    best_row = df.loc[df[metric].idxmax()]
                    print(f"Best {metric.replace('_', ' ')}: {best_row['Retriever']} ({best_row[metric]})")

def analyze_retrievers(evaluation_results: Dict[str, Any]) -> pd.DataFrame:
    """Accept both dict of objects OR dict of dicts
    
    # Handle case where evaluation_results might be nested
    # e.g., cached_results vs direct evaluate_result objects
    
    Args:
        evaluation_results: Dict of {retriever_name: evaluate_result}
    
    Returns:
        pandas DataFrame with performance metrics
    """
    
    analyzer = SimpleRetrieverAnalyzer()
    
    print(f"🚀 Analyzing {len(evaluation_results)} retrievers...")
    
    # Add each evaluation result
    for retriever_name, evaluate_result in evaluation_results.items():
        analyzer.add_evaluation_result(retriever_name, evaluate_result)
    
    # Get DataFrame
    df = analyzer.get_results_dataframe()
    
    # Print summary
    analyzer.print_summary()
    
    return df

# Example usage
if __name__ == "__main__":
    # Example usage:
    # evaluation_results = {
    #     'naive_retrieval_chain': naive_result,
    #     'bm25_retrieval_chain': bm25_result,
    #     'semantic_retrieval_chain': semantic_result
    # }
    # 
    # df = analyze_retrievers(evaluation_results)
    # 
    # # Access the DataFrame for further analysis
    # print(df.head())
    # 
    # # Save to CSV
    # df.to_csv('retriever_performance.csv', index=False)
    
    print("Simple Retriever Performance Analyzer")
    print("Usage: df = analyze_retrievers(evaluation_results)")