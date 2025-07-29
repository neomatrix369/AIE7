import pandas as pd

def rank_retrievers(csv_file, weights={'quality': 0.6, 'efficiency': 0.25, 'reliability': 0.15}):
    """Rank retrievers and return DataFrame with scores"""
    # Load data
    df = pd.read_csv(csv_file)
    
    # Calculate scores
    quality_cols = ['Correctness_Avg_Score', 'Helpfulness_Avg_Score', 'Empathy_Avg_Score', 'QA_Avg_Score']
    df['Quality_Score'] = df[quality_cols].fillna(0).mean(axis=1)
    df['Efficiency_Score'] = 1 - (df['Total_Cost'] / df['Total_Cost'].max())
    df['Reliability_Score'] = df['Quality_Score']  # Using quality as proxy
    
    # Final weighted score
    df['Final_Score'] = (df['Quality_Score'] * weights['quality'] + 
                        df['Efficiency_Score'] * weights['efficiency'] + 
                        df['Reliability_Score'] * weights['reliability'])
    
    # Rank and sort
    df['Rank'] = df['Final_Score'].rank(method='dense', ascending=False)
    return df.sort_values('Final_Score', ascending=False)

def get_summary_table(df):
    """Return summary DataFrame with key scores"""
    summary_df = df[['Rank', 'Retriever', 'Final_Score', 'Quality_Score', 'Efficiency_Score']].copy()
    summary_df['Rank'] = summary_df['Rank'].astype(int)
    return summary_df

def get_detailed_table(df):
    """Return detailed DataFrame with performance metrics"""
    detailed_df = df[['Rank', 'Retriever', 'Correctness_Avg_Score', 'Helpfulness_Avg_Score', 
                     'Empathy_Avg_Score', 'Total_Cost']].copy()
    detailed_df['Rank'] = detailed_df['Rank'].astype(int)
    return detailed_df

def get_comparison_table(scenarios_data):
    """Return DataFrame comparing top 3 across scenarios"""
    comparison_data = []
    for scenario_name, df in scenarios_data.items():
        top_3 = df.head(3)['Retriever'].tolist()
        comparison_data.append([scenario_name] + top_3)
    
    comparison_df = pd.DataFrame(comparison_data, columns=['Scenario', '1st Place', '2nd Place', '3rd Place'])
    return comparison_df

def get_winner_info(df):
    """Return winner information as dictionary"""
    winner = df.iloc[0]
    return {
        'retriever': winner['Retriever'],
        'final_score': winner['Final_Score'],
        'correctness': winner['Correctness_Avg_Score'],
        'helpfulness': winner['Helpfulness_Avg_Score'],
        'cost': winner['Total_Cost']
    }

def run_langsmith_analysis(csv_file='langsmith_retriever_summary_stats.csv'):
    """Run complete analysis and return all DataFrames"""
    scenarios = {
        "BALANCED": {'quality': 0.6, 'efficiency': 0.25, 'reliability': 0.15},
        "QUALITY_FIRST": {'quality': 0.8, 'efficiency': 0.1, 'reliability': 0.1},
        "COST_CONSCIOUS": {'quality': 0.4, 'efficiency': 0.5, 'reliability': 0.1}
    }
    
    # Process all scenarios
    scenarios_data = {}
    for name, weights in scenarios.items():
        scenarios_data[name] = rank_retrievers(csv_file, weights)
    
    # Generate all tables
    results = {
        'scenarios_data': scenarios_data,
        'summary_tables': {name: get_summary_table(df) for name, df in scenarios_data.items()},
        'detailed_tables': {name: get_detailed_table(df) for name, df in scenarios_data.items()},
        'comparison_table': get_comparison_table(scenarios_data),
        'winner_info': get_winner_info(scenarios_data["BALANCED"])
    }
    
    return results

# Convenience functions for individual tables
def get_balanced_summary(csv_file='langsmith_retriever_summary_stats.csv'):
    """Get summary table for balanced scenario"""
    df = rank_retrievers(csv_file)
    return get_summary_table(df)

def get_balanced_detailed(csv_file='langsmith_retriever_summary_stats.csv'):
    """Get detailed table for balanced scenario"""
    df = rank_retrievers(csv_file)
    return get_detailed_table(df)

def get_all_scenarios_comparison(csv_file='langsmith_retriever_summary_stats.csv'):
    """Get comparison table across all scenarios"""
    results = run_langsmith_analysis(csv_file)
    return results['comparison_table']

# Usage examples:
if __name__ == "__main__":
    # Get all results
    results = run_langsmith_analysis()
    
    # Access individual DataFrames
    print("Summary for BALANCED scenario:")
    print(results['summary_tables']['BALANCED'])
    
    print("\nDetailed metrics for BALANCED scenario:")
    print(results['detailed_tables']['BALANCED'])
    
    print("\nComparison across scenarios:")
    print(results['comparison_table'])
    
    print("\nWinner info:")
    winner = results['winner_info']
    print(f"Winner: {winner['retriever']} (Score: {winner['final_score']:.3f})")
    
    # Or use convenience functions
    summary_df = get_balanced_summary()
    detailed_df = get_balanced_detailed()
    comparison_df = get_all_scenarios_comparison()