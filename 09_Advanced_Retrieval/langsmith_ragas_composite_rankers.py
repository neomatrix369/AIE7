import pandas as pd
import numpy as np

class RetrieverRanker:
    def __init__(self, quality_weight=0.5, efficiency_weight=0.3, reliability_weight=0.2):
        self.weights = {
            'quality': quality_weight,
            'efficiency': efficiency_weight, 
            'reliability': reliability_weight
        }
        
    def load_data(self, langsmith_csv, ragas_csv):
        """Load and preprocess CSV files"""
        self.langsmith_df = pd.read_csv(langsmith_csv)
        self.ragas_df = pd.read_csv(ragas_csv)
        
        # Normalize retriever names for matching
        name_map = {
            'Bm25': 'bm25',
            'Contextual Compression': 'contextual_compression',
            'Ensemble': 'ensemble', 
            'Naive': 'naive',
            'Parent Document': 'parent_document',
            'Multi Query': 'multi_query'
        }
        
        self.langsmith_df['normalized_name'] = self.langsmith_df['Retriever'].map(name_map)
        self.ragas_df['normalized_name'] = self.ragas_df['retriever'].str.replace('_retrieval_chain', '')
        
    def calculate_scores(self):
        """Calculate quality, efficiency, and reliability scores"""
        results = []
        
        for _, ls_row in self.langsmith_df.iterrows():
            ragas_row = self.ragas_df[self.ragas_df['normalized_name'] == ls_row['normalized_name']]
            
            if ragas_row.empty:
                continue
                
            ragas_row = ragas_row.iloc[0]
            
            # Quality Score (0-100) - Optimized for new metrics
            langsmith_quality = (ls_row['Helpfulness_Avg_Score'] * 0.5 + ls_row['Correctness_Avg_Score'] * 0.5) * 100
            
            # RAGAS quality with new metrics - weighted by discriminative power and importance
            ragas_quality = (
                ragas_row['context_recall'] * 0.30 +                              # Core retrieval quality
                ragas_row['faithfulness'] * 0.25 +                                # Answer faithfulness  
                ragas_row['llm_context_precision_with_reference'] * 0.25 +        # Precision with ground truth
                ragas_row['context_entity_recall'] * 0.15 +                       # Entity capture
                ragas_row['llm_context_precision_without_reference'] * 0.05       # Additional precision signal
            ) * 100
            
            quality_score = langsmith_quality * 0.35 + ragas_quality * 0.65  # Increased RAGAS weight due to richer metrics
            
            # Efficiency Score (0-100) - lower cost = higher score
            max_cost = max(self.langsmith_df['Avg_Cost_Per_Run'].max(), self.ragas_df['Avg_Cost_Per_Run'].max())
            avg_cost = (ls_row['Avg_Cost_Per_Run'] + ragas_row['Avg_Cost_Per_Run']) / 2
            efficiency_score = (1 - (avg_cost / max_cost)) * 100
            
            # Reliability Score (0-100) - Enhanced with new metrics
            reliability_ls = (ls_row['Helpfulness_Success_Rate'] * 0.5 + ls_row['Correctness_Success_Rate'] * 0.5) * 100
            
            # Noise sensitivity (lower = better, so invert) + consistency bonus for high precision
            noise_robustness = (1 - ragas_row['noise_sensitivity_relevant']) * 100
            precision_consistency = ragas_row['llm_context_precision_without_reference'] * 100  # High baseline precision = reliable
            
            reliability_score = reliability_ls * 0.4 + noise_robustness * 0.4 + precision_consistency * 0.2
            
            # Composite Score
            composite_score = (
                quality_score * self.weights['quality'] +
                efficiency_score * self.weights['efficiency'] +
                reliability_score * self.weights['reliability']
            )
            
            results.append({
                'retriever': ls_row['Retriever'],
                'quality_score': round(quality_score, 1),
                'efficiency_score': round(efficiency_score, 1),
                'reliability_score': round(reliability_score, 1),
                'composite_score': round(composite_score, 1),
                'ls_helpfulness': ls_row['Helpfulness_Avg_Score'],
                'ls_correctness': ls_row['Correctness_Avg_Score'],
                'ragas_recall': ragas_row['context_recall'],
                'ragas_faithfulness': ragas_row['faithfulness'],
                'ragas_precision_with_ref': ragas_row['llm_context_precision_with_reference'],
                'ragas_precision_without_ref': ragas_row['llm_context_precision_without_reference'],
                'ragas_entity_recall': ragas_row['context_entity_recall'],
                'ragas_noise_sensitivity': ragas_row['noise_sensitivity_relevant'],
                'avg_cost': round((ls_row['Avg_Cost_Per_Run'] + ragas_row['Avg_Cost_Per_Run']) / 2, 6)
            })
            
        self.results_df = pd.DataFrame(results)
        return self.results_df.sort_values('composite_score', ascending=False).reset_index(drop=True)
    
    def get_rankings_table(self):
        """Get main rankings table"""
        df = self.results_df.copy()
        df['rank'] = range(1, len(df) + 1)
        return df[['rank', 'retriever', 'composite_score', 'quality_score', 'efficiency_score', 'reliability_score']]
    
    def get_detailed_metrics_table(self):
        """Get detailed metrics comparison table"""
        return self.results_df[['retriever', 'ls_helpfulness', 'ls_correctness', 'ragas_recall', 
                               'ragas_faithfulness', 'ragas_precision_with_ref', 'ragas_precision_without_ref',
                               'ragas_entity_recall', 'ragas_noise_sensitivity', 'avg_cost', 'composite_score']]
    
    def get_category_leaders_table(self):
        """Get best performer in each category"""
        leaders = []
        
        # Overall winner
        overall = self.results_df.iloc[0]
        leaders.append(['Overall Best', overall['retriever'], overall['composite_score']])
        
        # Category leaders
        quality_leader = self.results_df.loc[self.results_df['quality_score'].idxmax()]
        leaders.append(['Quality Leader', quality_leader['retriever'], quality_leader['quality_score']])
        
        efficiency_leader = self.results_df.loc[self.results_df['efficiency_score'].idxmax()]
        leaders.append(['Most Efficient', efficiency_leader['retriever'], efficiency_leader['efficiency_score']])
        
        reliability_leader = self.results_df.loc[self.results_df['reliability_score'].idxmax()]
        leaders.append(['Most Reliable', reliability_leader['retriever'], reliability_leader['reliability_score']])
        
        return pd.DataFrame(leaders, columns=['Category', 'Retriever', 'Score'])
    
def analyze_retrievers(langsmith_csv, ragas_csv, quality_weight=0.5, efficiency_weight=0.3, reliability_weight=0.2):
    """Main function to analyze retrievers and return DataFrames"""
    ranker = RetrieverRanker(quality_weight, efficiency_weight, reliability_weight)
    ranker.load_data(langsmith_csv, ragas_csv)
    ranker.calculate_scores()
    
    return {
        'rankings': ranker.get_rankings_table(),
        'detailed_metrics': ranker.get_detailed_metrics_table(),
        'category_leaders': ranker.get_category_leaders_table(),
        'full_results': ranker.results_df
    }

def get_rankings_table(langsmith_csv, ragas_csv, quality_weight=0.5, efficiency_weight=0.3, reliability_weight=0.2):
    """Get rankings table as DataFrame"""
    results = analyze_retrievers(langsmith_csv, ragas_csv, quality_weight, efficiency_weight, reliability_weight)
    return results['rankings']

def get_detailed_metrics_table(langsmith_csv, ragas_csv, quality_weight=0.5, efficiency_weight=0.3, reliability_weight=0.2):
    """Get detailed metrics table as DataFrame"""
    results = analyze_retrievers(langsmith_csv, ragas_csv, quality_weight, efficiency_weight, reliability_weight)
    return results['detailed_metrics']

def get_category_leaders_table(langsmith_csv, ragas_csv, quality_weight=0.5, efficiency_weight=0.3, reliability_weight=0.2):
    """Get category leaders table as DataFrame"""
    results = analyze_retrievers(langsmith_csv, ragas_csv, quality_weight, efficiency_weight, reliability_weight)
    return results['category_leaders']

def get_full_results(langsmith_csv, ragas_csv, quality_weight=0.5, efficiency_weight=0.3, reliability_weight=0.2):
    """Get complete results DataFrame with all metrics"""
    results = analyze_retrievers(langsmith_csv, ragas_csv, quality_weight, efficiency_weight, reliability_weight)
    return results['full_results']

# Example usage
if __name__ == "__main__":
    # Get individual tables as DataFrames
    rankings_df = get_rankings_table('langsmith_retriever_raw_stats.csv', 'ragas_retriever_raw_stats.csv')
    detailed_df = get_detailed_metrics_table('langsmith_retriever_raw_stats.csv', 'ragas_retriever_raw_stats.csv')
    leaders_df = get_category_leaders_table('langsmith_retriever_raw_stats.csv', 'ragas_retriever_raw_stats.csv')
    
    # Get all results at once
    all_results = analyze_retrievers('langsmith_retriever_raw_stats.csv', 'ragas_retriever_raw_stats.csv')
    
    # Quality-focused analysis
    quality_rankings = get_rankings_table('langsmith_retriever_raw_stats.csv', 'ragas_retriever_raw_stats.csv', 
                                        quality_weight=0.7, efficiency_weight=0.2, reliability_weight=0.1)
    
    # Cost-conscious analysis
    cost_rankings = get_rankings_table('langsmith_retriever_raw_stats.csv', 'ragas_retriever_raw_stats.csv',
                                     quality_weight=0.3, efficiency_weight=0.5, reliability_weight=0.2)