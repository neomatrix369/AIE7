import pandas as pd
import numpy as np

class RetrieverRanker:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.normalized_df = self._normalize_data()
    
    def _normalize_data(self):
        """Normalize metrics to 0-1 scale for fair comparison"""
        df_norm = self.df.copy()
        metrics = ['context_recall', 'faithfulness', 'factual_correctness', 
                  'answer_relevancy', 'context_entity_recall', 'noise_sensitivity_relevant', 'Avg_Cost_Per_Run']
        
        for metric in metrics:
            if metric in df_norm.columns:
                min_val, max_val = df_norm[metric].min(), df_norm[metric].max()
                if metric == 'Avg_Cost_Per_Run':  # Lower cost is better
                    df_norm[f'{metric}_norm'] = (max_val - df_norm[metric]) / (max_val - min_val) if max_val != min_val else 1
                else:  # Higher is better for quality metrics
                    df_norm[f'{metric}_norm'] = (df_norm[metric] - min_val) / (max_val - min_val) if max_val != min_val else 1
        
        return df_norm
    
    def weighted_score(self, weights=None):
        """Calculate weighted score based on custom weights"""
        if weights is None:
            weights = {
                'context_recall': 20, 'faithfulness': 20, 'factual_correctness': 15,
                'answer_relevancy': 20, 'context_entity_recall': 10, 
                'noise_sensitivity_relevant': 5, 'cost_efficiency': 10
            }
        
        total_weight = sum(weights.values())
        scores = []
        
        for _, row in self.normalized_df.iterrows():
            score = (
                weights['context_recall'] * row['context_recall_norm'] +
                weights['faithfulness'] * row['faithfulness_norm'] +
                weights['factual_correctness'] * row['factual_correctness_norm'] +
                weights['answer_relevancy'] * row['answer_relevancy_norm'] +
                weights['context_entity_recall'] * row['context_entity_recall_norm'] +
                weights['noise_sensitivity_relevant'] * row['noise_sensitivity_relevant_norm'] +
                weights['cost_efficiency'] * row['Avg_Cost_Per_Run_norm']
            ) / total_weight
            scores.append(score)
        
        return scores
    
    def quality_first_score(self):
        """Prioritize quality with cost penalty"""
        scores = []
        for _, row in self.df.iterrows():
            quality = (row['context_recall'] + row['faithfulness'] + 
                      row['answer_relevancy'] + row['factual_correctness']) / 4
            cost_penalty = max(0, (row['Avg_Cost_Per_Run'] - self.df['Avg_Cost_Per_Run'].min()) / 
                              (self.df['Avg_Cost_Per_Run'].max() - self.df['Avg_Cost_Per_Run'].min()) * 0.1)
            scores.append(max(0, quality - cost_penalty))
        return scores
    
    def balanced_score(self):
        """Balanced approach considering all factors"""
        scores = []
        for _, row in self.normalized_df.iterrows():
            core_quality = (row['context_recall_norm'] + row['faithfulness_norm'] + 
                           row['answer_relevancy_norm']) / 3
            accuracy_bonus = row['factual_correctness_norm'] * 0.2
            robustness = (row['context_entity_recall_norm'] + row['noise_sensitivity_relevant_norm']) / 2 * 0.1
            efficiency = row['Avg_Cost_Per_Run_norm'] * 0.1
            scores.append(core_quality + accuracy_bonus + robustness + efficiency)
        return scores
    
    def production_ready_score(self):
        """Production-ready with minimum thresholds"""
        thresholds = {'context_recall': 0.7, 'faithfulness': 0.8, 'answer_relevancy': 0.85}
        scores = []
        
        for _, row in self.df.iterrows():
            meets_thresholds = all(row[metric] >= threshold for metric, threshold in thresholds.items())
            if not meets_thresholds:
                scores.append(0)
                continue
            
            quality_excess = sum(max(0, row[metric] - thresholds[metric]) for metric in thresholds)
            cost_efficiency = self.normalized_df.loc[row.name, 'Avg_Cost_Per_Run_norm']
            scores.append(quality_excess + cost_efficiency * 0.3)
        
        return scores
    
    def get_rankings_table(self, algorithm='weighted', weights=None):
        """Generate rankings table for specified algorithm"""
        df_result = self.df.copy()
        df_result['retriever_chain'] = df_result['retriever'].str.replace('_retrieval_chain', '').str.replace('_', ' ').str.title()
        
        if algorithm == 'weighted':
            df_result['score'] = self.weighted_score(weights)
        elif algorithm == 'quality_first':
            df_result['score'] = self.quality_first_score()
        elif algorithm == 'balanced':
            df_result['score'] = self.balanced_score()
        elif algorithm == 'production_ready':
            df_result['score'] = self.production_ready_score()
        
        df_result = df_result.sort_values('score', ascending=False).reset_index(drop=True)
        df_result['rank'] = range(1, len(df_result) + 1)
        
        return df_result[['rank', 'retriever_chain', 'score', 'context_recall', 'faithfulness', 
                        'factual_correctness', 'answer_relevancy', 'Avg_Cost_Per_Run']].round(4)
    
    def get_metrics_comparison_table(self):
        """Compare all retrievers across key metrics"""
        df_comp = self.df.copy()
        df_comp['retriever_chain'] = df_comp['retriever'].str.replace('_retrieval_chain', '').str.replace('_', ' ').str.title()
        
        return df_comp[['retriever_chain', 'context_recall', 'faithfulness', 'factual_correctness', 
                       'answer_relevancy', 'context_entity_recall', 'noise_sensitivity_relevant', 
                       'Avg_Cost_Per_Run']].round(4)
    
    def get_algorithm_comparison_table(self):
        """Compare rankings across all algorithms"""
        algorithms = ['weighted', 'quality_first', 'balanced', 'production_ready']
        results = {}
        
        for algo in algorithms:
            rankings = self.get_rankings_table(algo)
            results[f'{algo}_rank'] = rankings.set_index('retriever_chain')['rank']
            results[f'{algo}_score'] = rankings.set_index('retriever_chain')['score']
        
        df_comp = pd.DataFrame(results)
        df_comp.index.name = 'retriever'
        return df_comp.round(4)
    
    def get_recommendations_table(self):
        """Generate recommendations for different use cases"""
        recommendations = []
        
        # Overall winner (weighted)
        winner = self.get_rankings_table('weighted').iloc[0]
        recommendations.append(['Overall Winner', winner['retriever_chain'], 
                               f"Score: {winner['score']:.3f}", 'Best balanced performance'])
        
        # Budget option
        budget_idx = self.df['Avg_Cost_Per_Run'].idxmin()
        budget_retriever = self.df.loc[budget_idx, 'retriever'].replace('_retrieval_chain', '').replace('_', ' ').title()
        budget_cost = self.df.loc[budget_idx, 'Avg_Cost_Per_Run']
        recommendations.append(['Budget Option', budget_retriever, 
                               f"Cost: ${budget_cost:.4f}", 'Lowest cost per run'])
        
        # Quality leader
        quality_scores = (self.df['context_recall'] + self.df['faithfulness'] + self.df['answer_relevancy']) / 3
        quality_idx = quality_scores.idxmax()
        quality_retriever = self.df.loc[quality_idx, 'retriever'].replace('_retrieval_chain', '').replace('_', ' ').title()
        quality_score = quality_scores.iloc[quality_idx]
        recommendations.append(['Quality Leader', quality_retriever, 
                               f"Quality: {quality_score:.3f}", 'Highest average quality metrics'])
        
        # Production ready
        prod_rankings = self.get_rankings_table('production_ready')
        prod_winner = prod_rankings[prod_rankings['score'] > 0].iloc[0] if (prod_rankings['score'] > 0).any() else prod_rankings.iloc[0]
        recommendations.append(['Production Ready', prod_winner['retriever_chain'], 
                               f"Score: {prod_winner['score']:.3f}", 'Meets minimum thresholds'])
        
        return pd.DataFrame(recommendations, columns=['Category', 'Retriever', 'Key Metric', 'Description'])

def main():
    # Initialize ranker
    ranker = RetrieverRanker('ragas_retriever_raw_stats.csv')
    
    print("🏆 RETRIEVER RANKING ANALYSIS")
    print("=" * 60)
    
    # 1. Overall Rankings (Weighted Algorithm)
    print("\n1. OVERALL RANKINGS (Weighted Algorithm)")
    print("-" * 45)
    rankings = ranker.get_rankings_table('weighted')
    print(rankings.to_string(index=False))
    
    # 2. Metrics Comparison
    print("\n\n2. DETAILED METRICS COMPARISON")
    print("-" * 45)
    metrics = ranker.get_metrics_comparison_table()
    print(metrics.to_string(index=False))
    
    # 3. Algorithm Comparison
    print("\n\n3. ALGORITHM COMPARISON (Rankings)")
    print("-" * 45)
    algo_comp = ranker.get_algorithm_comparison_table()
    print(algo_comp.to_string())
    
    # 4. Recommendations
    print("\n\n4. RECOMMENDATIONS")
    print("-" * 45)
    recommendations = ranker.get_recommendations_table()
    print(recommendations.to_string(index=False))
    
    # 5. Top 3 Summary
    print("\n\n5. TOP 3 SUMMARY")
    print("-" * 45)
    top3 = rankings.head(3)[['rank', 'retriever_chain', 'score', 'context_recall', 'faithfulness', 'Avg_Cost_Per_Run']]
    print(top3.to_string(index=False))

if __name__ == "__main__":
    main()