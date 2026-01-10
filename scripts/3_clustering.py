"""
Script 3: Patient Clustering with KMeans
- Perform KMeans clustering for patient profiling
- Visualize clusters
- Save model
"""

from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def create_spark_session():
    """Initialize Spark session"""
    spark = SparkSession.builder \
        .appName("Healthcare Clustering") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    return spark

def perform_clustering(spark):
    """Perform KMeans clustering on patient data"""
    
    print("Loading processed data...")
    df = spark.read.parquet('data/processed/healthcare_cleaned.parquet')
    
    print(f"Data loaded: {df.count()} rows")
    
    # Determine optimal number of clusters using elbow method
    print("\nFinding optimal number of clusters...")
    
    silhouette_scores = []
    K_range = range(3, 11)
    
    for k in K_range:
        kmeans = KMeans(featuresCol='features', predictionCol='cluster', k=k, seed=42)
        model = kmeans.fit(df)
        predictions = model.transform(df)
        
        evaluator = ClusteringEvaluator(featuresCol='features', 
                                        predictionCol='cluster', 
                                        metricName='silhouette')
        score = evaluator.evaluate(predictions)
        silhouette_scores.append(score)
        print(f"K={k}, Silhouette Score={score:.4f}")
    
    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(list(K_range), silhouette_scores, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (K)', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('data/processed/elbow_curve.png', dpi=300, bbox_inches='tight')
    print("\n✓ Elbow curve saved to data/processed/elbow_curve.png")
    plt.close()
    
    # Choose optimal K (using k=5 for patient profiling)
    optimal_k = 5
    print(f"\nUsing K={optimal_k} clusters for patient profiling")
    
    # Train final KMeans model
    kmeans = KMeans(featuresCol='features', predictionCol='cluster', k=optimal_k, seed=42)
    kmeans_model = kmeans.fit(df)
    
    # Make predictions
    clustered_df = kmeans_model.transform(df)
    
    # Save model
    os.makedirs('models/kmeans_model', exist_ok=True)
    kmeans_model.write().overwrite().save('models/kmeans_model')
    print("✓ KMeans model saved")
    
    # Evaluate
    evaluator = ClusteringEvaluator(featuresCol='features', 
                                    predictionCol='cluster', 
                                    metricName='silhouette')
    silhouette = evaluator.evaluate(clustered_df)
    print(f"\nFinal Silhouette Score: {silhouette:.4f}")
    
    # Analyze clusters
    print("\nCluster distribution:")
    clustered_df.groupBy('cluster').count().orderBy('cluster').show()
    
    # Analyze cluster characteristics
    print("\nCluster characteristics:")
    cluster_stats = clustered_df.groupBy('cluster').agg(
        {'age': 'avg', 'bmi': 'avg', 'bp_systolic': 'avg', 
         'glucose': 'avg', 'cholesterol': 'avg'}
    ).orderBy('cluster')
    
    cluster_stats.show()
    
    # Convert to pandas for visualization
    cluster_df = clustered_df.select('cluster', 'age', 'bmi', 'bp_systolic', 
                                      'glucose', 'cholesterol', 'disease').toPandas()
    
    # Sample for visualization (to avoid memory issues)
    if len(cluster_df) > 50000:
        cluster_df = cluster_df.sample(n=50000, random_state=42)
    
    # Visualize cluster characteristics
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    features = ['age', 'bmi', 'bp_systolic', 'glucose', 'cholesterol']
    
    for idx, feature in enumerate(features):
        row = idx // 3
        col = idx % 3
        
        sns.boxplot(data=cluster_df, x='cluster', y=feature, ax=axes[row, col], 
                    palette='Set2')
        axes[row, col].set_title(f'{feature.upper()} by Cluster', 
                                 fontsize=12, fontweight='bold')
        axes[row, col].set_xlabel('Cluster', fontsize=10)
        axes[row, col].set_ylabel(feature.replace('_', ' ').title(), fontsize=10)
    
    # Disease distribution by cluster
    disease_cluster = cluster_df.groupby(['cluster', 'disease']).size().reset_index(name='count')
    disease_pivot = disease_cluster.pivot(index='cluster', columns='disease', values='count').fillna(0)
    
    sns.heatmap(disease_pivot, annot=True, fmt='.0f', cmap='YlOrRd', 
                ax=axes[1, 2], cbar_kws={'label': 'Count'})
    axes[1, 2].set_title('Disease Distribution by Cluster', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('Disease', fontsize=10)
    axes[1, 2].set_ylabel('Cluster', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('data/processed/cluster_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Cluster analysis visualization saved to data/processed/cluster_analysis.png")
    plt.close()
    
    # Profile each cluster
    print("\n" + "="*60)
    print("CLUSTER PROFILES")
    print("="*60)
    
    for cluster_id in range(optimal_k):
        cluster_data = cluster_df[cluster_df['cluster'] == cluster_id]
        print(f"\nCluster {cluster_id} (n={len(cluster_data)}):")
        print(f"  Avg Age: {cluster_data['age'].mean():.1f}")
        print(f"  Avg BMI: {cluster_data['bmi'].mean():.1f}")
        print(f"  Avg BP: {cluster_data['bp_systolic'].mean():.1f}")
        print(f"  Avg Glucose: {cluster_data['glucose'].mean():.1f}")
        print(f"  Avg Cholesterol: {cluster_data['cholesterol'].mean():.1f}")
        print(f"  Top Diseases: {cluster_data['disease'].value_counts().head(3).to_dict()}")
    
    print("\n✓ Clustering analysis complete!")
    
    return kmeans_model, clustered_df

# Main execution
if __name__ == "__main__":
    spark = create_spark_session()
    
    try:
        model, clustered_data = perform_clustering(spark)
        print("\n✓ All clustering steps completed successfully!")
    except Exception as e:
        print(f"Error during clustering: {e}")
        raise
    finally:
        spark.stop()