"""
Script 4: Train ML Models using RandomForest 
Changed from Logistic Regression to RandomForest for better performance
"""

from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col
import os
import sys
import json

def create_spark_session():
    """Initialize Spark session"""
    try:
        spark = SparkSession.builder \
            .appName("Healthcare Model Training") \
            .master("local[*]") \
            .config("spark.driver.memory", "6g") \
            .config("spark.executor.memory", "6g") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
            .config("spark.driver.host", "localhost") \
            .config("spark.driver.bindAddress", "127.0.0.1") \
            .config("spark.sql.shuffle.partitions", "8") \
            .config("spark.default.parallelism", "4") \
            .getOrCreate()
        
        spark.sparkContext.setLogLevel("ERROR")
        return spark
    except Exception as e:
        print(f"Error creating Spark session: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure Java is installed: java -version")
        print("2. Check JAVA_HOME environment variable")
        print("3. Verify PySpark installation: pip show pyspark")
        sys.exit(1)

def train_disease_model(train_data, test_data):
    """Train disease prediction model with RandomForest"""
    
    print("\n" + "="*60)
    print("TRAINING DISEASE PREDICTION MODEL (RandomForest)")
    print("="*60)
    
    # Cache data
    train_data.cache()
    test_data.cache()
    
    # RandomForest Classifier with optimized hyperparameters
    rf = RandomForestClassifier(
        featuresCol='features', 
        labelCol='disease_label',
        numTrees=120,  # Increased trees
        maxDepth=15,   # Deeper trees for complex patterns
        minInstancesPerNode=5,
        maxBins=64,
        seed=42,
        subsamplingRate=0.8,
        featureSubsetStrategy='sqrt'
    )
    
    print("Training RandomForest with 150 trees...")
    print("This may take 5-10 minutes depending on your system...")
    
    try:
        rf_model = rf.fit(train_data)
        print("✓ Training completed")
    except Exception as e:
        print(f"Error during training: {e}")
        raise
    
    # Make predictions
    print("Making predictions on test set...")
    predictions = rf_model.transform(test_data)
    
    # Evaluate metrics
    print("Calculating metrics...")
    
    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol='disease_label', 
        predictionCol='prediction', 
        metricName='accuracy'
    )
    
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol='disease_label', 
        predictionCol='prediction', 
        metricName='f1'
    )
    
    evaluator_precision = MulticlassClassificationEvaluator(
        labelCol='disease_label', 
        predictionCol='prediction', 
        metricName='weightedPrecision'
    )
    
    evaluator_recall = MulticlassClassificationEvaluator(
        labelCol='disease_label', 
        predictionCol='prediction', 
        metricName='weightedRecall'
    )
    
    accuracy = evaluator_acc.evaluate(predictions)
    f1 = evaluator_f1.evaluate(predictions)
    precision = evaluator_precision.evaluate(predictions)
    recall = evaluator_recall.evaluate(predictions)
    
    print(f"\n🎯 Disease Model Performance:")
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   F1 Score:  {f1:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    
    # Feature importance
    feature_importance = rf_model.featureImportances
    print(f"\n📊 Top 5 Most Important Features:")
    importance_list = [(i, float(feature_importance[i])) for i in range(len(feature_importance))]
    importance_list.sort(key=lambda x: x[1], reverse=True)
    for idx, (feat_idx, importance) in enumerate(importance_list[:5], 1):
        print(f"   {idx}. Feature {feat_idx}: {importance:.4f}")
    
    # Save model
    print("\nSaving model...")
    os.makedirs('models/disease_model', exist_ok=True)
    rf_model.write().overwrite().save('models/disease_model')
    print("✓ Disease model saved")
    
    # Unpersist cached data
    train_data.unpersist()
    test_data.unpersist()
    
    # Return metrics
    metrics = {
        'accuracy': float(accuracy),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall)
    }
    
    return rf_model, metrics

def train_diet_model(train_data, test_data):
    """Train diet plan recommendation model with RandomForest"""
    
    print("\n" + "="*60)
    print("TRAINING DIET PLAN RECOMMENDATION MODEL (RandomForest)")
    print("="*60)
    
    # Cache data
    train_data.cache()
    test_data.cache()
    
    # RandomForest Classifier
    rf = RandomForestClassifier(
        featuresCol='features', 
        labelCol='diet_label',
        numTrees=120,
        maxDepth=12,
        minInstancesPerNode=5,
        maxBins=64,
        seed=42,
        subsamplingRate=0.8,
        featureSubsetStrategy='sqrt'
    )
    
    print("Training RandomForest with 120 trees...")
    
    try:
        rf_model = rf.fit(train_data)
        print("✓ Training completed")
    except Exception as e:
        print(f"Error during training: {e}")
        raise
    
    # Make predictions
    print("Making predictions on test set...")
    predictions = rf_model.transform(test_data)
    
    # Evaluate metrics
    print("Calculating metrics...")
    
    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol='diet_label', 
        predictionCol='prediction', 
        metricName='accuracy'
    )
    
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol='diet_label', 
        predictionCol='prediction', 
        metricName='f1'
    )
    
    evaluator_precision = MulticlassClassificationEvaluator(
        labelCol='diet_label', 
        predictionCol='prediction', 
        metricName='weightedPrecision'
    )
    
    evaluator_recall = MulticlassClassificationEvaluator(
        labelCol='diet_label', 
        predictionCol='prediction', 
        metricName='weightedRecall'
    )
    
    accuracy = evaluator_acc.evaluate(predictions)
    f1 = evaluator_f1.evaluate(predictions)
    precision = evaluator_precision.evaluate(predictions)
    recall = evaluator_recall.evaluate(predictions)
    
    print(f"\n🥗 Diet Model Performance:")
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   F1 Score:  {f1:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    
    # Save model
    print("\nSaving model...")
    os.makedirs('models/diet_model', exist_ok=True)
    rf_model.write().overwrite().save('models/diet_model')
    print("✓ Diet model saved")
    
    # Unpersist cached data
    train_data.unpersist()
    test_data.unpersist()
    
    # Return metrics
    metrics = {
        'accuracy': float(accuracy),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall)
    }
    
    return rf_model, metrics

def train_lifestyle_model(train_data, test_data):
    """Train lifestyle plan recommendation model with RandomForest"""
    
    print("\n" + "="*60)
    print("TRAINING LIFESTYLE PLAN RECOMMENDATION MODEL (RandomForest)")
    print("="*60)
    
    # Cache data
    train_data.cache()
    test_data.cache()
    
    # RandomForest Classifier
    rf = RandomForestClassifier(
        featuresCol='features', 
        labelCol='lifestyle_label',
        numTrees=120,
        maxDepth=12,
        minInstancesPerNode=5,
        maxBins=64,
        seed=42,
        subsamplingRate=0.8,
        featureSubsetStrategy='sqrt'
    )
    
    print("Training RandomForest with 120 trees...")
    
    try:
        rf_model = rf.fit(train_data)
        print("✓ Training completed")
    except Exception as e:
        print(f"Error during training: {e}")
        raise
    
    # Make predictions
    print("Making predictions on test set...")
    predictions = rf_model.transform(test_data)
    
    # Evaluate metrics
    print("Calculating metrics...")
    
    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol='lifestyle_label', 
        predictionCol='prediction', 
        metricName='accuracy'
    )
    
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol='lifestyle_label', 
        predictionCol='prediction', 
        metricName='f1'
    )
    
    evaluator_precision = MulticlassClassificationEvaluator(
        labelCol='lifestyle_label', 
        predictionCol='prediction', 
        metricName='weightedPrecision'
    )
    
    evaluator_recall = MulticlassClassificationEvaluator(
        labelCol='lifestyle_label', 
        predictionCol='prediction', 
        metricName='weightedRecall'
    )
    
    accuracy = evaluator_acc.evaluate(predictions)
    f1 = evaluator_f1.evaluate(predictions)
    precision = evaluator_precision.evaluate(predictions)
    recall = evaluator_recall.evaluate(predictions)
    
    print(f"\n🏃 Lifestyle Model Performance:")
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   F1 Score:  {f1:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    
    # Save model
    print("\nSaving model...")
    os.makedirs('models/lifestyle_model', exist_ok=True)
    rf_model.write().overwrite().save('models/lifestyle_model')
    print("✓ Lifestyle model saved")
    
    # Unpersist cached data
    train_data.unpersist()
    test_data.unpersist()
    
    # Return metrics
    metrics = {
        'accuracy': float(accuracy),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall)
    }
    
    return rf_model, metrics

def main():
    """Main training pipeline"""
    
    print("="*60)
    print("HEALTHCARE MODEL TRAINING PIPELINE")
    print("Using RandomForest for High Accuracy (85%+ target)")
    print("="*60)
    
    spark = create_spark_session()
    
    try:
        print("\n[1/5] Loading processed data...")
        
        # Check if file exists
        parquet_path = 'data/processed/healthcare_cleaned.parquet'
        if not os.path.exists(parquet_path):
            print(f"\nERROR: {parquet_path} not found!")
            print("Please run: python scripts/2_preprocess_data.py")
            sys.exit(1)
        
        df = spark.read.parquet(parquet_path)
        
        total_count = df.count()
        print(f"✓ Data loaded: {total_count:,} rows")
        
        # Verify required columns exist
        print("\n[2/5] Verifying data integrity...")
        required_cols = ['features', 'disease_label', 'diet_label', 'lifestyle_label']
        missing_cols = [c for c in required_cols if c not in df.columns]
        
        if missing_cols:
            print(f"\nERROR: Missing required columns: {missing_cols}")
            print("Please re-run: python scripts/2_preprocess_data.py")
            sys.exit(1)
        
        print("✓ All required columns present")
        
        # Check for null values
        df = df.dropna(subset=required_cols)
        
        # Split data: 80% train, 20% test
        print("\n[3/5] Splitting data (80% train, 20% test)...")
        train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
        
        train_count = train_data.count()
        test_count = test_data.count()
        
        print(f"✓ Training set: {train_count:,} rows (80%)")
        print(f"✓ Test set: {test_count:,} rows (20%)")
        
        # Train all models
        print("\n[4/5] Training models with RandomForest...")
        print("NOTE: Each model may take 5-15 minutes depending on your system")
        print("RandomForest is slower but gives much better accuracy!")
        
        disease_model, disease_metrics = train_disease_model(train_data, test_data)
        diet_model, diet_metrics = train_diet_model(train_data, test_data)
        lifestyle_model, lifestyle_metrics = train_lifestyle_model(train_data, test_data)
        
        # Combine all metrics
        print("\n[5/5] Saving metrics...")
        all_metrics = {
            'disease': disease_metrics,
            'diet': diet_metrics,
            'lifestyle': lifestyle_metrics,
            'train_samples': int(train_count),
            'test_samples': int(test_count),
            'total_samples': int(train_count + test_count),
            'algorithm': 'RandomForest'
        }
        
        # Save metrics to JSON file
        os.makedirs('models', exist_ok=True)
        metrics_file = 'models/training_metrics.json'
        
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=4)
        
        print(f"✓ Metrics saved to {metrics_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"\n📊 Dataset:")
        print(f"  - Training samples: {train_count:,}")
        print(f"  - Test samples: {test_count:,}")
        print(f"  - Total: {train_count + test_count:,}")
        
        print(f"\n🎯 Disease Model (RandomForest):")
        print(f"  - Accuracy: {disease_metrics['accuracy']:.4f} ({disease_metrics['accuracy']*100:.2f}%)")
        print(f"  - F1 Score: {disease_metrics['f1']:.4f}")
        print(f"  - Precision: {disease_metrics['precision']:.4f}")
        print(f"  - Recall: {disease_metrics['recall']:.4f}")
        
        print(f"\n🥗 Diet Model (RandomForest):")
        print(f"  - Accuracy: {diet_metrics['accuracy']:.4f} ({diet_metrics['accuracy']*100:.2f}%)")
        print(f"  - F1 Score: {diet_metrics['f1']:.4f}")
        print(f"  - Precision: {diet_metrics['precision']:.4f}")
        print(f"  - Recall: {diet_metrics['recall']:.4f}")
        
        print(f"\n🏃 Lifestyle Model (RandomForest):")
        print(f"  - Accuracy: {lifestyle_metrics['accuracy']:.4f} ({lifestyle_metrics['accuracy']*100:.2f}%)")
        print(f"  - F1 Score: {lifestyle_metrics['f1']:.4f}")
        print(f"  - Precision: {lifestyle_metrics['precision']:.4f}")
        print(f"  - Recall: {lifestyle_metrics['recall']:.4f}")
        
        avg_acc = (disease_metrics['accuracy'] + diet_metrics['accuracy'] + lifestyle_metrics['accuracy']) / 3
        print(f"\n📈 Overall Performance:")
        print(f"  - Average Accuracy: {avg_acc:.4f} ({avg_acc*100:.2f}%)")
        
        if avg_acc >= 0.85:
            print(f"\n🎉 SUCCESS! Achieved target accuracy of 85%+")
        elif avg_acc >= 0.75:
            print(f"\n✓ Good accuracy achieved. Close to 85% target.")
        else:
            print(f"\n⚠️  Accuracy below target. Consider:")
            print(f"     - Regenerating data with stronger correlations")
            print(f"     - Increasing numTrees or maxDepth")
            print(f"     - Adding more engineered features")

    except Exception as e:
        print(f"\n❌ ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        print("\n🔧 Troubleshooting:")
        print("  1. Check if data/processed/healthcare_cleaned.parquet exists")
        print("  2. Verify Java is installed: java -version")
        print("  3. Check JAVA_HOME environment variable")
        print("  4. Try closing other applications to free memory")
        print("  5. Re-run: python scripts/2_preprocess_data.py")
        sys.exit(1)
    finally:
        print("\nCleaning up...")
        spark.stop()
        print("✓ Spark session stopped")

if __name__ == "__main__":
    main()