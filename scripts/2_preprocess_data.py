"""
Script 2: Data Preprocessing with PySpark
- Load raw CSV
- Clean and transform features
- Feature engineering with StringIndexer, OneHotEncoder, VectorAssembler
- Save as Parquet
FIXED: Properly handles NaN values from data generation
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnan, isnull
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, Imputer
from pyspark.ml import Pipeline
import os
import sys

def create_spark_session():
    """Initialize Spark session"""
    try:
        spark = SparkSession.builder \
            .appName("Healthcare Data Preprocessing") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .getOrCreate()
        
        spark.sparkContext.setLogLevel("WARN")
        return spark
    except Exception as e:
        print(f"Error creating Spark session: {e}")
        print("\nTroubleshooting:")
        print("1. Check Java installation: java -version")
        print("2. Ensure Java 8 or 11 is installed")
        print("3. Set JAVA_HOME environment variable")
        print("4. Verify PySpark: pip show pyspark")
        sys.exit(1)

def preprocess_data(spark):
    """Load and preprocess healthcare data"""
    
    csv_path = 'data/raw/healthcare_data.csv'
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"\nError: {csv_path} not found!")
        print("Please run 1_generate_data.py first to generate the dataset.")
        sys.exit(1)
    
    print("Loading raw data...")
    try:
        df = spark.read.csv(csv_path, header=True, inferSchema=True)
    except Exception as e:
        print(f"\nError loading CSV: {e}")
        print("The CSV file may be corrupted. Try re-running 1_generate_data.py")
        sys.exit(1)
    
    row_count = df.count()
    col_count = len(df.columns)
    
    print(f"Original data shape: {row_count} rows, {col_count} columns")
    
    if row_count == 0:
        print("\nError: Dataset is empty!")
        print("Please re-run 1_generate_data.py to generate data.")
        sys.exit(1)
    
    # Select only required columns and drop extra ones from Kaggle merge
    required_cols = ['age', 'sex', 'bmi', 'bp_systolic', 'glucose', 'cholesterol',
                     'sleep_hours', 'stress_level', 'activity_level', 'diet_type',
                     'smoking', 'alcohol_units', 'disease', 'diet_plan', 'lifestyle_plan']
    
    # Check which columns exist
    existing_cols = [c for c in required_cols if c in df.columns]
    missing_cols = [c for c in required_cols if c not in df.columns]
    
    if missing_cols:
        print(f"\nError: Missing required columns: {missing_cols}")
        print("Please re-run 1_generate_data.py")
        sys.exit(1)
    
    # Select only required columns
    df = df.select(required_cols)
    
    print(f"Selected {len(required_cols)} required columns")
    
    # Show null counts before cleaning
    print("\nChecking for null values...")
    null_counts = df.select([col(c).isNull().cast("int").alias(c) for c in required_cols])
    null_summary = null_counts.groupBy().sum()
    
    # Handle missing values - only drop rows where critical columns are null
    critical_cols = ['age', 'sex', 'bmi', 'bp_systolic', 'glucose', 'cholesterol', 
                     'disease', 'diet_plan', 'lifestyle_plan']
    
    print(f"Dropping rows with nulls in critical columns: {critical_cols}")
    
    # Drop rows where any critical column is null
    for col_name in critical_cols:
        df = df.filter(col(col_name).isNotNull())
    
    # For non-critical numeric columns, fill with median
    numeric_cols = ['sleep_hours', 'alcohol_units']
    
    # Create imputer for numeric columns
    imputer = Imputer(
        inputCols=numeric_cols,
        outputCols=[f"{c}_imputed" for c in numeric_cols]
    ).setStrategy("median")
    
    df = imputer.fit(df).transform(df)
    
    # Replace original columns with imputed ones
    for col_name in numeric_cols:
        df = df.withColumn(col_name, col(f"{col_name}_imputed")).drop(f"{col_name}_imputed")
    
    # For categorical columns, fill with mode
    categorical_cols = ['stress_level', 'activity_level', 'diet_type', 'smoking']
    
    for col_name in categorical_cols:
        # Get mode (most frequent value)
        mode_value = df.groupBy(col_name).count().orderBy(col("count").desc()).first()[0]
        df = df.fillna({col_name: mode_value})
    
    final_count = df.count()
    print(f"After cleaning: {final_count} rows ({(final_count/row_count)*100:.1f}% retained)")
    
    if final_count < 1000:
        print("\nWarning: Very few rows remaining after cleaning!")
        print("Consider checking data quality or re-generating dataset.")
    
    # Cast numeric columns to proper types
    all_numeric_cols = ['age', 'bmi', 'bp_systolic', 'glucose', 'cholesterol', 
                        'sleep_hours', 'alcohol_units']
    
    for col_name in all_numeric_cols:
        df = df.withColumn(col_name, col(col_name).cast('double'))
    
    # Categorical columns to encode
    categorical_to_encode = ['sex', 'stress_level', 'activity_level', 'diet_type', 'smoking']
    
    # String Indexing
    indexers = [StringIndexer(inputCol=col_name, outputCol=col_name+"_index", 
                               handleInvalid="keep") 
                for col_name in categorical_to_encode]
    
    # One-Hot Encoding
    encoders = [OneHotEncoder(inputCol=col_name+"_index", outputCol=col_name+"_encoded",
                             dropLast=False)  # Don't drop last to avoid issues
                for col_name in categorical_to_encode]
    
    # Index target variables
    disease_indexer = StringIndexer(inputCol="disease", outputCol="disease_label", 
                                    handleInvalid="keep")
    diet_indexer = StringIndexer(inputCol="diet_plan", outputCol="diet_label", 
                                  handleInvalid="keep")
    lifestyle_indexer = StringIndexer(inputCol="lifestyle_plan", outputCol="lifestyle_label", 
                                       handleInvalid="keep")
    
    # Create pipeline for preprocessing
    pipeline_stages = indexers + encoders + [disease_indexer, diet_indexer, lifestyle_indexer]
    pipeline = Pipeline(stages=pipeline_stages)
    
    print("Fitting preprocessing pipeline...")
    try:
        model = pipeline.fit(df)
        df_transformed = model.transform(df)
    except Exception as e:
        print(f"\nError in preprocessing pipeline: {e}")
        print("This may be due to data format issues.")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Assemble feature vector
    feature_cols = all_numeric_cols + [col_name+"_encoded" for col_name in categorical_to_encode]
    
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw",
                                handleInvalid="skip")
    df_transformed = assembler.transform(df_transformed)
    
    # Scale features
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", 
                           withStd=True, withMean=True)
    scaler_model = scaler.fit(df_transformed)
    df_final = scaler_model.transform(df_transformed)
    
    # Select final columns
    final_cols = ['age', 'sex', 'bmi', 'bp_systolic', 'glucose', 'cholesterol', 
                  'sleep_hours', 'stress_level', 'activity_level', 'diet_type', 
                  'smoking', 'alcohol_units', 'disease', 'diet_plan', 'lifestyle_plan',
                  'disease_label', 'diet_label', 'lifestyle_label', 'features']
    
    df_final = df_final.select(final_cols)
    
    # Save as Parquet
    os.makedirs('data/processed', exist_ok=True)
    output_path = 'data/processed/healthcare_cleaned.parquet'
    
    print(f"Saving processed data to {output_path}...")
    try:
        df_final.write.mode('overwrite').parquet(output_path)
    except Exception as e:
        print(f"\nError saving parquet file: {e}")
        print("Check disk space and write permissions.")
        sys.exit(1)
    
    # Save models for later use
    os.makedirs('models/scaler_model', exist_ok=True)
    try:
        scaler_model.write().overwrite().save('models/scaler_model')
    except Exception as e:
        print(f"\nWarning: Could not save scaler model: {e}")
    
    print("\n✓ Preprocessing complete!")
    print(f"Processed data shape: {df_final.count()} rows")
    
    # Show sample
    print("\nSample processed data:")
    df_final.select('age', 'sex', 'bmi', 'disease', 'disease_label').show(5)
    
    return df_final

# Main execution
if __name__ == "__main__":
    spark = create_spark_session()
    
    try:
        df_processed = preprocess_data(spark)
        print("\n✓ All preprocessing steps completed successfully!")
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise
    finally:
        spark.stop()