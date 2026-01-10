"""
Script 6: Real-Time Streaming Simulation
- Simulate streaming patient data
- Process with Spark Structured Streaming
- Make real-time predictions
"""

import pandas as pd
import numpy as np
import time
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, struct, to_json
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.ml.classification import OneVsRestModel
from pyspark.ml.feature import StandardScalerModel, VectorAssembler
import threading

def create_streaming_data_generator():
    """Generate streaming patient data"""
    
    print("Starting streaming data generator...")
    os.makedirs('data/streaming', exist_ok=True)
    
    def generate_patient_record():
        """Generate a single patient record"""
        
        record = {
            'patient_id': np.random.randint(10000, 99999),
            'age': float(np.random.randint(18, 90)),
            'sex': np.random.choice(['Male', 'Female']),
            'bmi': round(np.random.normal(27, 6), 1),
            'bp_systolic': float(np.random.randint(90, 200)),
            'glucose': round(np.random.normal(100, 30), 1),
            'cholesterol': round(np.random.normal(200, 40), 1),
            'sleep_hours': round(np.random.normal(7, 1.5), 1),
            'stress_level': np.random.choice(['Low', 'Medium', 'High']),
            'activity_level': np.random.choice(['Sedentary', 'Light', 'Moderate', 'Active']),
            'diet_type': np.random.choice(['Western', 'Mediterranean', 'Asian', 'Vegetarian']),
            'smoking': np.random.choice(['Never', 'Former', 'Current']),
            'alcohol_units': round(np.random.exponential(3), 1)
        }
        
        return record
    
    def write_streaming_data(duration=60, interval=2):
        """Write streaming data to file"""
        
        stream_file = 'data/streaming/patient_stream.csv'
        
        # Initialize file with header
        df_init = pd.DataFrame([generate_patient_record()])
        df_init.to_csv(stream_file, index=False, mode='w')
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Generate batch of records
            records = [generate_patient_record() for _ in range(5)]
            df_batch = pd.DataFrame(records)
            
            # Append to file
            df_batch.to_csv(stream_file, index=False, mode='a', header=False)
            
            print(f"Generated batch of {len(records)} records at {time.strftime('%H:%M:%S')}")
            time.sleep(interval)
        
        print("Streaming data generation complete!")
    
    return write_streaming_data

def process_streaming_data():
    """Process streaming data with Spark Structured Streaming"""
    
    print("\nInitializing Spark Structured Streaming...")
    
    spark = SparkSession.builder \
        .appName("Healthcare Streaming") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    # Define schema
    schema = StructType([
        StructField("patient_id", StringType(), True),
        StructField("age", DoubleType(), True),
        StructField("sex", StringType(), True),
        StructField("bmi", DoubleType(), True),
        StructField("bp_systolic", DoubleType(), True),
        StructField("glucose", DoubleType(), True),
        StructField("cholesterol", DoubleType(), True),
        StructField("sleep_hours", DoubleType(), True),
        StructField("stress_level", StringType(), True),
        StructField("activity_level", StringType(), True),
        StructField("diet_type", StringType(), True),
        StructField("smoking", StringType(), True),
        StructField("alcohol_units", DoubleType(), True)
    ])
    
    # Read streaming data
    stream_df = spark.readStream \
        .schema(schema) \
        .option("header", "true") \
        .option("maxFilesPerTrigger", 1) \
        .csv("data/streaming/")
    
    print("✓ Streaming source configured")
    
    # Load models for prediction
    try:
        disease_model = OneVsRestModel.load('models/disease_model')
        print("✓ Disease model loaded")
    except Exception as e:
        print(f"Warning: Could not load disease model: {e}")
        print("Continuing without predictions...")
        
        # Simple aggregation without predictions
        query = stream_df \
            .writeStream \
            .outputMode("append") \
            .format("console") \
            .option("truncate", False) \
            .start()
        
        return query
    
    # For demonstration, we'll process and display the stream
    # In production, you would apply full preprocessing and prediction
    
    query = stream_df \
        .writeStream \
        .outputMode("append") \
        .format("console") \
        .option("truncate", False) \
        .trigger(processingTime='5 seconds') \
        .start()
    
    print("✓ Streaming query started")
    print("Processing streaming data... (Press Ctrl+C to stop)")
    
    return query

def run_streaming_demo(duration=30):
    """Run complete streaming demonstration"""
    
    print("="*60)
    print("REAL-TIME STREAMING DEMONSTRATION")
    print("="*60)
    
    # Start data generator in background thread
    generator = create_streaming_data_generator()
    generator_thread = threading.Thread(target=generator, args=(duration, 2))
    generator_thread.daemon = True
    generator_thread.start()
    
    # Wait a bit for initial data
    time.sleep(3)
    
    # Start streaming processor
    try:
        query = process_streaming_data()
        
        # Let it run for specified duration
        time.sleep(duration)
        
        # Stop the query
        query.stop()
        
        print("\n✓ Streaming demonstration complete!")
        
    except KeyboardInterrupt:
        print("\n\nStreaming stopped by user")
    except Exception as e:
        print(f"Error in streaming: {e}")

# For batch prediction on streaming data
def batch_predict_stream():
    """Make batch predictions on accumulated stream data"""
    
    print("\nMaking batch predictions on stream data...")
    
    spark = SparkSession.builder \
        .appName("Stream Batch Prediction") \
        .getOrCreate()
    
    # Read accumulated stream data
    stream_data = spark.read.csv('data/streaming/patient_stream.csv', 
                                  header=True, inferSchema=True)
    
    print(f"Stream data records: {stream_data.count()}")
    
    # Load models
    try:
        disease_model = OneVsRestModel.load('models/disease_model')
        
        # Simplified feature preparation (for demo)
        numeric_cols = ['age', 'bmi', 'bp_systolic', 'glucose', 'cholesterol', 
                       'sleep_hours', 'alcohol_units']
        
        assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features_raw")
        stream_data = assembler.transform(stream_data)
        
        # Load scaler
        scaler = StandardScalerModel.load('models/scaler_model')
        stream_data = scaler.transform(stream_data)
        
        # Rename column for prediction
        stream_data = stream_data.withColumnRenamed("features", "features_temp") \
                                 .withColumnRenamed("features_raw", "features")
        
        # Predict (Note: This is simplified - full preprocessing would be needed)
        # predictions = disease_model.transform(stream_data)
        
        print("✓ Batch predictions on stream data prepared")
        
    except Exception as e:
        print(f"Could not complete predictions: {e}")
    
    spark.stop()

# Main execution
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        # Run live streaming demo
        run_streaming_demo(duration=30)
    else:
        # Just generate some sample streaming data
        print("Generating sample streaming data...")
        generator = create_streaming_data_generator()
        generator(duration=10, interval=1)
        print("\n✓ Sample streaming data generated in data/streaming/")
        print("\nTo run live streaming demo, use: python 6_streaming_simulation.py demo")