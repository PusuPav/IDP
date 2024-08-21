import boto3
import json
import pandas as pd
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.context import SparkContext
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType

# Initialize AWS clients
s3_client = boto3.client('s3')
textract_client = boto3.client('textract')
bedrock_client = boto3.client('bedrock-runtime')

# Initialize Glue context
sc = SparkContext()
glue_context = GlueContext(sc)
spark = glue_context.spark_session
job = Job(glue_context)

def extract_text_from_s3(bucket, key):
    """Extract text from a PDF file in S3 using Textract"""
    response = textract_client.start_document_text_detection(
        DocumentLocation={'S3Object': {'Bucket': bucket, 'Name': key}}
    )
    job_id = response['JobId']

    # Wait for the Textract job to complete
    while True:
        response = textract_client.get_document_text_detection(JobId=job_id)
        status = response['JobStatus']
        if status in ['SUCCEEDED', 'FAILED']:
            break

    if status == 'SUCCEEDED':
        text = ''
        for item in response['Blocks']:
            if item['BlockType'] == 'LINE':
                text += item['Text'] + '\n'
        return text
    else:
        return None

def parse_gas_statement(text):
    """Parse the extracted text into structured data"""
    # Implement parsing logic here based on the structure of your gas statements
    # This is a simplified example and should be adapted to your specific format
    data = {
        'date': None,
        'operator': None,
        'gross_wellhead': None,
        'net_delivered': None,
        'fees': {},
        'plant_products': {},
        'residue_gas': {}
    }

    lines = text.split('\n')
    for line in lines:
        if 'Date:' in line:
            data['date'] = line.split('Date:')[1].strip()
        elif 'Operator:' in line:
            data['operator'] = line.split('Operator:')[1].strip()
        elif 'Gross Wellhead' in line:
            data['gross_wellhead'] = float(line.split()[-1])
        elif 'Net Delivered' in line:
            data['net_delivered'] = float(line.split()[-1])
        # Add more parsing logic for fees, plant products, and residue gas

    return data

def analyze_with_bedrock(data, contract_terms):
    """Use AWS Bedrock to analyze the gas statement data and identify discrepancies"""
    prompt = f"""
    Analyze the following gas statement data against the provided contract terms:

    Gas Statement:
    {json.dumps(data, indent=2)}

    Contract Terms:
    {json.dumps(contract_terms, indent=2)}

    Identify any discrepancies between the billed amounts and the contractual terms.
    Provide a detailed explanation of each discrepancy found.
    """

    response = bedrock_client.invoke_model(
        modelId='anthropic.claude-v2',
        contentType='application/json',
        accept='application/json',
        body=json.dumps({
            "prompt": prompt,
            "max_tokens_to_sample": 1000,
            "temperature": 0.5,
            "top_p": 1,
            "stop_sequences": []
        })
    )

    return json.loads(response['body'].read())['completion']

# UDF to process each gas statement
@udf(returnType=StringType())
def process_gas_statement(bucket, key, contract_terms):
    text = extract_text_from_s3(bucket, key)
    if text:
        data = parse_gas_statement(text)
        analysis = analyze_with_bedrock(data, contract_terms)
        return json.dumps({'data': data, 'analysis': analysis})
    return None

def main():
    # List gas statement files in S3 bucket
    bucket_name = 'your-gas-statements-bucket'
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix='gas_statements/')

    # Create a DataFrame with S3 file information
    file_data = [{'bucket': bucket_name, 'key': obj['Key']} for obj in response['Contents']]
    df = spark.createDataFrame(file_data)

    # Load contract terms (this should be adapted to your specific storage method)
    contract_terms = {
        'fees': {
            'compression': 0.24,
            'gathering': 0.23,
            'processing': 0.39
        },
        'plant_product_recoveries': {
            'ethane': 0.78,
            'propane': 0.94,
            'butane': 0.97
        }
    }

    # Process each gas statement
    processed_df = df.withColumn(
        'processed_data',
        process_gas_statement(col('bucket'), col('key'), json.dumps(contract_terms))
    )

    # Write results to Glue Data Catalog
    glue_context.write_dynamic_frame.from_options(
        frame=glue_context.create_dynamic_frame.from_spark_dataframe(processed_df),
        connection_type="s3",
        connection_options={"path": "s3://your-output-bucket/processed_gas_statements/"},
        format="parquet"
    )

    # Create a Glue table
    glue_client = boto3.client('glue')
    glue_client.create_table(
        DatabaseName='your_database_name',
        TableInput={
            'Name': 'processed_gas_statements',
            'StorageDescriptor': {
                'Columns': [
                    {'Name': 'bucket', 'Type': 'string'},
                    {'Name': 'key', 'Type': 'string'},
                    {'Name': 'processed_data', 'Type': 'string'}
                ],
                'Location': 's3://your-output-bucket/processed_gas_statements/',
                'InputFormat': 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat',
                'OutputFormat': 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat',
                'SerdeInfo': {
                    'SerializationLibrary': 'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe'
                }
            },
            'TableType': 'EXTERNAL_TABLE'
        }
    )

    job.commit()

if __name__ == "__main__":
    main()