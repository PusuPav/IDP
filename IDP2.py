import json
import re
from typing import Dict, List
import boto3
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.context import SparkContext
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, StructType, StructField

# Initialize AWS clients
bedrock_client = boto3.client('bedrock-runtime')

# Initialize Glue context
sc = SparkContext()
glue_context = GlueContext(sc)
spark = glue_context.spark_session
job = Job(glue_context)

def parse_targa_agreement(text: str) -> Dict[str, any]:
    """Parse the Targa agreement text to extract key terms"""
    terms = {
        'gathering_fee': None,
        'sour_gas_treating_fee': None,
        'compression_fee': None,
        'gas_lift_fee': None,
        'rich_gas_transportation_and_processing_fee': None,
        'residue_redelivery_fee': None,
        'ngl_tf_fee': None,
        'admin_fee': None,
        'low_pressure_gathering_fl_u': None,
        'high_pressure_gathering_fl_u': None,
        'processing_fuel': None,
    }
    
    fee_patterns = {
        'gathering_fee': r'Gathering Fee.*?\$(\d+\.\d+)',
        'sour_gas_treating_fee': r'Sour Gas Treating Fee.*?\$(\d+\.\d+)',
        'compression_fee': r'Compression Fee.*?\$(\d+\.\d+)',
        'gas_lift_fee': r'Gas Lift Fee.*?\$(\d+\.\d+)',
        'rich_gas_transportation_and_processing_fee': r'Rich Gas Transportation and Processing Fee.*?\$(\d+\.\d+)',
        'residue_redelivery_fee': r'Residue Redelivery Fee.*?\$(\d+\.\d+)',
        'ngl_tf_fee': r'NGL T&F Fee.*?\$(\d+\.\d+)',
        'admin_fee': r'Admin Fee.*?\$(\d+\.\d+)',
        'low_pressure_gathering_fl_u': r'Low Pressure Gathering FL&U.*?(\d+\.\d+)',
        'high_pressure_gathering_fl_u': r'High Pressure Gathering FL&U.*?(\d+\.\d+)',
        'processing_fuel': r'Processing Fuel.*?(\d+\.\d+)',
    }
    
    for key, pattern in fee_patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        if match:
            terms[key] = float(match.group(1))
    
    return terms

def parse_gas_statement(text: str) -> Dict[str, any]:
    """Parse the gas statement text to extract key information"""
    data = {
        'date': None,
        'operator': None,
        'gross_wellhead': None,
        'net_delivered': None,
        'fees': {},
        'plant_products': {},
        'residue_gas': {},
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
        elif 'Fees' in line:
            fee_section = True
        elif 'Plant Products' in line:
            fee_section = False
        
        if 'Fee Value' in line:
            parts = line.split()
            fee_name = ' '.join(parts[:-2]).lower()
            fee_value = float(parts[-1].replace('$', '').replace(',', ''))
            data['fees'][fee_name] = fee_value
        
        if 'Ethane' in line or 'Propane' in line or 'Butane' in line or 'Gasoline' in line:
            parts = line.split()
            product_name = parts[0].lower()
            allocated_quantity = float(parts[3].replace(',', ''))
            price = float(parts[-2].replace('$', ''))
            value = float(parts[-1].replace('$', '').replace(',', ''))
            data['plant_products'][product_name] = {
                'allocated_quantity': allocated_quantity,
                'price': price,
                'value': value
            }
        
        if 'Residue' in line and 'mmbtu' in line.lower():
            parts = line.split()
            data['residue_gas']['quantity'] = float(parts[2].replace(',', ''))
            data['residue_gas']['price'] = float(parts[-2].replace('$', ''))
            data['residue_gas']['value'] = float(parts[-1].replace('$', '').replace(',', ''))
    
    return data

def analyze_with_bedrock(agreement_terms: Dict[str, any], statement_data: Dict[str, any]) -> str:
    """Use AWS Bedrock to analyze the gas statement data against the agreement terms"""
    prompt = f"""
    Analyze the following gas statement data against the provided contract terms:

    Gas Statement:
    {json.dumps(statement_data, indent=2)}

    Contract Terms:
    {json.dumps(agreement_terms, indent=2)}

    Identify any discrepancies between the billed amounts and the contractual terms.
    Provide a detailed explanation of each discrepancy found, including calculations where applicable.
    Focus on comparing fees, product allocations, and prices.
    """

    response = bedrock_client.invoke_model(
        modelId='anthropic.claude-v2',
        contentType='application/json',
        accept='application/json',
        body=json.dumps({
            "prompt": prompt,
            "max_tokens_to_sample": 2000,
            "temperature": 0.5,
            "top_p": 1,
            "stop_sequences": []
        })
    )

    return json.loads(response['body'].read())['completion']

# UDF to process each gas statement
@udf(returnType=StringType())
def process_gas_statement(statement_text: str, agreement_terms: str) -> str:
    statement_data = parse_gas_statement(statement_text)
    agreement_terms = json.loads(agreement_terms)
    analysis = analyze_with_bedrock(agreement_terms, statement_data)
    return json.dumps({'statement_data': statement_data, 'analysis': analysis})

def main():
    # Read the Targa agreement
    with open('Targa.pdf', 'r') as file:
        targa_text = file.read()
    
    # Parse the Targa agreement
    agreement_terms = parse_targa_agreement(targa_text)
    
    # Read the sample gas statement
    with open('sample gas statement.pdf', 'r') as file:
        gas_statement_text = file.read()
    
    # Create a DataFrame with the gas statement
    df = spark.createDataFrame([(gas_statement_text,)], ['statement_text'])
    
    # Process the gas statement
    processed_df = df.withColumn(
        'processed_data',
        process_gas_statement('statement_text', json.dumps(agreement_terms))
    )
    
    # Define the schema for the processed data
    processed_schema = StructType([
        StructField("statement_data", StringType(), True),
        StructField("analysis", StringType(), True)
    ])
    
    # Extract the processed data into separate columns
    final_df = processed_df.select(
        'statement_text',
        processed_df.processed_data.cast(processed_schema).alias('processed')
    ).select(
        'statement_text',
        'processed.statement_data',
        'processed.analysis'
    )
    
    # Write results to Glue Data Catalog
    glue_context.write_dynamic_frame.from_options(
        frame=glue_context.create_dynamic_frame.from_spark_dataframe(final_df),
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
                    {'Name': 'statement_text', 'Type': 'string'},
                    {'Name': 'statement_data', 'Type': 'string'},
                    {'Name': 'analysis', 'Type': 'string'}
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