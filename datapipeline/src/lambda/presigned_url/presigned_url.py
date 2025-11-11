import json 
import boto3
import uuid 
from datetime import datetime, timedelta
import sys 
import os 

s3_client = boto3.client('s3')
BUCKET_NAME = os.environ['BUCKET_NMAE']

def lambda_handler(event, context) -> dict: 
    try:
        if 'body' in event:
            body = json.loads(event['body'])
        else:
            body = {}
        
        job_id = f"job-{datetime.now}"
    except Exception as e:
        print("Error:", str(e))
        return {
            'statusCode': 500
        }
    