import sys
from boto.s3.key import Key
import logging
import boto3
from botocore.exceptions import ClientError
import os
import json
import csv
from datetime import datetime
import argparse

creds = None
with open('cred.json', 'r+') as infile:
    creds = json.load(infile)


AWS_ACCESS_KEY_ID = creds['AWS_ACCESS_KEY_ID']
AWS_SECRET_ACCESS_KEY = creds['AWS_SECRET_ACCESS_KEY']

def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    print(f'uploading file {object_name}...')


    # Upload the file
    s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True

def incr_model_ver(stats):
    stats[0] = str(int(stats[0]) + 1)
    return stats

def write_stats(stats):
    with open('stats.txt', 'w+') as infile:
        for line in stats:
            infile.write(line)

def del_best():
    os.remove('best.txt')

# get the best model from s3
def get_best():

    s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    try:
        s3_client.download_file('buffnet', 'best.txt', 'best.txt')
    except ClientError:
        return None
    
    with open('best.txt', 'r+') as infile:
        fl = infile.readline()
        fl = float(fl)
        return fl

def handle_model(tag, model_dir):

    # open the model dir
    model_file = os.path.join(model_dir, 'weights', 'best.pt')
    csv_file = os.path.join(model_dir, 'results.csv')
    _map = None

    # find the data csv file. Get the results
    with open(csv_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        li = []
        for line in reader:
            li.append(line)
        #map
        _map = float(li[-1][6])

    # compare this with get_best()
    obj = s3_client.get_object(Bucket='buffnet', Key='best.pt')
    """

    try:
        acl = self.object.Acl()
        # Putting an ACL overwrites the existing ACL, so append new grants
        # if you want to preserve existing grants.
        print(acl.grants)
        return
        grants = []
        grants.append({
            'Grantee': {
                'Type': 'AmazonCustomerByEmail',
                'EmailAddress': email
            },
            'Permission': 'READ'
        })
        acl.put(
            AccessControlPolicy={
                'Grants': grants,
                'Owner': acl.owner
            }
        )
    except ClientError:
        print("client error")
    """

    best_aws = get_best()
    if best_aws is None or best_aws > _map:
        #upload best.pt
        with open('best.txt', 'w+') as outfile:
            outfile.write(str(_map))
        upload_file('best.txt', 'buffnet')
        upload_file(model_file, 'buffnet', 'best.pt')

        # get the object

    else:
        now = datetime.now().strftime('%b-%d-%H:%M:%S')
        file_name = tag + '-' + now + '.pt'
        upload_file(model_file, 'buffnet', object_name=file_name)
        
        s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

        """ 
        obj = s3_client.get_object(Bucket='buffnet', Key=file_name)
         try:
            acl = self.object.Acl()
            # Putting an ACL overwrites the existing ACL, so append new grants
            # if you want to preserve existing grants.
            grants = []
            grants.append({
                'Grantee': {
                    'Type': 'AmazonCustomerByEmail',
                    'EmailAddress': email
                },
                'Permission': 'READ'
            })
            acl.put(
                AccessControlPolicy={
                    'Grants': grants,
                    'Owner': acl.owner
                }
            )
        except ClientError:
            print("this doesnt work")
        """

        

    del_best()

def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--model_dir", help="pass the yolov5 checkpoint to add to aws")
    parser.add_argument("--tag", help="the s3 bucket organizes models by tags")

    args = parser.parse_args()
    if not args.model_dir:
        print("Please pass a valid yolov5 checkpoint")
        return
    if not args.tag:
        print("Please pass a valid tag")
    else:
        handle_model(args.tag, args.model_dir)

if __name__ == "__main__":
    main()
