import os
from google.cloud import storage
import fire
import multiprocessing as mp
from multiprocessing import Pool, cpu_count


def upload_cameylon():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/manukastratta/Developer/CS329D/test-time-training-project/manuka/cs329d-project-385522-888630fdc09e.json"
    storage_client = storage.Client()

    bucket_name = 'cs329d-bucket'
    bucket = storage_client.bucket(bucket_name)

    folder_path = '/Users/manukastratta/Developer/CS329D/test-time-training-project/data/camelyon17_v1.0'
    f = 0
    for subdir, dirs, files in os.walk(folder_path):
        for file in files:
            f += 1
            file_path = os.path.join(subdir, file)
            relative_path = os.path.relpath(file_path, folder_path)
            blob = bucket.blob(relative_path)
            blob.upload_from_filename(file_path)
            if f % 1000 == 0:
                print("i: ", i)
                print("Uploaded file: ", relative_path)

def upload_blob(args):
    file_path, bucket_name, relative_path = args
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(relative_path)
    blob.upload_from_filename(file_path)

def upload_cameylon_parallelized():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/manukastratta/Developer/CS329D/test-time-training-project/manuka/cs329d-project-385522-888630fdc09e.json"

    bucket_name = 'cs329d-bucket'

    folder_path = '/Users/manukastratta/Developer/CS329D/test-time-training-project/data/camelyon17_v1.0'
    blobs = []
    for subdir, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(subdir, file)
            relative_path = os.path.relpath(file_path, folder_path)
            blobs.append((file_path, bucket_name, relative_path))

    with Pool(processes=cpu_count()) as pool:
        pool.map(upload_blob, blobs)

if __name__ == "__main__":
    fire.Fire()