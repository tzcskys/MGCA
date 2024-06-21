import pickle
import numpy as np
import pandas as pd
from mgca.constants import *
from sklearn.model_selection import train_test_split
import json

np.random.seed(0)


# create bounding boxes
def create_bbox(row):
    if row["Target"] == 0:
        return 0
    else:
        x1 = row["x"]
        y1 = row["y"]
        x2 = x1 + row["width"]
        y2 = y1 + row["height"]
        return [x1, y1, x2, y2]


def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.loads(file.readline())

    return data

def preprocess_tbx11k_data(test_fac=0.15):
    try:
        df_raw_train = read_json_file(TBX11K_ORIGINAL_TRAIN_CSV)
        df_raw_val = read_json_file(TBX11K_ORIGINAL_VAL_CSV)
    except:
        raise Exception(
            "Please make sure the the TBX11K dataset is \
            stored at {TBX11K_DATA_DIR}"
        )
    
    #### process the raw train data and split into train and val
    images_raw_train = pd.DataFrame(df_raw_train['images'])
    annotations_raw_train = pd.DataFrame(df_raw_train['annotations'])

    bbox_raw_train_data = []
    for _, row in annotations_raw_train.iterrows():
        bbox = row['bbox']
        x, y, width, height = bbox
        x_max = x + width
        y_max = y + height

        bbox_raw_train_data.append({
            'image_id': row['image_id'],
            'bbox_list': [x, y, x_max, y_max]
        })
    
    bbox_raw_train = pd.DataFrame(bbox_raw_train_data)

    ## aggregate multiple boxes for each image
    aggreate_bbox_raw_train = images_raw_train.merge(bbox_raw_train, left_on='id', right_on='image_id')

    ## Group by image_id and aggregate bounding boxes
    grouped_raw_train = aggreate_bbox_raw_train.groupby('image_id').agg({
        'file_name': 'first',
        'bbox_list': lambda x: list(x)
    }).reset_index()

    ## rename some columns
    grouped_raw_train.rename(columns={
        'file_name': 'patientId',
        'bbox_list': 'bbox'
        }, inplace=True)
    grouped_raw_train = grouped_raw_train.drop(columns=['image_id'])

    ## split data
    train_df, valid_df = train_test_split(grouped_raw_train, test_size=0.1, random_state=42)

    #### same process for test data
    images_raw_val = pd.DataFrame(df_raw_val['images'])
    annotations_raw_val = pd.DataFrame(df_raw_val['annotations'])

    bbox_raw_val_data = []
    for _, row in annotations_raw_val.iterrows():
        bbox = row['bbox']
        x, y, width, height = bbox
        x_max = x + width
        y_max = y + height

        bbox_raw_val_data.append({
            'image_id': row['image_id'],
            'bbox_list': [x, y, x_max, y_max]
        })
    
    bbox_raw_val = pd.DataFrame(bbox_raw_val_data)

    ## aggregate multiple boxes for each image
    aggreate_bbox_raw_val = images_raw_val.merge(bbox_raw_val, left_on='id', right_on='image_id')

    ## Group by image_id and aggregate bounding boxes
    grouped_raw_val = aggreate_bbox_raw_val.groupby('image_id').agg({
        'file_name': 'first',
        'bbox_list': lambda x: list(x)
    }).reset_index()

    ## rename some columns
    grouped_raw_val.rename(columns={
        'file_name': 'patientId',
        'bbox_list': 'bbox'
        }, inplace=True)
    grouped_raw_val = grouped_raw_val.drop(columns=['image_id'])

    test_df = grouped_raw_val


    print(f"Number of train samples: {len(train_df)}")
    print(train_df["patientId"].value_counts())
    print(f"Number of valid samples: {len(valid_df)}")
    print(valid_df["patientId"].value_counts())
    print(f"Number of test samples: {len(test_df)}")
    print(test_df["patientId"].value_counts())

    train_df.to_csv(TBX11K_TRAIN_CSV, index=False)
    valid_df.to_csv(TBX11K_VALID_CSV, index=False)
    test_df.to_csv(TBX11K_TEST_CSV, index=False)


if __name__ == "__main__":
    preprocess_tbx11k_data()