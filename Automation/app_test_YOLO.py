# from flask import render_template
# from flask import Flask, request
from py_src.model_category import ModelCategory
import sys
# app = Flask(__name__)
import os
import pandas as pd
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from PIL import Image
import numpy as np
import time
from scipy.spatial import distance
import torch
import glob

# 경고끄기 (option)
import warnings
warnings.filterwarnings('ignore')

#%%

def crop(input_file):
    print("--Start object detection--")
    current_dir = os.getcwd()
    os.chdir('/home/ubuntu/yolov5')
    os.system(f"python detect.py --source {input_file} --project /home/ubuntu/image_model/cropped_user_image --device cpu") # GPU:out of memory 해결을 위해 cpu로 돌림
    # --project 뒤에 파일이 저장될 경로를 지정
    os.chdir(current_dir)
    
#%%
    
def find_crops():
        # crops 폴더 찾기 (최신순)
    root = "/home/ubuntu/image_model/cropped_user_image/"
    folder_and_time_list = []

    for f_name in os.listdir(root):
        written_time = os.path.getctime(f"{root}{f_name}")
        folder_and_time_list.append((f_name, written_time))

    sorted_file_list = sorted(folder_and_time_list, key=lambda x: x[1], reverse=True)
    recent_dir = sorted_file_list[0][0]
    
    configfile = glob.glob(root +  recent_dir + '/**/*.jpg', recursive=True)
    return configfile
    
#%%

def extract(file):
    global model

    file = Image.open(file).convert('RGB').resize((224, 224))
    file = np.array(file) / 255.0  # 정규화

    embedding = model.predict(file[np.newaxis, ...])
    feature_np = np.array(embedding)
    flattened_feature = feature_np.flatten()

    return flattened_feature

# %%

def get_dataframe(category):
    global output_path
    global model_name
    tmp_filename = np.load(output_path + f'{category}_filename({model_name}).npy', allow_pickle=True)
    tmp_output = np.load(output_path + f'{category}_output({model_name}).npy', allow_pickle=True)
    df = pd.DataFrame({'filename': tmp_filename, 'output': tmp_output})
    return df


# %%

def get_cos_sim(file, category, metric='cosine'):
    file2vec = extract(file)  # 이미지 벡터화
    df = get_dataframe(category)  # 데이터프레임 가져오기
    df = df.append({'filename': file, 'output': file2vec}, ignore_index=True)
    cos_sim_array = np.zeros((len(df)))
    for i in range(0, len(df)):
        cos_sim_array[i] = distance.cdist([file2vec], [df.iloc[i, 1]], metric)[0]  # 벡터화된 이미지 기준
    df['cos_sim'] = cos_sim_array
    return df  # 런타임 비교용


# %%

# crop된 파일을 인풋 파일으로 넣어줘야함
def search_img(category, cropped_file, threshold=0.4):
    global output_path
    cos_sim_df = get_cos_sim(cropped_file, category=category)
    df_top_sim = cos_sim_df[cos_sim_df.cos_sim <= threshold].sort_values(by='cos_sim')[1:50]

    return df_top_sim.filename.values.tolist()


# %%

# 메인
if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # gpu cash 지워주기
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    model_category = ModelCategory()
    model_category.set_model01()
    # gpu 메모리 부족으로 절반 먼저 로딩, 캐시 초기화 후 나머지 절반 로딩 시도
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    model_category.set_model02()

#-------------

    category = input('category :')  # 카테고리 전달
    input_file = input('input_file :')  # 입력 이미지 전달
    start_time = time.time()
    crop(input_file)
    print(find_crops())
    configfile = find_crops()[0]
    
    model = model_category.model_dict[category]['model']
    model_name = model_category.model_dict[category]['model_name']
    threshold = model_category.model_dict[category]['threshold']
    
    output_path = f"/home/ubuntu/image_model/vector_frame_95/{category}/"  # npy파일 보관된 경로
    
    result = search_img(category, configfile, threshold=threshold) # 인자로 input_file이 아닌 crops된 파일의 경로로 바꿔줘야함
    print(result)
    print(type(result))
    print(f'소요시간 : {time.time() - start_time:.3f}초')  # 테스트용 코드 -> 추후 삭제

    # app.run()
    
