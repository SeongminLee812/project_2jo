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

# 경고끄기 (option)
import warnings
warnings.filterwarnings('ignore')



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
    # global image_path
    global output_path
    cos_sim_df = get_cos_sim(cropped_file, category=category)
    df_top_sim = cos_sim_df[cos_sim_df.cos_sim <= threshold].sort_values(by='cos_sim')[:50]

    return df_top_sim.filename.values


# %%

# 메인
if __name__ == '__main__':
    '''
    코드 유지관리를 위해서는 가급적 클래스 구조로 작성하기를 추천합니다. 
    '''
    model_category = ModelCategory()
    model_category.set_model()

#-------------
    # crop(input_file) # 인풋파일 잘라내기 + 카테고리 판단해주기
    category = input('category :')  # 카테고리 전달
    input_file = input('input_file :')  # 입력 이미지 전달
    model = model_category.model_dict[category]['model']
    model_name = model_category.model_dict[category]['model_name']
    start_time = time.time()
    # image_path = f"../crops/{category}/" # crops된 이미지 경로
    output_path = f"../vector_frame/{category}/"  # npy파일 보관된 경로
    if not os.path.exists(output_path):
        print('디렉토리가 없으므로 생성합니다.')
        os.mkdir(output_path)

    # threshold의 디폴트 값은 0.4이고, search_img 함수의 마지막 인자로 넣어주면 바꿀 수 있음
    print(search_img(category, input_file))  # 인자로 input_file이 아닌 crops된 파일의 경로로 바꿔줘야함
    print(f'소요시간 : {time.time() - start_time:.3f}초')  # 테스트용 코드 -> 추후 삭제

    # app.run()
