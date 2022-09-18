import os
import pandas as pd
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.spatial import distance

# 경고끄기 (option)
import warnings
warnings.filterwarnings('ignore')

#%%

def make_dict():
    category_model_dict = dict()

    # 카테고리 별 모델은 변경될 수 있음
    category_model_dict['wallet'] = 'R50x1_object'
    category_model_dict['phone'] = 'R50x1_equipment'
    category_model_dict['cap'] = 'R50x1_clothing'
    category_model_dict['card'] = 'R50x1_device'
    category_model_dict['bag'] = 'R50x1_consumer'
    category_model_dict['book'] = 'Mobilenet_v3'
    category_model_dict['shopping_bag'] = 'Mobilenet_v2'
    category_model_dict['earphones'] = 'R50x1_device'
    category_model_dict['car_key'] = 'Efficientnet_lite0'
    category_model_dict['shoes'] = 'R101x3'
    category_model_dict['document'] = 'R50x1_object'
    category_model_dict['watch'] = 'R50x1_equipment'
     # 아래 카테고리들은 테스트 후 다시 모델 삽입 예정임 임시로 넣어 놈
    category_model_dict['cloth'] = 'R50x1_object'
    category_model_dict['gloves'] = 'R50x1_object'
    category_model_dict['muffler'] = 'R50x1_object'
    category_model_dict['necklace'] = 'R50x1_object'
    category_model_dict['ring'] = 'R50x1_object'
    
    return category_model_dict

#%%

def select_model(category):
    # 카테고리를 입력받아 모델을 생성해서 모델명과 생성된 모델을 리턴함
    model_dict = make_dict()
    model_name = model_dict[category]
    model_path = '../models/' + model_name
    tmp_model = tf.saved_model.load(model_path)
    
    layer = hub.KerasLayer(tmp_model, input_shape=(224, 224)+(3,))
    model = tf.keras.Sequential([layer])
    model.build([None, 244, 244, 3])
    
    return model_name, model

#%%
# 함수부

# 이미지 객체탐지
def crop(images):
    os.system(f"python yolov5/detect.py --source {images} --weights yolov5/runs/train/cate_29_epoch10/weights/best.pt --conf 0.4 --save-crop")

#%%

def extract(file):
    file = Image.open(file).convert('RGB').resize((224, 224))
    file = np.array(file)/255.0 # 정규화

    embedding = model.predict(file[np.newaxis, ...])
    feature_np = np.array(embedding)
    flattened_feature = feature_np.flatten()

    return flattened_feature

#%%

def get_dataframe(category):
    global output_path    
    global model_name
    tmp_filename = np.load(output_path+f'{category}_filename({model_name}).npy', allow_pickle=True)
    tmp_output = np.load(output_path+f'{category}_output({model_name}).npy', allow_pickle=True)
    df = pd.DataFrame({'filename':tmp_filename, 'output':tmp_output})
    return df

#%%

def get_cos_sim(file, category, metric='cosine'):
    before_time = time.time()
    file2vec = extract(file) # 이미지 벡터화
    df = get_dataframe(category) # 데이터프레임 가져오기
    df = df.append({'filename':file, 'output':file2vec}, ignore_index=True)
    
    cos_sim_array = np.zeros((len(df)))
    for i in range(0, len(df)):
        cos_sim_array[i] = distance.cdist([file2vec] , [df.iloc[i, 1]], metric)[0] # 벡터화된 이미지 기준
    df['cos_sim']=cos_sim_array
    after_time = time.time()
    runtime = after_time-before_time
    return df, runtime # 런타임 비교용

#%%

# crop된 파일을 인풋 파일으로 넣어줘야함
def search_img(category, cropped_file, threshold=0.4):
    global image_path
    global output_path
    cos_sim_df, runtime = get_cos_sim(cropped_file, category=category)
    df_top_sim = cos_sim_df[cos_sim_df.cos_sim <= threshold].sort_values(by='cos_sim')[:50]
    
    return df_top_sim.filename.values                               

#%%

# 메인
if __name__ == '__main__':
    # crop(input_file) # 인풋파일 잘라내기 + 카테고리 판단해주기
    category = input('category :') # 카테고리 이미지 전달
    input_file = input('image_path :') # 입력 이미지 전달
    start_time = time.time()
    image_path = f"../crops/{category}/" # crops된 이미지 경로
    output_path = f"../vector_frame/{category}/" # npy파일 보관된 경로
    if not os.path.exists(output_path):
        print('디렉토리가 없으므로 생성합니다.')
        os.mkdir(output_path)
    
    model_name, model = select_model(category)
    
    # threshold의 디폴트 값은 0.4이고, search_img 함수의 마지막 인자로 넣어주면 바꿀 수 있음
    print(search_img(category, input_file)) # 인자로 input_file이 아닌 crops된 파일의 경로로 바꿔줘야함
    print(f'소요시간 : {time.time() - start_time:.3f}초') # 테스트용 코드 -> 추후 삭제 ㄱ
    

# /Users/iseongmin/workspaces/project2/crops/phone/F2022030100000776-1.jpg
# In[ ]:




