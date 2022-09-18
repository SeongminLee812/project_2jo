
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

#%%

def make_dict():
    category_model_dict = dict()

    # 카테고리 별 모델은 변경될 수 있음
    category_model_dict['wallet'] = ['R50x1_object', 'https://tfhub.dev/google/experts/bit/r50x1/in21k/object/1']
    category_model_dict['phone'] = ['Efficientnet_b0', 'https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2']
    category_model_dict['cap']  = ['Efficientnet_lite0', 'https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2']
    category_model_dict['card'] = ['R50x1_object', 'https://tfhub.dev/google/experts/bit/r50x1/in21k/object/1']
    category_model_dict['bag'] = ['R50x1_object', 'https://tfhub.dev/google/experts/bit/r50x1/in21k/object/1']
    category_model_dict['book'] = ['R101x3', 'https://tfhub.dev/google/bit/m-r101x3/1']
    category_model_dict['shopping_bag'] = ['Mobilenet_v2', 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4']
    category_model_dict['earphones'] = ['R50x1_device', 'https://tfhub.dev/google/experts/bit/r50x1/in21k/device/1']
    category_model_dict['car_key'] = ['Efficientnet_lite0', 'https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2']
    category_model_dict['shoes'] = ['Mobilenet_v2', 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4']
    category_model_dict['document'] = ['R101x3', 'https://tfhub.dev/google/bit/m-r101x3/1']
    category_model_dict['watch'] = ['Mobilenet_v2', 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4']
     # 아래 카테고리들은 테스트 후 다시 모델 삽입 예정임 임시로 넣어 놈
    category_model_dict['cloth'] = ['R50x1_object', 'https://tfhub.dev/google/experts/bit/r50x1/in21k/clothing/1']
    category_model_dict['gloves'] = ['R50x1_object', 'https://tfhub.dev/google/experts/bit/r50x1/in21k/clothing/1']
    category_model_dict['muffler'] = ['R50x1_object', 'https://tfhub.dev/google/experts/bit/r50x1/in21k/clothing/1']
    category_model_dict['necklace'] = ['R50x1_object', 'https://tfhub.dev/google/experts/bit/r50x1/in21k/object/1']
    category_model_dict['ring'] = ['R50x1_object', 'https://tfhub.dev/google/experts/bit/r50x1/in21k/object/1']
    
    return category_model_dict

#%%

def select_model(category):
    # 카테고리를 입력받아 모델을 생성해서 모델명과 생성된 모델을 리턴함
    start_time = time.time()
    model_dict = make_dict()
    model_name = model_dict[category][0]
    model_url = model_dict[category][1]
    
    layer = hub.KerasLayer(model_url, input_shape=(224, 224)+(3,))
    model = tf.keras.Sequential([layer])
    model.build([None, 244, 244, 3])
    print(f'모델생성소요시간 : {time.time() - start_time:.3f}')

    return model_name, model

#%%

# 이미지 객체탐지
def crop(images):
    os.system(f"python yolov5/detect.py --source {images} --weights yolov5/runs/train/cate_29_epoch10/weights/best.pt --conf 0.4 --save-crop")

#%%

def extract(file):
    start_time = time.time()
    file = Image.open(file).convert('RGB').resize((224, 224))
    file = np.array(file)/255.0 # 정규화

    embedding = model.predict(file[np.newaxis, ...])
    feature_np = np.array(embedding)
    flattened_feature = feature_np.flatten()

    print(f'벡터 추출 소요시간 : {time.time() - start_time:.3f}')

    return flattened_feature

#%%

def get_dataframe(category):
    start_time = time.time()
    global output_path
    global model_name
    tmp_filename = np.load(output_path+f'{category}_filename({model_name}).npy', allow_pickle=True)
    tmp_output = np.load(output_path+f'{category}_output({model_name}).npy', allow_pickle=True)
    df = pd.DataFrame({'filename':tmp_filename, 'output':tmp_output})
    print(f'DF생성 소요시간 : {time.time() - start_time:.3f}')
    return df

#%%

def get_cos_sim(file, category, metric='cosine'):
    start_time=time.time()
    file2vec = extract(file) # 이미지 벡터화
    df = get_dataframe(category) # 데이터프레임 가져오기
    df = df.append({'filename':file, 'output':file2vec}, ignore_index=True)
    cos_sim_array = np.zeros((len(df)))
    for i in range(0, len(df)):
        cos_sim_array[i] = distance.cdist([file2vec] , [df.iloc[i, 1]], metric)[0] # 벡터화된 이미지 기준
    df['cos_sim']=cos_sim_array
    print(f'코사인유사도 계산 소요시간 : {time.time() - start_time:.3f}')
    return df# 런타임 비교용

#%%

# crop된 파일을 인풋 파일으로 넣어줘야함
def search_img(category, cropped_file, threshold=0.4):
    # global image_path
    global output_path
    cos_sim_df = get_cos_sim(cropped_file, category=category)
    df_top_sim = cos_sim_df[cos_sim_df.cos_sim <= threshold].sort_values(by='cos_sim')[:50]

    return df_top_sim.filename.values                               

#%%

# 메인
if __name__ == '__main__':
    # crop(input_file) # 인풋파일 잘라내기 + 카테고리 판단해주기
    category = input('category :') # 카테고리 이미지 전달
    input_file = input('image_path :') # 입력 이미지 전달
    start_time = time.time()
    # image_path = f"../crops/{category}/" # crops된 이미지 경로
    output_path = f"../vector_frame/{category}/" # npy파일 보관된 경로
    if not os.path.exists(output_path):
        print('디렉토리가 없으므로 생성합니다.')
        os.mkdir(output_path)
    
    model_name, model = select_model(category)
    
    # threshold의 디폴트 값은 0.4이고, search_img 함수의 마지막 인자로 넣어주면 바꿀 수 있음
    print(search_img(category, input_file)) # 인자로 input_file이 아닌 crops된 파일의 경로로 바꿔줘야함
    print(f'소요시간 : {time.time() - start_time:.3f}초') # 테스트용 코드 -> 추후 삭제





