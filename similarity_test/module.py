def create_model(url, IMAGE_SHAPE):
    model_url = url
    layer = hub.KerasLayer(model_url, input_shape=IMAGE_SHAPE+(3,))
    model = tf.keras.Sequential([layer])
    model.summary()
    model.build([None, 244, 244, 3])
    return model


def extract(file, model):
    file = Image.open(file).resize((224, 224))
    file = np.array(file)/255.0 # 정규화

    embedding = model.predict(file[np.newaxis, ...])
    feature_np = np.array(embedding)
    flattened_feature = feature_np.flatten()

    return flattened_feature


def make_dataframe(category, model_name, image_path, output_path, model):
    file_list = os.listdir(image_path)
    file_list_img = [file for file in file_list if file.endswith(".png") or file.endswith(".jpeg") or file.endswith(".jpg")]
    tmp_df = pd.DataFrame()
    for i, img in enumerate(tqdm(file_list_img)):
        output = extract(image_path+'/'+img, model)
        tmp_df = tmp_df.append({'filename':img, 'output':output}, ignore_index=True)

    np.save(output_path+f'{category}_filename({model_name}).npy', tmp_df['filename'])
    np.save(output_path+f'{category}_output({model_name}).npy', tmp_df['output']) 

    
def get_dataframe(category, model_name, image_path, output_path):
    tmp_output = np.load(output_path+f'{category}_output({model_name}).npy', allow_pickle=True)
    tmp_filename = np.load(output_path+f'{category}_filename({model_name}).npy', allow_pickle=True)

    df = pd.DataFrame({'filename':tmp_filename, 'output':tmp_output})
    return df


def get_cos_sim(category, model_name, image_path, output_path, metric='cosine'):
    before_time = time.time()
    file2vec = extract(file) # 이미지 벡터화
    df = get_dataframe(category, model_name, image_path, output_path) # 데이터프레임 가져오기
    df = df.append({'filename':file, 'output':file2vec}, ignore_index=True)
    
    cos_sim_array = np.zeros((len(df)))
    for i in range(0, len(df)):
        cos_sim_array[i] = distance.cdist([file2vec] , [df.iloc[i, 1]], metric)[0] # 벡터화된 이미지 기준
    df['cos_sim']=cos_sim_array
    after_time = time.time()
    runtime = after_time-before_time
    return df, runtime # 런타임 비교용


def show_sim(filename, category, model_name, image_path, output_path):
    cos_sim_df, runtime = get_cos_sim(filename)
    df_top_sim = cos_sim_df.sort_values(by='cos_sim')[:15]

    # 그래프 그리는 부분은 서비스 시 생략 가능
    f, ax = plt.subplots(3, 5, figsize=(40, 20))

    for i in range(len(df_top_sim)):
        if i == 0: 
            tmp_img = Image.open(df_top_sim.iloc[i, 0]).convert('RGB')
            title = f'Original \n{df_top_sim.iloc[i, 0]}'
        else : 
            tmp_img = Image.open(image_path+'/'+df_top_sim.iloc[i, 0]).convert('RGB')
            title = f'similarity no.{i} \n{df_top_sim.iloc[i, 0]}'

        sim = f'cos : {df_top_sim.iloc[i, 2]:.3f}' 
        ax[i//5][i%5].imshow(tmp_img, aspect='auto')
        ax[i//5][i%5].set_title(title, pad=20, size=25) # 5열짜리 표를 만드는 것이므로 단순히 5로 나눈 나머지와 몫을 사용한 것임
        ax[i//5][i%5].annotate(sim, (0,10), fontsize=18, color='red')
    
    print(f'소요시간 : {runtime:.3f}')
    plt.show()
    

def show_sim_threshold(filename, category, model_name, image_path, output_path, threshold):
    cos_sim_df, runtime = get_cos_sim(filename, category)
    df_top_sim = cos_sim_df[cos_sim_df.cos_sim < threshold].sort_values(by='cos_sim')[:30]
    # 그래프 그리는 부분은 서비스 시 생략 가능
    if len(df_top_sim) <= 10:
        f, ax = plt.subplots(2, 5, figsize=(40, 20))
    elif len(df_top_sim) <=15:
        f, ax = plt.subplots(3, 5, figsize=(40, 30))
    elif len(df_top_sim) <=20:
        f, ax = plt.subplots(4, 5, figsize=(40, 40))
    elif len(df_top_sim) <=25:
        f, ax = plt.subplots(5, 5, figsize=(40, 45))
    else:
        f, ax = plt.subplots(6, 5, figsize=(40, 50))

    for i in range(len(df_top_sim)):
        if i == 0: 
            tmp_img = Image.open(df_top_sim.iloc[i, 0]).convert('RGB')
            title = f'Original \n{df_top_sim.iloc[i, 0]}'
        else : 
            tmp_img = Image.open(image_path+'/'+df_top_sim.iloc[i, 0]).convert('RGB')
            title = f'similarity no.{i} \n{df_top_sim.iloc[i, 0]}'

        sim = f'cos : {df_top_sim.iloc[i, 2]:.3f}' 
        ax[i//5][i%5].imshow(tmp_img, aspect='auto')
        ax[i//5][i%5].set_title(title, pad=20, size=25) # 5열짜리 표를 만드는 것이므로 단순히 5로 나눈 나머지와 몫을 사용한 것임
        ax[i//5][i%5].annotate(sim, (0,10), fontsize=18, color='red')
    
    print(f'소요시간 : {runtime:.3f}')
    plt.show()


# def get_runtime():
#     runtime_list = list()
#     for i in range(100):
#         df, runtime = get_cos_sim('cap_marant.jpg', 'cap')
#         runtime_list.append(runtime)
#     runtime_mean = np.mean(runtime_list)
#     print(f'100회 평균 소요시간 : {runtime_mean:.3f}')




