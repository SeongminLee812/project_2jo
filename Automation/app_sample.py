from flask import render_template
from flask import Flask, request
from py_src.model_category import ModelCategory
import sys
app = Flask(__name__)

@app.route('/')     # '/'은 http://xxx.xxx.xxx.xxx:5000/ 과 같이 포트 뒤에 붙는 url 주소를 의미합니다. 여기에 작성하지는 않았지만 POST요청인지, GET 요청인지 명시하는게 좋습니다.
def extract():
    '''
    이 부분에서 중요한 것은 app.run으로 flask 앱을 띄우기 전에 생성한 model_category 객체를 접근할 수 있다는 점입니다.
    따라서 사용자 사진을 입력받았을 때 아래 코드처럼 미리 만들어놓은 모델에 접근할 수 있습니다.
    '''
    return_str = f'''
wallet주소: {str(id(model_category.model_dict['wallet']['model']))}
phone주소: {str(id(model_category.model_dict['phone']['model']))}
'''
    return return_str


if __name__ == '__main__':
    '''
    코드 유지관리를 위해서는 가급적 클래스 구조로 작성하기를 추천합니다. 
    '''
    model_category = ModelCategory()
    model_category.set_model()
    app.run()