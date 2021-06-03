import json
from flask import Flask, render_template, request
from preprocessing_class import prepro_data, Apply_model
from preprocessing_class import viz_data
import pandas as pd
import plotly
from flask_bootstrap import Bootstrap
import numpy as np

global target_case
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
Bootstrap(app)

test_data_df = pd.read_pickle('./samplecase/temp_ill.pkl')
test_data_df = test_data_df.sample(n=10)
test_data_cus_list = test_data_df.CUS_REF_NO_7.values

df = pd.read_pickle('edadf.pkl')
group_df = pd.read_pickle('./group_df.pkl')

# LabelEnconding에 필요한 dictionary pkl 파일 폴더 및 모델이 저장된 경로
dict_dir = './le_dictionary'
model_dir = './trained_LGBM.pkl'


@app.route('/')
def mainpage():
    return render_template('index.html', caselist=test_data_cus_list)  # 메인 페이지에 신고번호 리스트 표시


@app.route('/result', methods=['POST', 'GET'])
def show_res_testcase():  # case_num이라는 dict 변수 반환 ( key:testcase / value:0~9 )
    case_num = 0
    if request.method == 'POST':
        case_dict = request.form
        case_num = case_dict['testcase']  # case_num 변수에 key값만 저장 ( key값 = test_case_df 행 번호로 사용 )

    test_case = test_data_df.iloc[int(case_num), :]  # 받아온 case_num에 해당하는 열 데이터 추출
    target_case = test_case
    target_origin = target_case
    drop_col = test_case.index
    # 사용자 함수 클래스에 포함된 전처리 함수 통해 입력 데이터 전처리 진행
    target_data = prepro_data.r6m_data(test_case, dict_dir)
    target_data = prepro_data.r6m_date_data(target_data)
    target_data = prepro_data.r6m_drop(target_data, drop_col)
    print(target_data)

    res_0, res_1 = Apply_model.print_proba(target_data, model_dir)  # res_0 : 위법물이 아닐 확률 / res_1 : 위법물일 확률
    fig = viz_data.proba_graph(np.round(res_0, 3), np.round(res_1, 3))  # res_0, res_1을 기반으로 plotly 그래프 제작
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)  # html에 표시하기 위해 json으로 변경

    # CON_NAME_8 기준으로 총 수입금액(VAL_FINANCIAL_VALUE_12), 총 수입건수(CUS_REF_NO_7) eda 항목 추가
    col_name = ['PERSON_NAME_54', 'TRD_NAME_2', 'IMP_TRADING_COUNTRY_11', 'cat_packages_num', 'date_quart']
    x_name = ['관세사', '화주', '국가', '품목수량', '분기', '총 수입건수', '총 수입금액']
    y_name = '적발비율(%)'
    graph_title = ['관세사별 적발비율', '화주별 적발비율', '국가별 적발비율',
                   '수량별 적발비율', '분기별 적발비율', '업자별 총 수입건수', '업자별 총 수입금액']

    # 결과 페이지에 전달해줄 차트 객체인 json과 적발비율을 사용자 함수 통해 생성
    nonill_target_name = target_origin['CON_NAME_8']

    target_origin = prepro_data.cat_val(target_origin)

    person_json, person_ratio = viz_data.show_viz_bar(df, col_name[0], x_name[0],
                                                      y_name, graph_title[0], target_origin[col_name[0]])

    trd_json, trd_ratio = viz_data.show_viz_bar(df, col_name[1], x_name[1],
                                                y_name, graph_title[1], target_origin[col_name[1]])

    country_json, country_ratio = viz_data.show_viz_bar(df, col_name[2], x_name[2],
                                                        y_name, graph_title[2], target_origin[col_name[2]])

    package_num_json, package_num_ratio = viz_data.show_viz_bar(df, col_name[3], x_name[3],
                                                                y_name, graph_title[3], target_origin[col_name[3]])

    date_json, date_ratio = viz_data.show_viz_bar(df, col_name[4], x_name[4],
                                                  y_name, graph_title[4], target_origin[col_name[4]])

    item_num = int(target_origin['CUS_TOTAL_NUMBER_OF_PACKAGES_6'])

    # 업자 관련 정보는 기존 사용자 함수와는 맞지 않으므로 json과 전달해줄 값을 별도로 구현
    count_group_df = group_df.sort_values(ascending=False, by='count')
    con_count_num = count_group_df.loc[target_origin['CON_NAME_8'], 'count']
    count_group_df = count_group_df[:10]
    count_group_fig = px.bar(x=count_group_df['count'], y=list(count_group_df.index),
                             text=list(count_group_df['count']), orientation='h')
    count_group_fig.update_layout(title_text='업자별 총 수입건수', title_font_size=25, title_font_color='RebeccaPurple',
                                  title_x=0.5,
                                  xaxis_title='총 수입건수', xaxis_color='red',
                                  yaxis_title='화주', yaxis_color='blue')
    count_group_JSON = json.dumps(count_group_fig, cls=plotly.utils.PlotlyJSONEncoder)

    sum_group_df = group_df.sort_values(ascending=False, by='sum')
    sum_group_sum = np.round(sum_group_df.loc[target_case['CON_NAME_8'], 'sum'], 2)
    sum_group_df = sum_group_df[:10]
    sum_group_fig = px.bar(x=sum_group_df['sum'], y=list(sum_group_df.index),
                           text=np.round(list(sum_group_df['sum']), 2), orientation='h')
    sum_group_fig.update_layout(title_text='업자별 총 수입금액', title_font_size=25, title_font_color='RebeccaPurple',
                                title_x=0.5,
                                xaxis_title='총 수입금액', xaxis_color='red',
                                yaxis_title='화주', yaxis_color='blue')
    sum_group_JSON = json.dumps(sum_group_fig, cls=plotly.utils.PlotlyJSONEncoder)

    # html 페이지 쪽에 전달할 그래프 객체, 컬럼명, 컬럼에 해당하는 값의 리스트
    json_list = [person_json, trd_json, country_json, package_num_json, date_json, count_group_JSON, sum_group_JSON]
    res_value_list = [person_ratio, trd_ratio, country_ratio, package_num_ratio, date_ratio,
                      con_count_num, sum_group_sum]
    target_value_list = [target_case[col_name[0]], target_case[col_name[1]], target_case[col_name[2]],
                         str(item_num), target_case[col_name[4]], nonill_target_name, nonill_target_name]

    return render_template('eda_chart.html', resfig=graphJSON, target_name_list=x_name,
                           target_value_list=target_value_list, ill_ratio_list=res_value_list, fig=json_list)


if __name__ == '__main__':
    app.run(debug=True)
