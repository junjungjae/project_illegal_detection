class prepro_data:
    """
    입력받은 원본 데이터를 학습 모델에 최적화 시키기 위한 전처리 함수들이 포함된 클래스
    """

    def r6m_data(input_data, dict_dir):
        """
        :param input_data: 전처리가 전혀 진행되지 않은 원본 데이터 행
        :param dict_dir: 파생변수를 만드는 데 필요한 dict들이 모여있는 경로
        :return: 기존 입력 데이터에서 파생변수 20개가 추가된 행 데이터
        """
        import os
        import joblib
        import warnings
        warnings.filterwarnings('ignore')

        dict_folder = os.listdir(dict_dir)
        # 파생변수 제작에 사용되는 column들
        r6m_col = ['CON_NAME_8', 'COM_COMBINED_NOMENCLATURE_33', 'IDG_COUNTRY_OF_ORIGIN_34', 'REP_TIN_54']
        # 파생변수 이름에 사용될 단어 리스트
        r6m_name = ['CON', 'CODE', 'COUNTRY', 'REP']
        r6m_bname = ['IMP_CNT', 'IMP_AMT', 'NUMB_OF_INSPECTION', 'NUMB_OF_DETECTED', 'RATIO_OF_DETECTED']

        name_cnt = 0
        for col_name in input_data.index:
            if col_name in r6m_col:  # 입력된 데이터의 컬럼을 돌며 r6m_col에 포함된 컬럼이 있을때만 동작
                for j, bname in enumerate(r6m_bname):
                    if (r6m_name[name_cnt] is 'CODE') and (bname is 'RATIO_OF_DETECTED'):  # 해당 파생변수는 쓰이지 않으므로 넘어감
                        continue
                    col_str = 'R6M_' + str(r6m_name[name_cnt]) + '_' + str(r6m_bname[j])  # 파생변수 column 이름 생성
                    files = [file for file in dict_folder if col_str in file]  # 생성된 파생변수 이름에 맞는 dict파일 찾기
                    dict_save_dir = dict_dir + '/{}'.format(str(files[0]))
                    col_dict = joblib.load(dict_save_dir)
                    # 해당 dict 파일에 맞는 key값이 있다면 value값을, 아니라면 0을 return
                    try:
                        input_data[col_str] = col_dict[input_data['CON_NAME_8']]
                    except:
                        input_data[col_str] = 0
                name_cnt += 1
        return input_data

    def r6m_date_data(input_data):
        """
        :param input_data: 파생변수가 1차적으로 추가된 행 데이터
        :return: 해당 입력 데이터에서 월, 시간 데이터가 추가된 행 데이터
        """
        import pandas as pd
        import joblib

        column_list = ['ACCEPTANCE_DATE', 'IMP_DATE_OF_DECLARATION_54', 'CUS_REF_NO_7']
        edf = input_data[column_list]

        # A_HOUR 칼럼 생성
        edf['A_TIME'] = edf['ACCEPTANCE_DATE'][9:17]
        edf['D_A_TIME'] = pd.to_datetime(edf['A_TIME'])
        edf['NEW_A_TIME'] = edf['D_A_TIME'].strftime('%H')
        edf['NEW_A_TIME'] = int(edf['NEW_A_TIME'])

        # C_MONTH 칼럼 생성
        edf['NEW_C_DATE'] = '20' + edf['CUS_REF_NO_7'][11:13] + edf['CUS_REF_NO_7'][9:11] + edf['CUS_REF_NO_7'][7:9]
        edf['MONTH'] = edf['NEW_C_DATE'][4:6]
        edf.MONTH = int(edf.MONTH)

        input_data['C_MONTH'] = edf['MONTH']
        input_data['A_HOUR'] = edf['NEW_A_TIME']

        # dict에 포함되 있다면 해당 key값의 value를, 아니면 0을 return
        import joblib
        con_ill_dict = joblib.load('./le_dictionary/CON_ILLEGAL_RATIO_dict.pkl')
        try:
            input_data['CON_ILLEGAL_RATIO'] = con_ill_dict[input_data['CON_NAME_8']]
        except:
            input_data['CON_ILLEGAL_RATIO'] = 0



        return input_data


    def cat_val(input_data):
        # 범주화를 통해 명확한 시각화 가능
        def do_cat(data):
            if data >= 1000 and data < 10000:
                return 'k'
            elif data >= 10000 and data < 100000:
                return '10k'
            elif data >= 100000 and data < 1000000:
                return '100k'
            elif data >= 1000000 and data < 10000000:
                return 'M'
            else:
                return '10M'
        #CUS_REF_NO_7 컬럼의 데이터를 기반으로 날짜 데이터 처리
        date_data = input_data['CUS_REF_NO_7']
        temp_year = date_data[11:13]
        temp_month = int(date_data[9:11])
        quart_list = ['1분기', '2분기', '3분기', '4분기']
        res = temp_year + '년 ' + quart_list[int((temp_month - 1) / 3)]

        input_data['cat_packages_num'] = do_cat(input_data['CUS_TOTAL_NUMBER_OF_PACKAGES_6'])
        input_data['date_quart'] = res
        return input_data



    # 사용하지 않는 컬럼 drop
    def r6m_drop(input_data, drop_col):
        res_data = input_data.drop(drop_col)
        return res_data


class Apply_model:
    """
    전처리된 데이터를 바탕으로 학습된 모델에 넣어 예측값을 도출하는 클래스
    """

    def print_proba(input_data, model_dir):
        """
        :param input_data: LabelEncoding이 완료된 data. 입력 형식은 Series.
        :param model_dir: 기존에 학습시킨 LightGBM 모델이 있는 pkl의 경로. 입력 형식은 문자열
        :return: 입력받은 데이터를 학습된 모델에 넣었을 때의 predict_proba 각각의 결과값
        """
        import joblib
        lgbm_model = joblib.load(model_dir)

        save_target = input_data.values
        temp_res = save_target.reshape(1, -1)
        temp_res = lgbm_model.predict_proba(temp_res)
        return temp_res[0][0], temp_res[0][1]


class viz_data:
    # 사용자 함수 작성 => plotly 형태로 출력하는 그래프
    def show_viz_bar(df, col_name, x_name, y_name, title, target_ratio, show_num=10):
        """
        :param df: 시각화를 진행할 데이터가 포함된 데이터프레임
        :param col_name: 입력받은 데이터프레임 중 시각화를 진행할 column의 이름
        :param x_name: barplot의 x축 제목
        :param y_name: barplot의 y축 제목
        :param title: barplot의 제목
        :param show_num: 시각화해야되는 x축 데이터의 개수를 지정할 수 있음. 표시되는 x축 데이터는 내림차순 기준으로 입력 수치만큼만 생성됨.
        :return: 위 파라미터를 기반으로 하는 plotly barplot
        """
        import plotly.express as px
        import pandas as pd
        import numpy as np
        import json
        import plotly

        idf = df.loc[df.LABEL == 1]  # 적발된 데이터만 추출한 데이터프레임
        a_index = idf[col_name].value_counts().index  # 적발 df에서 입력한 column에 대한 값들의 index
        a = idf[col_name].value_counts()  # 적발 df에서의 입력한 column에 대한 value들 전체
        b = df[col_name].value_counts()[a_index]  # 전체 df에서 비교를 위해 a_index를 기반으로 한 value들 전체

        temp_ratio = []  # 적발 비율을 담기 위한 list
        temp_case = []  # 적발 건수를 담기 위한 list
        for anum, bnum in zip(a, b):  # zip으로 적발건수, 전체 거래건수 출력
            temp_ratio.append(round((anum / bnum) * 100, 4))  # (적발건수/전체 거래건수) * 100 => 해당 항목에 대한 적발 비율. 계산 후 list에 추가
            temp_case.append(anum)  # 단순 적발 건수를 list에 추가
        temp_df = pd.DataFrame([temp_ratio, temp_case], index=['ratio', 'casenum'],
                               columns=a_index).T  # 컬럼을 ratio(적발 비율), casenum(적발 건수)로 가지는 데이터프레임 생성

        res_df = temp_df.sort_values('ratio', ascending=False)[:show_num]
        fig = px.bar(y=res_df.index, x=list(res_df.ratio),
                     text=list(res_df.casenum), orientation='h')  # x축을 시리즈의 인덱스로, y축을 시리즈의 데이터(적발 비율)로 설정하고 각 barplot에 건수 데이터 입력
        fig.update_layout(title_text=title, title_font_size=25, title_font_color='RebeccaPurple', title_x=0.5)
        fig.update_layout(xaxis_title=y_name, xaxis_color='red')
        fig.update_layout(yaxis_title=x_name, yaxis_color='blue')  # 플롯 제목, x축 이름, y축 이름을 각각 입력받은 값으로 설정
        fig.update_xaxes(range=[0, 100])
        fig.update_layout(template="plotly")
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        try:
            res_ratio = np.round(res_df.loc[target_ratio, 'ratio'], 3)
        except:
            res_ratio = 0
        return graphJSON, res_ratio

    def proba_graph(proba_a, proba_b):  # (proba_a, proba_b)였는데, 위법물일 확률만 나타내면 되서, proba_b만 사용함
        import plotly.graph_objects as go
        import numpy as np

        # 위법 여부를 게이지 차트로 시각화
        fig = go.Figure(go.Indicator(
            mode='gauge+number',
            value=np.round(proba_b * 100, 1),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': '위법물일 확률(%)'},
            gauge={'axis': {'range': [0, 100]},  # 범위 1~100까지
                   'steps': [
                       {'range': [50, 75], 'color': "lightgray"},
                       {'range': [75, 100], 'color': "gray"}],
                   'bar': {'color': "darkred"}}))

        fig.update_layout(
            autosize=False,
            width=500,
            height=500,
        )
        return fig