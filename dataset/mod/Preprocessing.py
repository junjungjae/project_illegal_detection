########## Class ##########


### 전처리에 사용할 class

class Preprocess :
    
    def __init__(self, df) :
        self.df = df
    
    
    ##### Class Variables #####
    
    
    # 사용이 불가능하다고 판단, 데이터프레임에서 아예 드랍시킬 컬럼들(추후 변동 가능)
    
    drop_col_list = ['IMP_TYPE_OF_DECLARATION_1', 'IMP_TYPE_OF_DECLARATION_2',
                     'TRD_TIN_2', 'CUS_SHIPMENT_SPEC_4',
                     'PER_TIN_9', 'PER_NAME_9', 'PER_COUNTRY_9', 'PER_ADDR_9',
                     'DNT_TIN_14', 'DNT_NAME_14', 'DNT_COUNTRY_14', 'DNT_ADDR_14',
                     'IMP_CONTAINER_FLAG_19', 'LOD_LOCATION_NAME_27', 'IDG_QUOTA',
                     'ZQNTY', 'ZAUXUOM', 'PERSON_POSITION_54', 'LOC_LOCATION_NAME_30',
                     'COV_CUST_VALUE_METHOD', 'IDG_STAT_VALUE_VAL_METH_46']
    
    
    # 사용은 가능하지만 중간에 null값이 있는 명목형 컬럼들. fillna로 처리(추후 변동 가능)
    
    fillna_col_list = ['TRD_COUNTRY_2', 'CUS_REF_NO_7', 'CON_COUNTRY_8',
                       'IMP_TRADING_COUNTRY_11', 'VAL_CURRENCY_12',
                       'IMP_CNT_OF_DISPATCH_EXP_CD_15', 'DEL_DELIVERY_TERM_CODE_20',
                       'TOT_CURRENCY_22',
                       'DEL_PLACE_OF_DELIVERY_20']
    
    
    # 사용은 가능하지만 중간에 null값이 있는 연속형 컬럼들. fillna로 처리(추후 변동 가능)

    num_col_list = ['CUS_TOTAL_NUMBER_OF_PACKAGES_6', 'VAL_FINANCIAL_VALUE_12',
                    'TOT_FINANCIAL_VALUE_22', 'IMP_EXCHANGE_RATE_23',
                    'GDS_GROSS_MASS_35', 'IDG_NET_MASS_38', 'FIN_FINANCIAL_VALUE_42',
                    'STC_FINANCIAL_VALUE_46']
               

    # Label 에 영향을 미치지 않아 단순히 null이 존재하는 인덱스만 드랍(추후 변동 가능)

    index_drop_list = ['PRF_PREFERENCE_CODE_1', 'COR_FINANCIAL_VALUE', 'GEND_REFERENCE_54',
                       'CAL_TYPE_OF_TAX_47', 'CAL_ADDITIONAL_RATE_OF_TAX_47']
    
    
    ##### Class Methods #####
    
    
    # null값 제거 사용자 함수
    def null_solution(self):

        df = self.df.copy()
        for column_name in df.columns :
            
            # 사용하기 힘든 칼럼들 drop
            if column_name in self.drop_col_list :
                df.drop(column_name, axis=1, inplace=True)
            
            # fillna_col_list에 포함되있다면 null값을 'null'로 대체
            elif column_name in self.fillna_col_list :
                df[column_name].fillna('null', inplace=True)
            
            # num_col_list에 포함되 있다면 컬럼의 평균값으로 null값을 대체
            elif column_name in self.num_col_list :
                mean = df[column_name].mean()
                df[column_name].fillna(mean, inplace=True)
            
            # index_drop_list에 속한 칼럼들은 해당 index를 삭제
            elif column_name in self.index_drop_list :
                drop_idx = df[df[column_name].isnull()].index
                df.drop(drop_idx, axis=0, inplace=True)
            
        return df
       

 ### 인코딩에 사용할 class
class Encoder :
    
    def __init__(self, df) :
        self.df = df

    # Label Encoder
    def label(self, classes=None) :
        
        # 초기화 때 입력한 DataFrame의 사본을 사용
        df = self.df.copy()
        
        # LabelEncoder 객체 생성
        from sklearn.preprocessing import LabelEncoder
        import numpy as np
        import pandas as pd
        import pickle

        le = LabelEncoder()
                
        # for문을 사용, 각 칼럼별로 인코딩 적용
        for column in df.columns:
            # 칼럼의 데이터 타입이 str인 것만 인코딩 실시
            if type(df[column][0]) == str :
                # Label Encoding 실시
                column_encoded = le.fit_transform(df[column])
                df[column] = column_encoded
                # 인코딩한 칼럼 이름을 변수명으로 하는 dict 변수 생성
                # 칼럼_이름 = {원래 데이터 : 인코딩 된 번호}
                encoding_val = np.sort(df[column].unique())
                decoding_val = le.classes_
                val_dict = dict(zip(decoding_val, encoding_val))
                # dict를 pkl로 저장
                with open('./dataset/Label_Encoding_dict/{}.pickle'.format(column), 'wb') as f :
                  pickle.dump(val_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

            else : pass
               
        return df
    
    # One-hot Encoder
    def one_hot(self) :

      import numpy as np
      import pandas as pd

      # self.df의 복사본 생성
      df = self.df.copy()
      # 인코딩 대상 칼럼 넣는 DataFrame
      # 이렇게 하는 이유 : 안그러면 칼럼 이름에 값만 뜸
      # 이렇게 해주면 칼럼 이름으로 원래 칼럼 이름 + 값이 뜸
      oh_columns = pd.DataFrame()

      # for loop로 대상 칼럼 oh_columns에 넣고
      # df에서는 삭제
      for column in df.columns :
        if str(df[column].dtype) == 'object' :
          oh_columns[column] = df[column]
          df.drop(column, axis=1, inplace=True)
        else : pass
      # get_dummies 함수 사용하여 one_hot 진행
      oh_df = pd.get_dummies(oh_columns)
      for value in oh_df.columns :
        df[value] = oh_df[value]
      
      return df

    


########## Methods ##########


# VIF 계산
def calculate_vif(df):
    
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import pandas as pd
    
    if 'LABEL' in df.columns :
        X = df.drop('LABEL', axis=1)
    else : X = df.copy()
    
    vif = pd.DataFrame()
    vif['VARIABLES'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in  range(X.shape[1])]
    
    sorted_vif = vif.sort_values(by='VIF', ascending=False)
    
    return sorted_vif
    

# Under Sampling Check Tool

def usct(path) :
    
    import pandas as pd
    
    # 데이터 로드    
    df = pd.read_pickle(path)
    
    # unique값 넣을 DataFrame 생성
    max_check = pd.DataFrame()
    # COLUMN 칼럼에는 df의 칼럼 이름 리스트 입력
    max_check['COLUMN'] = df.columns
    
    # unique의 len값 넣을 리스트 생성
    lenlist = []
    
    # 0번째 칼럼부터 차례대로 for문 돌려서 unique의 len값을 lenlist에 더해줌
    for i in range(df.shape[1]) :
        value = df.iloc[:, i].unique()
        numb = len(value)
        lenlist.append(numb)
    
    max_check['UNIQUE'] = lenlist
    # unique의 len 값이 가장 높은 칼럼 순으로 정렬
    max_check.sort_values(by='UNIQUE', ascending=False, inplace=True)
    # 소수 클래스 수 출력
    print('Numb of Minor Class : {}'.format(len(df[df['LABEL'] == 1])))
    return max_check



########## Variables ##########





                
                
                
                
                
                