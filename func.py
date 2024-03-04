import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from nltk import RegexpTokenizer
import pandas as pd
import numpy  as np


#функции для анализа датасета

#Подробнее изучить датасеты
def analize_df(df):
    print("dtypes")#типы
    print(df.dtypes)
    print("shape")#размер
    print(df.shape)
    print("isnull")#пустые
    print(df.isnull().sum())
    print("value_counts")#наполнение колонок
    index_list=df.columns#list(df)
    for i in (index_list):
        print('\n\n')
        print(i,len(df[i].unique()),"!!!!!!!!!!!!!!!!!!!!!!/")
        print(df[i].value_counts())
        
        
        
#Подготовка датасета


#функция замены выбросов
def replace_outliers(index, min_value, max_value, df, print_info=True):#функция их замены, но автозамена выключена
    if min_value >= max_value:
        raise Exception(index, 'replace_outliers - ОШИБКА! НЕ КОРРЕКТНЫЕ МИНИМУМ И МАКСИМУМ')
    
    if print_info:
        print(index)
        #сколько % заменяем
        print("low",sum(df[index] < min_value)/df.shape[0]*100)
        print("high",sum(df[index] > max_value)/df.shape[0]*100)
        print("all",(sum(df[index] > max_value)+sum(df[index] < min_value))/df.shape[0]*100)
    df[index+'_outlier'] = 0
    df.loc[(df[index] < min_value) | (df[index] > max_value), index+'_outlier'] = 1
    
    df.loc[df[index] < min_value, index] = min_value
    df.loc[(df[index] > max_value), index] = max_value
    return df


#Поиск выбросов
def find_outliers(index_array, df, autozam=False, osn=''):
    for target in (index_array):
        print(target)
        column = df[target]
        sns.displot(column,kde=True, bins=10, rug=True)
        plt.xlabel('Numbers')
        plt.ylabel('Volume')
        plt.title(target)
        plt.grid(True)
        plt.show()
        plt.boxplot(column)
        plt.title("Ящиковая диаграмма "+target)
        plt.show()
        print(column.describe()["min"],column.describe()["25%"],column.describe()["mean"],column.describe()["75%"],column.describe()["max"])
        IQR=(column.describe()["75%"]-column.describe()["25%"])*1.5
        print('Межквартильный размах:', column.describe()["25%"]-IQR,column.describe()["75%"]+IQR)
        if autozam:
            df=replace_outliers(target,column.describe()["25%"]-IQR,column.describe()["75%"]+IQR, df)
            if target!=osn:
                target_max=df[target].max()
                if target_max==0:
                    print(target, 'УГРОЗА ДЕЛЕНИЯ НА НОЛЬ! ОПЕРАЦИЯ ПРИВЕДЕНИЯ К ШКАЛЕ ОТ 0 ДО 1 НЕ ВЫПОЛНЕНА')
                else:
                    df[target]=df[target]/target_max
        print('\n\n\n')
    return df
        



#Вытаскиваем значимые "теги" фильмов в дэми переменную
def popular_tok(df, index_array, delit_after_work=False, min_df=0.05, max_df=0.85, use_other=False):
    tokenizer = RegexpTokenizer(", ", gaps=True)
    vectorizer = CountVectorizer(min_df=min_df, max_df=max_df, tokenizer=tokenizer.tokenize)
    if use_other:
        vectorizer2 = CountVectorizer(min_df=0, max_df=min_df, tokenizer=tokenizer.tokenize) 
    for i in (index_array):

        try:
            slov = vectorizer.fit_transform(df[i])
            if use_other:
                slov2 = vectorizer2.fit_transform(df[i])
        except ValueError:
            print(i, 'After runing, no terms remain')
        else:
            if use_other:
                df=pd.concat(
    [df,pd.DataFrame(slov.A, columns=vectorizer.get_feature_names_out()),pd.DataFrame(np.array([sum(slov2.A.T)>0]).T, columns=['other_'+i])],
    axis=1)
            else:
                df=pd.concat(
            [df,pd.DataFrame(slov.A, columns=vectorizer.get_feature_names_out())
            ],
            axis=1)
            print(i, 'Success')
        if delit_after_work:
            print(i, 'drop')
            df=df.drop([i],axis=1)
        
    return df 