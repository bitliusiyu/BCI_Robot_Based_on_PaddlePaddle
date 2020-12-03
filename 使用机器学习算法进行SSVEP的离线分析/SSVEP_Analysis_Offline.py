# coding:utf-8
# __author__ = "Liu SiYu"
import os
import glob
from mne.io import read_raw_brainvision, read_raw_fif
from mne import Epochs, create_info, events_from_annotations
from mne.time_frequency import psd_welch
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, Normalizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier, BaggingClassifier, \
    GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import pandas as pd


class SSVEP_Analysis_Offline:
    '''
    本类用于ssvep数据的分析，主要包含5个功能，分别为：数据加载，数据预处理，特征提取，分类器构建，离线分类与结果统计。
    '''

    def __init__(self):
        self.all_acc = []
        self.all_itr = []
        pass

    def load_data(self, filename='aaaa', data_format='eeg',
                  trial_list={'Stimulus/S  1': 1, 'Stimulus/S  2': 2, 'Stimulus/S  3': 3, 'Stimulus/S  4': 4},
                  tmin=0.0, tmax=8.0, fmin=5.5, fmax=35.0):
        '''

        :param filename: 脑电数据文件名
        :param data_format: 数据格式，支持.eeg与.fif两种脑电数据格式。
        :param trial_list: 需要分析的mark列表
        :param tmin: 分析时间段的起始时间
        :param tmax: 分析时间段的结束时间
        :param fmin: 分析频段的起始频率
        :param fmax: 分析频段的截止频率
        :return:
        '''
        if data_format == 'eeg':
            self.raw_data = read_raw_brainvision(filename, preload=True, verbose=False)
        elif data_format == 'fif':
            self.raw_data = read_raw_fif(filename, preload=True, verbose=False)
        else:
            print('当前没有添加读取该文件格式的函数，欢迎补充')

        self.sfreq = self.raw_data.info['sfreq']  # 采样率

        self.trial_list = trial_list   # mark列表
        # self.trial_list = {'Start': 1, 'End': 2}

        self.tmin, self.tmax = tmin, tmax # mark起止时间
        self.fmin, self.fmax = fmin, fmax  # 滤波频段

    def data_preprocess(self, window_size=2., window_step=0.1, data_augmentation=True):
        '''

        :param window_size: 滑动时间窗窗长
        :param window_step: 滑动时间窗步长
        :param data_augmentation: 是否需要进行脑电数据样本扩增
        :return:
        '''
        self.window_size = window_size
        events, _ = events_from_annotations(self.raw_data, event_id=self.trial_list, verbose=False)

        flag = self.tmin
        events_step = np.zeros_like(events)
        events_step[:, 0] += int(window_step*self.sfreq)
        event_augmentation = events
        events_temp = events
        while flag < self.tmax - window_size:
            events_temp = events_temp + events_step
            event_augmentation = np.concatenate((event_augmentation, events_temp), axis=0)
            flag += window_step

        if data_augmentation == True:
            all_event = event_augmentation
            all_event = all_event[np.argsort(all_event[:, 0])]
        else:
            all_event = events

        # 提取epoch
        self.epochs = Epochs(self.raw_data, events=all_event, event_id=self.trial_list, tmin=0, tmax=window_size,
                             proj=False, baseline=None, preload=True, verbose=False)
        self.epochs.drop_bad()

        # 滤波
        self.epochs_filter = self.epochs.filter(self.fmin, self.fmax, n_jobs=-1, fir_design='firwin', verbose=False)

        # 对标签进行编码
        le = LabelEncoder()
        self.all_label = le.fit_transform(self.epochs_filter.events[:, 2])

    def feature_extract(self, method=None, plot=True):
        '''

        :param method: 特征提取方法，目前仅支持PSD方法，后续会加入时域特征，频域特征，熵，以及组合特征。
                       由于目前结果已经很好了，就先这样吧。
        :param plot: 是否对特征进行可视化。
        :return:
        '''
        self.all_feature, self.frequencies = psd_welch(self.epochs_filter, fmin=self.fmin, fmax=self.fmax, verbose=False, n_per_seg=128)
        self.all_feature = self.all_feature.reshape(self.all_feature.shape[0], -1)  # 将各个通道的脑电特征拉平为一维
        # self.all_feature = np.mean(self.all_feature, axis=1)  # 对通道这个维度求平均，降维

        if plot == True:

            for label in np.unique(self.all_label):
                _, ax = plt.subplots(1, 1)
                ax.set_title(list(self.trial_list.keys())[label])
                # plt.show()
                self.epochs_filter.copy().drop(indices=(self.all_label != label)).\
                    plot_psd(dB=False, fmin=0.5, fmax=32., color='blue', ax=ax)

    def classifier_building(self, scaler_form='StandardScaler', train_with_GridSearchCV=False):
        '''

        :param scaler_form: 对输入数据X标准化进行标准化的类型，一共三种：StandardScaler，MinMaxScaler, Normalizer。
                            默认为StandardScaler。
        :param train_with_GridSearchCV: 是否使用网格搜索进行参数寻优。由于目前分类效果已经不错，
                                        所以该功能的优先级放在了最后，目前该功能还在完善中。
        :return:
        '''

        if scaler_form == 'StandardScaler':
            scaler = StandardScaler()
        elif scaler_form == 'MinMaxScaler':
            scaler = MinMaxScaler()
        else:
            scaler = Normalizer()

        # scaler = StandardScaler()
        # scaler.fit(X)
        # print(scaler.mean_, scaler.var_)
        # X = scaler.transform(X)

        # 创建文件夹，存储分类器model
        if not os.path.exists('./record_model'):
            os.mkdir('./record_model')

        # 准备数据集
        X = self.all_feature
        y = self.all_label

        # from sklearn.utils import shuffle
        # X, y = shuffle(X, y, random_state=0)  # 将数据打乱

        # X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  # 划分为训练集与测试集
        print(X.shape)

        X, X_test, y, y_test = X[0:int(len(y)*3/4), :], X[int(len(y)*3/4):, :], \
                               y[0:int(len(y)*3/4)], y[int(len(y)*3/4):]
        # 由于脑电信号在时域上差异较大，所以不建议打乱重排后再划分训练集。
        # （用前面的数据训练分类器来预测后面数据的类别，更显公允。）

        # 构建分类器
        lr_clf = LogisticRegression(multi_class='auto', solver="liblinear", random_state=42)
        lda_clf = LinearDiscriminantAnalysis()
        gn_clf = GaussianNB()
        svm_clf_1 = SVC(kernel='linear', gamma="auto", random_state=42)
        svm_clf_2 = SVC(kernel='poly', gamma="auto", random_state=42)
        svm_clf_3 = SVC(kernel='rbf', gamma="auto", random_state=42)
        svm_clf_4 = SVC(kernel='sigmoid', gamma="auto", random_state=42)
        knn_clf = KNeighborsClassifier()
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        gb_clf = GradientBoostingClassifier()
        ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=1000, algorithm="SAMME.R",
                                     learning_rate=0.5, random_state=42)
        bag_clf_rf = BaggingClassifier(n_jobs=-1, random_state=42)
        bag_clf_knn = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
        xgb_clf = XGBClassifier()
        mlp_clf = MLPClassifier(solver='sgd', activation='logistic', alpha=1e-4, hidden_layer_sizes=(30, 10),
                                random_state=42, max_iter=1000, verbose=False, learning_rate_init=.1)

        # voting_clf = VotingClassifier(estimators=[('lr_clf', lr_clf), ('lda_clf', lda_clf), ('gn_clf', gn_clf),
        #                                           ('svm_clf_1', svm_clf_1), ('svm_clf_2', svm_clf_2),
        #                                           ('svm_clf_3', svm_clf_3), ('svm_clf_4', svm_clf_4),
        #                                           ('knn_clf', knn_clf), ('rf_clf', rf_clf), ('gb_clf', gb_clf),
        #                                           ('ada_clf', ada_clf), ('bag_clf_rf', bag_clf_rf),
        #                                           ('bag_clf_knn', bag_clf_knn), ('xgb_clf', xgb_clf),
        #                                           ('mlp_clf', mlp_clf)],
        #                               voting='soft', n_jobs=-1)

        # 将以上的分类器进行投票，构造投票分类器。可以为这些分类器添加不同的权重，权重系数为weights=None
        voting_clf = VotingClassifier(estimators=[('lr_clf', lr_clf), ('lda_clf', lda_clf), ('gn_clf', gn_clf),
                                                  # ('svm_clf_1', svm_clf_1), ('svm_clf_2', svm_clf_2),
                                                  # ('svm_clf_3', svm_clf_3), ('svm_clf_4', svm_clf_4),
                                                  ('knn_clf', knn_clf), ('rf_clf', rf_clf), ('gb_clf', gb_clf),
                                                  ('ada_clf', ada_clf), ('bag_clf_rf', bag_clf_rf),
                                                  ('bag_clf_knn', bag_clf_knn), ('xgb_clf', xgb_clf),
                                                  ('mlp_clf', mlp_clf)],
                                      voting='soft', n_jobs=-1, weights=None)

        # 将以上所有分类器组合成一个列表表示
        listing_clf = [lr_clf, lda_clf, gn_clf, svm_clf_1, svm_clf_2, svm_clf_3, svm_clf_4, knn_clf,
                       rf_clf, gb_clf, ada_clf, bag_clf_rf, bag_clf_knn, xgb_clf, mlp_clf, voting_clf]
        self.listing_clf_name = ['lr_clf', 'lda_clf', 'gn_clf', 'svm_clf_1', 'svm_clf_2', 'svm_clf_3', 'svm_clf_4'
                            , 'knn_clf', 'rf_clf', 'gb_clf', 'ada_clf', 'bag_clf_rf', 'bag_clf_knn',
                            'xgb_clf', 'mlp_clf', 'voting_clf']

        # 进行训练（不进行参数寻优）
        if train_with_GridSearchCV == False:
            # 开始训练
            cv = StratifiedKFold(n_splits=5, shuffle=True)
            cv_scores = np.zeros((len(listing_clf)))  # 数组，分类结果准确率
            for i, classify in enumerate(listing_clf):
                models = Pipeline(memory=None,
                                  steps=[
                                          ('Scaler', scaler),  # 数据标准化
                                          (self.listing_clf_name[i], classify),  # 分类器
                                         ])

                models.fit(X=X, y=y)
                joblib.dump(classify, './record_model/' + str(i) + '-' + self.listing_clf_name[i] + '.pkl')
                print('第', i, '个分类器:', self.listing_clf_name[i])
                # new_svm = joblib.load('svm.pkl')

                y_pred = models.predict(X_test)

                # 计算acc 与 itr
                acc = accuracy_score(y_pred, y_test)
                self.all_acc.append(acc)
                itr = self.cal_itr(len(self.trial_list), acc, self.window_size)
                self.all_itr.append(itr)
                print('正确率：', acc, '   itr: ', itr)


                # # 使用K-fold交叉验证进行评估
                # cv_scores[i] = np.mean(cross_val_score(estimator=classify, X=X, y=y,
                #                                        scoring='accuracy', cv=cv, n_jobs=-1), axis=0)

            # print(cv_scores)

        # 进行训练（参数寻优，随机搜索）
        else:
            print('sss')

            param_dist = {
                'n_estimators': range(80, 200, 4),
                'max_depth': range(2, 15, 1),
                'learning_rate': np.linspace(0.01, 2, 20),
                'subsample': np.linspace(0.7, 0.9, 20),
                'colsample_bytree': np.linspace(0.5, 0.98, 10),
                'min_child_weight': range(1, 9, 1)
            }
            grid_search = RandomizedSearchCV(xgb_clf, param_dist, n_iter=300, cv=5,
                                             scoring='accuracy', n_jobs=-1)
            print('sss')
            grid_search.fit(X, y)

            print(grid_search.best_estimator_.feature_importances_)
            print(grid_search.best_params_)
            print(grid_search.best_estimator_)

            xgb_clf_final = grid_search.best_estimator_
            y_pred = xgb_clf_final.predict(X_test)
            print('正确率：', accuracy_score(y_pred, y_test))

            pass

    def classify_offline(self, model_file='./record_model/15-voting_clf.pkl'):
        '''

        :param model_file: 文件名
        :return:
        '''
        # 准备数据集
        X_test = self.all_feature
        y_test = self.all_label

        scaler = StandardScaler()
        scaler.fit(X_test)
        print(scaler.mean_, scaler.var_)
        X_test = scaler.transform(X_test)

        # 加载模型文件
        model_clf = joblib.load(model_file)

        # 进行预测
        y_pred = model_clf.predict(X_test)
        print('正确率：', accuracy_score(y_pred, y_test))

    def result_statistics(self):
        '''
        待补充
        :return:
        '''
        return self.all_acc, self.all_itr
        pass

    def cal_itr(self, q, p, t):
        '''
        BCI性能衡量指标丨信息传输速率 Information Transfer Rate
        itr的计算方法为理想ITR计算，即平均试次时间不包含模拟休息时长。

        :param q: 目标个数
        :param p: 识别正确率
        :param t: 平均试次时长，单位为s
        :return: itr: 信息传输速率
        '''

        if p == 1:
            itr = np.log2(q)*60/t
        else:
            itr = 60/t*(np.log2(q) + p*np.log2(p) + (1-p)*np.log2((1-p)/(q-1)))

        return itr


# demo: SSVEP_Analysis_Offline类的基本使用方法

if __name__ == '__main__':

    file_folder = [i for i in glob.glob(r'./data/*.vhdr')]
    window_size_list = [0.5, 1., 1.5, 2., 2.5, 3., 3.5]
    # window_size_list = [0.5, 1.]
    acc_list, itr_list = [], []

    for window_size in window_size_list:

        analysis_offline = SSVEP_Analysis_Offline()

        analysis_offline.load_data(file_folder[0], data_format='eeg',
                      trial_list={'Stimulus/S  1': 1, 'Stimulus/S  2': 2, 'Stimulus/S  3': 3, 'Stimulus/S  4': 4},
                      tmin=0.0, tmax=8.0, fmin=5.5, fmax=35.0)

        analysis_offline.data_preprocess(window_size=window_size, window_step=0.1, data_augmentation=True)

        analysis_offline.feature_extract(method=None, plot=False)

        analysis_offline.classifier_building(scaler_form='StandardScaler', train_with_GridSearchCV=False)

        acc, itr = analysis_offline.result_statistics()

        acc_list.append(acc)
        itr_list.append(itr)

    # 储存结果
    acc_pd = pd.DataFrame(np.array(acc_list), columns=analysis_offline.listing_clf_name, index=window_size_list)
    itr_pd = pd.DataFrame(np.array(itr_list), columns=analysis_offline.listing_clf_name, index=window_size_list)
    acc_pd.to_csv('./acc_pd.csv')
    itr_pd.to_csv('./itr_pd.csv')


    # if not os.path.exists('./record_model/15-voting_clf.pkl'):
    #   analysis_offline.classifier_building(scaler_form='StandardScaler', train_with_GridSearchCV=False)
    #
    # else:
    #     analysis_offline.classify_offline(model_file='./record_model/15-voting_clf.pkl')








































