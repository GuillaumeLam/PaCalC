import pickle
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import pandas as pd
import glob
from scipy import interpolate
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import AveragePooling1D


class data_processing(object):
    def __init__(self):
        self.enviroment='GaitLab2Go'

    def find_files(self,path='.',ext='.xlsx'):
        serach_term=path +'/**/*'+ ext
        X=np.array([])
        i=0
        for file_name in glob.iglob(serach_term, recursive=True):
            Y=np.array([file_name])
            X=np.append(X,Y)
            i=+1
        return X

    def randomize(self,LL):
        A=np.array(list(range(LL)))
        np.random.shuffle(A)
        return A

    def string2int(self,data,colum_index=1):
        unique_identity=np.unique(data[:,colum_index])
        self.identity=unique_identity
        for i in range(unique_identity.shape[0]):
            data[:,colum_index]=np.where(data[:,colum_index]==unique_identity[i],int(i),data[:,colum_index])
        return data

    def train_test_split(self,data,parameters,subject_index=-2,subjects_wise_random=True,
    validation=False,output_parameters='all',
    prediction_index=-1,test_percentage=0.25,
    normalization=True):

        if output_parameters=='all':
            out_colum_idx = np.delete(np.arange(len(parameters)),[subject_index,prediction_index])
            #out_colum_idx= np.delete(out_colum_idx,prediction_index-1)
        else:
            out_colum_idx = np.where(np.char.find(list(parameters),output_parameters)==0)[0]

        n_x,n_y= data[:,out_colum_idx],data[:,prediction_index]
        if subjects_wise_random:

            subjects=np.unique(data[:,subject_index])
            test_subjects=int(round(subjects.shape[0]*test_percentage))
            test_inds=self.randomize(subjects.shape[0])[0:test_subjects]
            test_subjects=subjects[test_inds]
            train_inds=np.delete(np.arange(subjects.shape[0]),test_inds)
            train_subjects=subjects[train_inds]
            self.train_subjects=train_subjects
            self.test_subjects=test_subjects

            inds={'train':np.array([]),'test':np.array([]),'validation':np.array([])}
            for subject in range(train_subjects.shape[0]):
                inds['train']=np.append(inds['train'],np.where(data[:,subject_index]==train_subjects[subject])[0])

            for subject in range(test_subjects.shape[0]):
                inds['test']=np.append(inds['test'],np.where(data[:,subject_index]==test_subjects[subject])[0])

            if validation:
                pass

            x_train,y_train=n_x[inds['train'].astype('int64'),:],n_y[inds['train'].astype('int64')]
            x_test,y_test=n_x[inds['test'].astype('int64'),:],n_y[inds['test'].astype('int64')]
            train_idexs=self.randomize(y_train.shape[0])
            test_idexs=self.randomize(y_test.shape[0])
            x_train,y_train,x_test,y_test=x_train[train_idexs],y_train[train_idexs],x_test[test_idexs],y_test[test_idexs]
            if validation:
                pass

        else:
            test_num_idex=int(round(n_y.shape[0]*test_percentage))
            train_num_idex=int(n_y.shape[0]-test_num_idex)
            if validation:
                pass

            idexs=self.randomize(n_y.shape[0])
            test_idex,train_idex=idexs[0:test_num_idex],idexs[test_num_idex+1:train_num_idex]
            x_train,y_train=n_x[train_idex,:],n_y[train_idex]
            x_test,y_test=n_x[test_idex,:],n_y[test_idex]

        self.x_train,self.y_train,self.x_test,self.y_test=x_train,y_train,x_test,y_test
        self.output_parameters,self.subjects_wise_random=output_parameters,subjects_wise_random
        if normalization:
            self.normalize()


    def normalize(self):
        from sklearn import preprocessing
        self.scaler = preprocessing.StandardScaler().fit(self.x_train)
        self.x_train=self.scaler.transform(self.x_train)
        self.x_test=self.scaler.transform(self.x_test)
        #self.valid[0]=self.scaler.transform(self.valid[0])
        #self.x_mean = np.mean(self.x_train,axis=0)
        #self.x_std = np.std(self.x_train,axis=0)
        #self.x_train=(self.x_train-self.x_mean)/self.x_std
        #self.y_train=(self.x_test-self.x_mean)/self.x_std

    def ForwardFeed(self):
        import ForwardFeed as FF
        self.FF=FF


    def Normalized_gait(self,A):
        x=np.arange(A.shape[-1])
        A=np.where(np.isnan(A)==1,0,A)
        Y= interpolate.interp1d(x,A, kind='cubic')(np.linspace(x.min(), x.max(), 101))
        return Y

    def convert_data(self,data):

        allpara=list(data.columns)
        parameters=allpara[0:-2]
        norm_data={}
        for para in parameters:
            norm_data[f"{para}"]=[]
        for j in range(data.shape[0]):
            A=data.iloc[j,0:-2].to_numpy()
            normdata=np.zeros([A.shape[0],A[0].shape[0]],dtype='float64')
            for i in range (A.shape[0]):
                if A[i].shape[0]>A[i].shape[1]:
                    normdata[i,:]=A[i].reshape(A[i].shape[0])
                elif A[i].shape[0]<A[i].shape[1]:
                    normdata[i,:]=A[i].reshape(A[i].shape[1])
            new=self.Normalized_gait(normdata)

            for para in range(len(parameters)):
                norm_data[f"{parameters[para]}"].append(new[para])
        self.Ndata=norm_data
        self.Ndata["Subjects"]=data[allpara[-2]].values
        self.Ndata["Surface"]=data[allpara[-1]].values

    def surface_data(self):
        key=list(self.Ndata.keys())
        mdata=pd.read_pickle('mean_std_surface.plk')
        surface=mdata.to_dict()
        subject={}
        for i in range(len(self.identity)):
            indexs=np.where(self.Ndata["Surface"]==i)[0]
            subject[f"{self.identity[i]}"]=np.array(self.Ndata["Subjects"])[indexs]
            for j in range(len(key)-2):
                surface[f"{self.identity[i]}"][f"{key[j]}"]["data"]=np.array(self.Ndata[f"{key[j]}"])[indexs]

        self.Sdata=surface
        self.subjects=subject


    def feature_extraction(self):
        import features as fs
        mdata=pd.read_pickle('mean_std_surface.plk')
        para=list(self.Ndata.keys())

        self.features={}
        for i in range(len(para)-int(2)):
            outdata=pd.DataFrame(self.Ndata[para[i]])
            self.features[f'{para[i]}']=fs.feature(outdata,mdata,para[i])
        self.features[para[-2]]=self.Ndata[para[-2]]
        self.features[para[-1]]=self.Ndata[para[-1]]

    def all_features(self):
        para=list(self.features.keys())
        self.f={}
        for i in range(len(para)-int(2)):
            fetur=list(self.features[para[i]].keys())
            for j in range(len(fetur)):
                self.f[f'{para[i]}_{fetur[j]}']=self.features[para[i]][fetur[j]]
        self.f[para[-2]]=self.features[para[-2]]
        self.f[para[-1]]=self.features[para[-1]]
        self.computed_features=fetur
        self.sensors=para


    def ANN(self,hid_layers,model,output,input_shape,activation_hid='relu'):
        ann = tf.keras.models.Sequential()
        ann.add(tf.keras.Input(shape=input_shape))
        for l in hid_layers:
            ann.add(tf.keras.layers.Dense(units=l,activation=activation_hid))
        if model=='classification':
            ann.add(tf.keras.layers.Dense(units=output,activation='softmax'))
            ann.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
            return ann
        elif model=='regression':
            ann.add(tf.keras.layers.Dense(units=output))
            ann.compile(optimizer='adam',loss='mean_squared_error')
            return ann

    def CNN_test(self,input_shape,output_shape):
        model = tf.keras.models.Sequential()
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        model.add(Dropout(0.5))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(output_shape, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def CNN_Hu(self,input_shape,output_shape):
        model = tf.keras.models.Sequential()
        model.add(Conv1D(filters=100, kernel_size=3, activation='relu', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=100, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Conv1D(filters=100, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=50, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=3))
        model.add(Dropout(0.2))
        model.add(Conv1D(filters=50, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=50, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(AveragePooling1D())
        model.add(Flatten())
        #model.add(Dense(100, activation='relu'))
        model.add(Dense(output_shape, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def CNN1D(self,filters,kernel_size,input_shape,pool_size,model,hid_layers,activation_hid,output):
        cnn1d = tf.keras.models.Sequential()
        for i in range(len(filters)):
            if i==0:
                cnn1d.add(tf.keras.layers.Conv1D(filters[i],kernel_size[i],activation=activation_hid,input_shape=input_shape))
                cnn1d.add(tf.keras.layers.MaxPooling1D(pool_size[i]))

            else:
                cnn1d.add(tf.keras.layers.Conv1D(filters[i],kernel_size[i],activation=activation_hid))
                cnn1d.add(tf.keras.layers.MaxPooling1D(pool_size[i]))
        cnn1d.add(tf.keras.layers.Flatten())
        for l in hid_layers:
            cnn1d.add(tf.keras.layers.Dense(units=l,activation=activation_hid))
        if model=='classification':
            cnn1d.add(tf.keras.layers.Dense(units=output,activation='softmax'))
            cnn1d.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            return cnn1d
        elif model=='regression':
            cnn1d.add(tf.keras.layers.Dense(units=output))
            cnn1d.compile(optimizer='adam',loss='mean_squared_error')
            return cnn1d

    def one_hot(self,y):
        uniq=np.unique(y)
        y_hot=np.zeros([y.shape[0],uniq.shape[0]])
        for i in range(len(uniq)):
            index=np.where(y==uniq[i])[0]
            y_hot[index,i]=1
        self.surface_name=uniq
        return y_hot

    def one_hot_y(self,y):
        uniq=self.surface_name
        y_hot=np.zeros([y.shape[0],uniq.shape[0]])
        for i in range(len(uniq)):
            index=np.where(y==uniq[i])[0]
            y_hot[index,i]=1
        self.surface_name=uniq
        return y_hot


if __name__ == "__main__":
    # WRITE CODE HERE
    # Instantiate, train, and evaluate your classifiers in the space below
    pass
