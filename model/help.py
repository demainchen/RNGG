import pickle
import numpy as np
# def process_distance_map(distance_map_file, cutoff = 14):
#
#     length = distance_map_file.shape[0]
#     distance_map = np.zeros((length, length))
#
#
#     for i in range(0,length):
#         for j in range(0,length):
#             if distance_map_file[i][j]<=cutoff:
#                 distance_map[i][j]=1
#             else:
#                 distance_map[i][j]=0
#
#     distance_map = distance_map + distance_map.T + np.eye(length)
#     return distance_map

# data=pickle.load(open('/home/chenwenqi/PPI-SITE/STRUCTURE+SEQUENCE/数据集/Test_60.pkl','rb'))
# for key in data.keys():
# #     data[key]['s2']=process_distance_map(data[key]['structure_emb'])
# # pickle.dump(data,open('/root/autodl-tmp/数据集/Test_60.pkl','wb'))
#     print(data[key].keys())

#
# import pickle
# key='>2v9tA'
# train_=pickle.load(open('/home/chenwenqi/PPI-SITE/STRUCTURE+SEQUENCE/数据集/train_335.pkl','rb'))
#
# test_=pickle.load(open('/home/chenwenqi/PPI-SITE/STRUCTURE+SEQUENCE/数据集/Test_60.pkl','rb'))
# data=test_[key]
# train_[key]=data
# pickle.dump(train_,open('/home/chenwenqi/PPI-SITE/STRUCTURE+SEQUENCE/数据集/train_335.pkl','wb'))







# import pickle
# key='>3vz9D'

# train_=pickle.load(open('/home/chenwenqi/PPI-SITE/STRUCTURE+SEQUENCE/数据集/Test_60.pkl','rb'))

# if key in train_.keys():
#     print("true")





import pickle

data=pickle.load(open('/home/chenwenqi/PPI-SITE/STRUCTURE+SEQUENCE/数据集/Test_60.pkl','rb'))['>3vz9D']
pickle.dump(data,open('/home/chenwenqi/PPI-SITE/STRUCTURE+SEQUENCE/数据集/3vz9D.pkl','wb'))



















