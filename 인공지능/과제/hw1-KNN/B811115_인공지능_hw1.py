import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris # Iris ������ �ҷ�����
from math import sqrt

k_list = [3,5,9]
iris = load_iris()
X = iris.data # [data1, data2, data3, data4]
y = iris.target # 0,1,2
y_name = iris.target_names # setosa, versicolor, virginica
X_trains = []
y_trains = []
X_test = []
y_test = []



def test_train_divide():
  #�׽�Ʈ�� �������� �и�
  num=0
  for _ in range(10):
      for _ in range(14):
          X_trains.append(X[num])
          y_trains.append(y[num])
          num+=1
      X_test.append(X[num])
      y_test.append(y[num])
      num+=1



def modified_form():                                # target���� ���������� �߰���Ŵ
  for i in range(140):
    X_trains[i]=np.append(X_trains[i],y_trains[i])

  for i in range(10):
    X_test[i] = np.append(X_test[i],y_test[i])



def calc_distance(a,b):                         # X_test[a] �� X_trains[j] ������ ��Ŭ���� �Ÿ��� ���ϴ� �Լ�
  dist = 0.0
  for i in range(4):
    dist += (X_test[a][i]-X_trains[b][i])**2
  return sqrt(dist)



def find_neighbor_majorityvote(num):                          # k_list�� �ε����� ǥ���� num�� �Ű������� �Է¹���
  print(f"[k = {k_list[num]}]")                
  correct_cnt = 0                                             # ��Ȯ���� ��Ÿ���� ���� ����
  for i in range(10):                                         # 10���� Test ������ üũ
    dist_test_i = []                                          # �� Test��� ������ 140�� Trains������ ��Ŭ���� �Ÿ��� ������ �迭 ����
    for j in range(140):
      dist_test_i.append((calc_distance(i,j),X_trains[j]))    # X_test[i]�� X_trains[j]�� �Ÿ��� X_trains[j]�� �Բ� ����
    dist_test_i.sort(key=lambda x: x[0])                      # ������ ���� �迭�� Distance�� �������� ������������ ����
    neighbors = []                                            # k���� ����� Neighbors �� ������ �迭 ����
    for k in range(k_list[num]):                              
      neighbors.append(dist_test_i[k])                        # neighbors �迭�� ����� ������ k�� ����
    results = []
    for neighbor in neighbors:
      results.append(neighbor[-1][-1])                        # neighbors �迭�� X_trains�� ������ ��(target)�� result�� ������
    predict = max(set(results),key=results.count)             # results���� ���� ���� ���� predict�� ����        
    print(f"Test Data Index: {i} Computed class: {y_name[int(predict)]}, True class: {y_name[int(X_test[i][-1])]}") # ��� ���
    if predict == X_test[i][-1]:                              # ���� ���� ���� ���� ���ٸ� ���� Ƚ���� ������Ŵ    
      correct_cnt += 1

  accuracy = (correct_cnt/10)*100                             # ��Ȯ�� ��� �� ���
  print(f"Accuracy = {accuracy}%")



def find_neighbor_weightedmajorityvote(num):
  print(f"[k = {k_list[num]}]")                
  correct_cnt = 0                                             # ��Ȯ���� ��Ÿ���� ���� ����
  for i in range(10):                                         # 10���� Test ������ üũ
    dist_test_i = []                                          # �� Test��� ������ 140�� Trains������ ��Ŭ���� �Ÿ��� ������ �迭 ����
    for j in range(140):
      dist_test_i.append((calc_distance(i,j),X_trains[j]))    # X_test[i]�� X_trains[j]�� �Ÿ��� X_trains[j]�� �Բ� ����
    dist_test_i.sort(key=lambda x: x[0])                      # ������ ���� �迭�� Distance�� �������� ������������ ����
    neighbors = []                                            # k���� ����� Neighbors �� ������ �迭 ����
    for k in range(k_list[num]):                              
      neighbors.append(dist_test_i[k])                        # neighbors �迭�� ����� ������ k�� ����
    results = []
    cnt = k_list[num]
    for neighbor in neighbors:
      results = results + ([neighbor[-1][-1]]*cnt)            # neighbors �迭�� (X_trains�� ������ ��(target)*�Ÿ��� ���� ����ġ(�������� ū ��))�� result�� ������
      cnt -= 1                                                
    predict = max(set(results),key=results.count)             # results���� ���� ���� ���� predict�� ����        
    print(f"Test Data Index: {i} Computed class: {y_name[int(predict)]}, True class: {y_name[int(X_test[i][-1])]}") # ��� ���
    if predict == X_test[i][-1]:                              # ���� ���� ���� ���� ���ٸ� ���� Ƚ���� ������Ŵ    
      correct_cnt += 1

  accuracy = (correct_cnt/10)*100                             # ��Ȯ�� ��� �� ���
  print(f"Accuracy = {accuracy}%")




test_train_divide()
modified_form()
print("===============================Majority Vote===============================")
for i in range(3):
  find_neighbor_majorityvote(i)
print("==========================Weighted Majority Vote==========================")
for i in range(3):
  find_neighbor_weightedmajorityvote(i)