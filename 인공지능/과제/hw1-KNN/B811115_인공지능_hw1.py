import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris # Iris 데이터 불러오기
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
  #테스트와 나머지를 분리
  num=0
  for _ in range(10):
      for _ in range(14):
          X_trains.append(X[num])
          y_trains.append(y[num])
          num+=1
      X_test.append(X[num])
      y_test.append(y[num])
      num+=1



def modified_form():                                # target값을 데이터폼에 추가시킴
  for i in range(140):
    X_trains[i]=np.append(X_trains[i],y_trains[i])

  for i in range(10):
    X_test[i] = np.append(X_test[i],y_test[i])



def calc_distance(a,b):                         # X_test[a] 와 X_trains[j] 사이의 유클리드 거리를 구하는 함수
  dist = 0.0
  for i in range(4):
    dist += (X_test[a][i]-X_trains[b][i])**2
  return sqrt(dist)



def find_neighbor_majorityvote(num):                          # k_list의 인덱스를 표현할 num을 매개변수로 입력받음
  print(f"[k = {k_list[num]}]")                
  correct_cnt = 0                                             # 정확도를 나타내기 위한 변수
  for i in range(10):                                         # 10개의 Test 각각을 체크
    dist_test_i = []                                          # 각 Test들과 나머지 140개 Trains사이의 유클리드 거리를 저장할 배열 선언
    for j in range(140):
      dist_test_i.append((calc_distance(i,j),X_trains[j]))    # X_test[i]와 X_trains[j]의 거리와 X_trains[j]를 함께 저장
    dist_test_i.sort(key=lambda x: x[0])                      # 저장이 끝난 배열을 Distance를 기준으로 오름차순으로 정렬
    neighbors = []                                            # k개의 가까운 Neighbors 를 저장할 배열 선언
    for k in range(k_list[num]):                              
      neighbors.append(dist_test_i[k])                        # neighbors 배열에 가까운 순서로 k개 저장
    results = []
    for neighbor in neighbors:
      results.append(neighbor[-1][-1])                        # neighbors 배열의 X_trains의 마지막 값(target)을 result에 저장함
    predict = max(set(results),key=results.count)             # results에서 가장 많은 값을 predict에 저장        
    print(f"Test Data Index: {i} Computed class: {y_name[int(predict)]}, True class: {y_name[int(X_test[i][-1])]}") # 결과 출력
    if predict == X_test[i][-1]:                              # 예측 값이 실제 값과 같다면 정답 횟수를 증가시킴    
      correct_cnt += 1

  accuracy = (correct_cnt/10)*100                             # 정확도 계산 및 출력
  print(f"Accuracy = {accuracy}%")



def find_neighbor_weightedmajorityvote(num):
  print(f"[k = {k_list[num]}]")                
  correct_cnt = 0                                             # 정확도를 나타내기 위한 변수
  for i in range(10):                                         # 10개의 Test 각각을 체크
    dist_test_i = []                                          # 각 Test들과 나머지 140개 Trains사이의 유클리드 거리를 저장할 배열 선언
    for j in range(140):
      dist_test_i.append((calc_distance(i,j),X_trains[j]))    # X_test[i]와 X_trains[j]의 거리와 X_trains[j]를 함께 저장
    dist_test_i.sort(key=lambda x: x[0])                      # 저장이 끝난 배열을 Distance를 기준으로 오름차순으로 정렬
    neighbors = []                                            # k개의 가까운 Neighbors 를 저장할 배열 선언
    for k in range(k_list[num]):                              
      neighbors.append(dist_test_i[k])                        # neighbors 배열에 가까운 순서로 k개 저장
    results = []
    cnt = k_list[num]
    for neighbor in neighbors:
      results = results + ([neighbor[-1][-1]]*cnt)            # neighbors 배열의 (X_trains의 마지막 값(target)*거리에 대한 가중치(가까울수록 큰 값))을 result에 저장함
      cnt -= 1                                                
    predict = max(set(results),key=results.count)             # results에서 가장 많은 값을 predict에 저장        
    print(f"Test Data Index: {i} Computed class: {y_name[int(predict)]}, True class: {y_name[int(X_test[i][-1])]}") # 결과 출력
    if predict == X_test[i][-1]:                              # 예측 값이 실제 값과 같다면 정답 횟수를 증가시킴    
      correct_cnt += 1

  accuracy = (correct_cnt/10)*100                             # 정확도 계산 및 출력
  print(f"Accuracy = {accuracy}%")




test_train_divide()
modified_form()
print("===============================Majority Vote===============================")
for i in range(3):
  find_neighbor_majorityvote(i)
print("==========================Weighted Majority Vote==========================")
for i in range(3):
  find_neighbor_weightedmajorityvote(i)