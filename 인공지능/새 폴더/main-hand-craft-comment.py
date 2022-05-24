import sys, os

sys.path.append(os.pardir)

import numpy as np
from numpy.lib.stride_tricks import as_strided
from dataset.mnist import load_mnist
from PIL import Image
from knn import KNN


def pooling_2d_arr(arr):
    arr = np.pad(arr, 0, mode='constant')
    # 출력값의 차원이 작아지거나 변형이 되는것을 방지하기위해 패딩 적용
    output_size, target_size = (14, 14), (2, 2)
    # output_size 는 28 x 28 크기의 배열을 어느 사이즈로 출력할것인지 정의
    # 인접한 4칸은 2 x 2 칸이다 target_size 로 정의해주었다.
    arr_w = as_strided(arr, shape=output_size + target_size,
                       strides=(2 * arr.strides[0],
                                2 * arr.strides[1]) + arr.strides)
    # 인접한 4칸을 분리하여 새로운 배열을 만들기 위해 사용
    arr_w = arr_w.reshape(-1, *target_size)
    # 적용한 메소드 구조상 생긴 바깥의 배열을 제거하기 위해 사용
    return arr_w.max(axis=(1, 2)).reshape(output_size)
    #axis1 에 max 를 적용하고 다시 2에 대해 적용을 하면 4칸의 인접한 데이터에 대해서 최대값을 반환
    #반환시 원하는 output_size 로 reshape 하여 출력

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
# x_train 는 training data feature 이다
# t_train 는 training data label 이다
# x_test 는 test data feature 이다
# t_test 는 test data label 이다.
x_train = x_train.astype(np.float32).reshape(60000, 28, 28)
# x_train 의 numpy array 데이터타입을 float32 로 변경
# 또한 28 x 28 의 2차원 배열 구조로 바꿔준다
x_test = x_test.astype(np.float32).reshape(10000, 28, 28)
# x_test 의 numpy array 데이터타입을 float32 로 변경
# 또한 28 x 28 의 2차원 배열 구조로 바꿔준다
x_train = np.array([pooling_2d_arr(i) for i in x_train]).reshape((60000, 196))
x_test = np.array([pooling_2d_arr(i) for i in x_test]).reshape((10000, 196))
# 각각 x_train 과 x_test 둘다 정의해둔 pooling_2d_arr 함수를 이용하여
# 28 x 28 배열을 14 x 14 의 형태로 바꿔준다
# 바꿔줄때 인접한 4칸을 1칸으로 치환하는 방법을 이용하여 feature 를 스케일링 해주었다
# 또한 인접한 4칸중 가장 큰 값을 1칸으로  채택하는 방식을 이용하였다

label_name = [str(i) for i in range(10)]#label 의 이름 '0','1', ... ,'9' 을 저장하는 배열

test_counts = 100#테스트 갯수
test_idxs = np.random.randint(0, t_test.shape[0], test_counts)
#랜덤한 테스트 갯수만큼의 인덱스 배열
for k in [3, 5, 7, 10]: #k를 기준으로 순회
    wrong, tries = 0, 0
    print("k is : ", k)
    #틀렷을 경우를 기억하는 변수 wrong
    #실행 횟수를 기억하는 변수 tries
    for test_case, answer in zip(x_test[test_idxs], t_test[test_idxs]):
        # x_test[test_idxs] 전체 test data feature 중 랜덤한 인덱스에 해당되는 테스트 feature 들
        # t_test[test_idxs] 전체 test label 중 랜덤한 인덱스에 해당되는 테스트 인덱스
        if tries % 1000 == 0:
            print("test num: ", tries) #확인용 출력
        knn = KNN(x_train, t_train, label_name, k)  #knn 클래스 생성자를 이용하여 train set 전달
        knn.k_nearest_neighbor(test_case) # weighted majority vote 를 이용하여 예측값 추정

        predict = str(knn.weighted_majority_vote())
        ans = str(answer)

        print("predict : ",predict," ,ans : ",ans,end="")
        if predict != ans:
            print("  predict fail !!!",end="")
            wrong += 1
        print("")
        tries += 1

    print("k is : ", k)
    print("tries : ", tries)
    print("answer : ", tries - wrong, "/ wrong : ", wrong)
    print("accuracy : ", ((tries - wrong) / tries) * 100, "%")
