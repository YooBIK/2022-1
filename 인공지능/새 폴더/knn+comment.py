import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

class KNN:
    def __init__(self, feature, targets, target_names, k):
        self.feature = feature # 훈련에 사용되는 여러 비트맵의 특징들이 담겨있는 2차원 넘파이 배열
        self.targets = targets # 훈련에 사용되기 위한 테스트의 실제 숫자값 배열 (답안지)
        self.target_names = target_names # 0~9 까지 숫자를 기억하고 있는 배열
        self.k = k # knn 알고리즘에서 정의하는 k 값
        self.distances = [] # 테스트 입력이 주어질경우 모든 데이터와 거리비교를 하여 저장할 배열
        self.k_cnt_idx = [] # 거리에 따라 정렬을 한후 거리가 작은순서의 distances 의 원소의 index값 를 기억하기 위한 배열

    def calculate_distance(self, p1, p2):
        # euclidian distance 를 구하기 위한 메소드
        # numpy의 braodcast 성질을 이용하여 계산
        return np.sqrt(((p1-p2)**2).sum(axis=1))
        #p1 ( 60000 길이의 배열) : 비교군 / p2 (feautre 길이의 배열) 실험군


    def k_nearest_neighbor(self, testcase):
        # calculate_distance 메소드를 이용하여 모든 훈련에 사용되는 분꽃들의 특징과 실험되는 분꽃 과의 계산한 거리값을
        # self.distances  라는 배열에다가 담는다
        self.distances = self.calculate_distance(self.feature, testcase)
        # boardcast 성질을 이용하여 쉽게 각거리의 값의 배열을 계산할수 있다.
        self.k_cnt_idx = (self.distances.argsort())[:self.k]
        # argsort()를 통해서 배열이 나오게 되는데
        # 모든 거리들중에서 작은원소 순서의 index를 k개만큼 기억하고 있는 배열이다.
        #numpy.array에서는 배열의 리스트의 index를 정하는 [ ] 안에 인덱스를 순서를 기억하는 배열을 인자로 넣어주는게 가능하다.


    def majority_vote(self):
        vote = np.zeros(len(self.target_names), dtype='int')
        #k개의 특징점중 최빈값을 알기 위해 숫자의종류만큼  배열을 선언해놓는다.
        for i in self.targets[self.k_cnt_idx]:
            vote[i] += 1
        #가장 가까운 k개의 숫자를 확인하면서 vote배열의 해당 숫자 index 에 1씩 증가시켜준다
        return self.target_names[np.where(vote == max(vote))[0][0]]
        #가장 최대 득표가 나온 target_name을 출력한다.

    def weighted_majority_vote(self):
        vote = np.zeros(len(self.target_names), dtype='float')
        # k개의 특징점중 최대 weight를 가진 분꽃을 알기위해 숫자의종류만큼 (0~9) 배열을 선언해놓는다.
        inverse_distance = np.array([float("inf") if (i == 0) else (1 / i) for i in self.distances[self.k_cnt_idx]],dtype=np.float64)
        # 거리에 대해서 가까울수록 가중치를 두기 위하여 새로운 inverse_distance라는 배열을 선언하고
        # 이 배열안에 모든 거리값들의 역수를 취해준다 (divide by zero exception 을 피하기 위해 삼항연산자 사용)
        # 이미 distance 배열이 순서가 정해져있으므로 역수로 변환만 시켜준다
        for target, w in zip(self.targets[self.k_cnt_idx], inverse_distance):
            vote[target] += w
        # 각각의 target 에 해당하는 index에 역수로 취해준 가중치를 더해준다
        return np.argmax(vote) #가장 가중치가 높은 target_names 를 반환한다
