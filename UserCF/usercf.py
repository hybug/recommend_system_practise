import random
import math
from operator import itemgetter

class UserBasedCF():
    # 找到与目标用户兴趣相似的20个用户，为其推荐10部电影
    def __init__(self, n_sim_user, n_rec_movie):
        self.n_sim_user = n_sim_user
        self.n_rec_movie = n_rec_movie

        self.train_data = {}
        self.test_data = {}

        # 用户相似度矩阵
        self.user_sim_matrix = {}
        self.movie_cnt = 0

        print(f'Similar user number = {self.n_sim_user}')
        print(f'Recommended movie number = {self.n_rec_movie}')

    def get_data(self, file_path, pivot=0.75):
        train_data_length = 0
        test_data_length = 0
        for line in self.load_file(file_path):
            user, movie, rating, timestamp = line.split(',')
            if random.random() < pivot:
                self.train_data.setdefault(user, {})
                self.train_data[user][movie] = rating
                train_data_length += 1
            else:
                self.test_data.setdefault(user, {})
                self.test_data[user][movie] = rating
                test_data_length += 1
        print(f'length of train_data = {train_data_length}')
        print(f'length of test_data = {test_data_length}')

    def load_file(self, file_path):
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                yield line.strip('\r\n')

    def calc_user_sim(self):
        # 构建“电影-用户”倒排索引
        # key = movieID, value = list of userIDs who have seen this movie
        print('Building movie-user table...')
        movie_user = {}
        for user, movies in self.train_data.items():
            for movie in movies:
                if movie not in movie_user:
                    movie_user[movie] = set()
                movie_user[movie].add(user)
        print('Finish building movie-user table')

        self.movie_cnt = len(movie_user)
        print(f'Total movie number = {self.movie_cnt}')

        print('Build user co-rated movies matrix...')
        for movie, users in movie_user.items():
            for u in users:
                for v in users:
                    if u == v:
                        continue
                    self.user_sim_matrix.setdefault(u, {})
                    self.user_sim_matrix[u].setdefault(v, 0)
                    self.user_sim_matrix[u][v] += 1
        print('Build user co-rated movie matrix success')

        # 计算相似性
        print('Calculating user similarity matrix...')
        for u, related_users in self.user_sim_matrix.items():
            for v, count in related_users.items():
                self.user_sim_matrix[u][v] = count / math.sqrt(len(self.train_data[u]) * len(self.train_data[v]))
        print('Calculate user similarity matrix success!')

    # 针对目标用户U，找到其最相似的K个用户，产生N个推荐
    def recommend(self, user):
        K = self.n_sim_user
        N = self.n_rec_movie
        rank = {}
        watched_movies = self.train_data[user]

        # v=similar user, wuv=similar factor
        for v, wuv in sorted(self.user_sim_matrix[user].items(), key=itemgetter(1), reverse=True)[0:K]:
            for movie in self.train_data[v]:
                if movie in watched_movies:
                    continue
                rank.setdefault(movie, 0)
                rank[movie] += wuv
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N]

    # 产生推荐并通过准确率、召回率和覆盖率进行评估
    def evaluate(self):
        print('Evaluation start ...')
        N = self.n_rec_movie
        # 准确率和召回率
        hit = 0
        rec_count = 0
        test_count = 0
        # 覆盖率
        all_rec_movies = set()

        for i, user, in enumerate(self.train_data):
            test_movies = self.test_data.get(user, {})
            rec_movies = self.recommend(user)
            for movie, w in rec_movies:
                if movie in test_movies:
                    hit += 1
                all_rec_movies.add(movie)
            rec_count += N
            test_count += len(test_movies)

        precision = hit / (1.0 * rec_movies)
        recall = hit / (1.0 * test_count)
        coverage = len(all_rec_movies) / (1.0 * self.movie_cnt)
        print(f'precison={precision}, recall={recall}, coverage={coverage}')

if __name__ == '__main__':
    rating_file = './data/ratings.csv'
    userCF = UserBasedCF(n_sim_user=20, n_rec_movie=10)
    userCF.get_data(rating_file)
    userCF.calc_user_sim()
    userCF.evaluate()