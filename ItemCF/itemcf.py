import random
import math
from operator import itemgetter

class ItemBasedCF():
    def __init__(self, n_sim_movie, n_rec_movie):
        # 找到相似的20部电影，为目标用户推荐10部电影
        self.n_sim_movie = n_sim_movie
        self.n_rec_movie = n_rec_movie

        self.train_data = {}
        self.test_data = {}

        # 电影相似度矩阵
        self.movie_sim_matrix = {}
        self.movie_popular = {}
        self.movie_count = 0

        print(f'Similar movie number = {self.n_sim_movie}')
        print(f'Recommended movie number = {self.n_rec_movie}')

    def get_dataset(self, filename, pivot=0.75):
        traindata_len = 0
        testdata_len = 0
        for line in self.load_file(filename):
            user, movie, rating, timestamp = line.split(',')
            if random.random() < pivot:
                self.train_data.setdefault(user, {})
                self.train_data[user][movie] = rating
                traindata_len += 1
            else:
                self.test_data.setdefault(user, {})
                self.test_data[user][movie] = rating
                testdata_len += 1
        print('Split training dataset and test dataset success!')
        print(f'train dataset = {traindata_len}')
        print(f'test dataset = {testdata_len}')

    def load_file(self, filename):
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                yield line.strip('\r\n')
        print(f'Load {filename} success!')

    # 计算电影之间的相似度
    def calc_movie_sim(self):
        for user, movies in self.train_data.items():
            for movie in movies:
                if movie not in self.movie_popular:
                    self.movie_popular[movie] = 0
                self.movie_popular[movie] += 1

        self.movie_count = len(self.movie_popular)
        print(f'Total movie number = {self.movie_count}')

        for user, movies in self.train_data.items():
            for m1 in movies:
                for m2 in movies:
                    if m1 == m2:
                        continue
                    self.movie_sim_matrix.setdefault(m1, {})
                    self.movie_sim_matrix[m1].setdefault(m2, 0)
                    self.movie_sim_matrix[m1][m2] += 1
        print('Build co-rated movies matrix sucess!')

        # 计算电影之间的相似度
        print('Calculate movie similarity matrix...')
        for m1, ralated_movies in self.movie_sim_matrix.items():
            for m2, count in ralated_movies.items():
                # 注意0向量的处理，即某电影的用户数为0
                if self.movie_popular[m1] == 0 or self.movie_popular[m2] == 0:
                    self.movie_sim_matrix[m1][m2] = 0
                else:
                    self.movie_sim_matrix[m1][m2] = count / math.sqrt(self.movie_popular[m1] * self.movie_popular[m2])
        print('Calculate movie similarity matrix success!')

    # 针对目标用户U，找到K部相似的电影，并推荐其N部电影
    def recommend(self, user):
        K = self.n_sim_movie
        N = self.n_rec_movie
        rank = {}
        watched_movies = self.train_data[user]

        for movie, rating in watched_movies.items():
            for related_movie, w in sorted(self.movie_sim_matrix[movie].items(), key=itemgetter(1), reverse=True)[:K]:
                if related_movie in watched_movies:
                    continue
                rank.setdefault(related_movie, 0)
                rank[related_movie] += w * float(rating)
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[:N]

    # 产生推荐并通过准确率、召回率和覆盖率进行评估
    def evaluate(self):
        print('Evaluating start...')
        N = self.n_rec_movie
        hit = 0
        rec_count = 0
        test_count = 0
        # 覆盖率
        all_rec_movies = set()

        for i, user in enumerate(self.train_data):
            test_movies = self.test_data.get(user, {})
            rec_movies = self.recommend(user)
            for movie, w in rec_movies:
                if movie in test_movies:
                    hit += 1
                all_rec_movies.add(movie)
            rec_count += 1
            test_count += len(test_movies)

        precision = hit / (1.0 * rec_count)
        recall = hit / (1.0 * test_count)
        coverage = len(all_rec_movies) / (1.0 * self.movie_count)
        print(f'precision={precision}, recal={recall}, coverage={coverage}')

if __name__ == '__main__':
    rating_file = 'rating.csv'
    itemCF = ItemBasedCF(20, 10)
    itemCF.get_dataset(rating_file)
    itemCF.calc_movie_sim()
    itemCF.evaluate()