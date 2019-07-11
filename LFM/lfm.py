import random
import operator

class LFM():

    def __int__(self):
        self.all_item_set = set()
        self.train_data = {}
        self.test_data = {}

    def load_file(self, file_path):
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                yield line.strip('\r\n')

    def get_data(self, file_path, pivot=0.8):
        for line in self.load_file(file_path):
            user, movie, rating, timestamp = line.split(',')
            if random.random() < pivot:
                self.train_data.setdefault(user, {})
                self.train_data[user][movie] = rating
            else:
                self.test_data.setdefault(user, {})
                self.test_data[user][movie] = rating
        print(f'length of train dataset: {len(self.train_data)}')
        print(f'length of test dataset: {len(self.test_data)}')

    def init_all_item_set(self, user_items):
        self.all_item_set.clear()
        for user, items in user_items.items():
            for i, r in items.items():
                self.all_item_set.add(i)

    def init_item_pool(self, items):
        interacted_items = set(items.keys())
        items_pool = list(self.all_item_set - interacted_items)
        return items_pool

    def random_select_negative_sample(self, items):
        ret = dict()
        for i in items.keys():
            ret[i] = 1
        n = 0
        for i in range(0, len(items) * 3):
            items_pool = self.init_item_pool(items)
            item = items_pool[random.randint(0, len(items_pool - 1))]
            if item in ret:
                continue
            ret[item] = 0
            n += 1
            if n > len(items):
                break
        return ret

    def predict(self, user, item, P, Q):
        rate = 0
        for f, puf in P[user].items():
            qif = Q[item][f]
            rate += puf * qif
        return rate

    def init_model(self, user_items, F):
        P = dict()
        Q = dict()
        for user, item in user_items.items():
            P[user] = dict()
            for f in range(0, F):
                P[user][f] = random.random()
            for i, r in item.items():
                if i not in Q:
                    Q[i] = dict()
                    for f in range(0, F):
                        Q[i][f] = random.random()
        return P, Q

    def latent_factor_model(self, user_items, F, T, alpha, lamb):
        self.init_all_item_set(user_items)
        [P, Q] = self.init_model(user_items, F)
        for step in range(T):
            for user, items in user_items.items():
                samples = self.random_select_negative_sample(items)
                for item, rui in samples.items():
                    eui = rui - self.predict(user, item, P, Q)
                    for f in range(F):
                        P[user][f] += alpha * (eui * Q[item][f] - lamb * P[user][f])
                        Q[item][f] += alpha * (eui * P[user][f] - lamb * Q[item][f])
            alpha *= 0.9
        return P, Q

    def recommend(self, user, train_data, P, Q):
        rank = dict()
        interacted_items = train_data[user]
        for i in Q:
            if i in interacted_items.keys():
                continue
            rank.setdefault(i, 0)
            for f, qif in Q[i].items():
                puf = P[user][f]
                rank[i] += puf * qif
        return rank

    def recommendation(self, users, train, P, Q):
        result = dict()
        for user in users:
            rank = self.recommend(user, train, P, Q)
            R = sorted(rank.items(), key=operator.itemgetter(1), reverse=True)
            result[user] = R
        return result