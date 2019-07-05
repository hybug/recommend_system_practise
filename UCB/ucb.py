import numpy as np

# T轮试验
T = 1000
# N个赌博机
N = 10

# 每个赌博机真是的吐钱的概率
true_rewards = np.random.uniform(low=0, high=1, size=N)
# 每个赌博机吐钱的预估概率，初始值为0
estimated_rewards = np.zeros(N)
# 每个赌博机当前已经探索的次数，初始值为0
chosen_cnt = np.zeros(N)
total_rewards = 0

# calculate delta
def cal_delta(T: int, item: int) -> float:
    if chosen_cnt[item] == 0:
        return 1.0
    else:
        return np.sqrt(2 * np.log(T) / chosen_cnt[item])

def udb(t: int, N: int) -> tuple:
    upper_bound_probs = [estimated_rewards[item] + cal_delta(t, item) for item in range(N)]
    idx = np.argmax(upper_bound_probs)
    reward = np.random.binomial(n=1, p=true_rewards[idx])
    return idx, reward


for t in range(1, T):
    idx, reward = udb(t, N)
    total_rewards += reward

    # 更新赌博机的吐钱概率
    estimated_rewards[idx] = ((t - 1) * estimated_rewards[idx] + reward) / t
    if t % 100 == 0:
        print(f'第{t}轮预估的赌博机概率{estimated_rewards}')
    chosen_cnt[idx] += 1

print(f'\n赌博机的真实概率{true_rewards}')
