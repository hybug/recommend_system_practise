import numpy as np
import random
import itertools
import scipy.misc
import matplotlib.pyplot as plt


class gameOb():
    def __init__(self, coordinates, size, intensity, channel, reward, name):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.intensity = intensity
        self.channel = channel
        self.reward = reward
        self.name = name


class gameEnv():
    def __init__(self):
        self.sizeX = 5
        self.sizeY = 5
        self.actions = 4
        self.objects = []
        self.hero = None
        self.goal = None
        self.score = 0
        self.reset()
        # plt.imshow(a, interpolation="nearest")

    def get_next_state_with_random_choice(self, game):
        random_choice = random.choice([choice for choice in AVAILABLE_CHOICES])
        game.moveChar(random_choice)


    def reset(self):
        self.objects = []
        lst = []
        # hero = gameOb(self.newPosition(), 1, 1, 2, None, 'hero')
        # self.objects.append(hero)
        # bug = gameOb(self.newPosition(), 1, 1, 1, 1, 'goal')
        # self.objects.append(bug)
        # for j in range(7):
        #     hole = gameOb(self.newPosition(), 1, 1, 0, -1, 'fire')
        #     self.objects.append(hole)

        hero = gameOb([0,2], 1, 1, 2, None, 'hero')
        self.objects.append(hero)
        bug = gameOb([3,3], 1, 1, 1, 1, 'goal')
        self.objects.append(bug)
        hole_lst = [[0,1], [2,1], [4,1], [4,2], [0, 3], [2,3]]
        for l in hole_lst:
            hole = gameOb(l, 1, 1, 0, -1, 'fire')
            self.objects.append(hole)

        lst.append([hero.x, hero.y])
        lst.append([bug.x, bug.y])

        if len(self.objects) >= 2:
            self.hero = self.objects[0]
            self.goal = self.objects[1]
        # self.objects.append(bug4)
        # state = self.renderEnv()
        # self.state = state

        # return state

    def moveChar(self, direction):
        # 0 - up, 1 - down, 2 - left, 3 - right
        # hero = self.objects[0]
        for obj in self.objects:
            if obj.name == 'hero':
                hero = obj
            if obj.name == 'goal':
                goal = obj
        if direction == 0:
            hero.y -= 1
        if direction == 1:
            hero.y += 1
        if direction == 2:
            hero.x -= 1
        if direction == 3:
            hero.x += 1
        self.objects[0] = hero

    def newPosition(self):
        iterables = [range(self.sizeX), range(self.sizeY)]
        points = []
        for t in itertools.product(*iterables):
            points.append(t)
        currentPositions = []
        for objectA in self.objects:
            if (objectA.x, objectA.y) not in currentPositions:
                currentPositions.append((objectA.x, objectA.y))
        for pos in currentPositions:
            points.remove(pos)
        location = np.random.choice(range(len(points)), replace=False)
        return points[location]

    def game_is_over(self):
        others = []
        for obj in self.objects:
            if obj.name == 'hero':
                hero = obj
            else:
                others.append(obj)
        for other in others:
            if hero.x == other.x and hero.y == other.y:
                return True
        if hero.x < 0 or hero.x >= self.sizeX:
            return True
        if hero.y < 0 or hero.y >= self.sizeY:
            return True
        return False

    def cal_score(self):
        others = []
        for obj in self.objects:
            if obj.name == 'hero':
                hero = obj
            else:
                others.append(obj)
        for other in others:
            if hero.x == other.x and hero.y == other.y:
                if other.name == 'goal':
                    self.score += 10
                else:
                    assert other.name == 'fire'
                    self.score += -1

        if hero.x < 0 or hero.x >= self.sizeX:
            self.score += -1
        if hero.y < 0 or hero.y >= self.sizeY:
            self.score += -1

    def renderEnv(self):
        matrix = np.zeros([self.sizeY, self.sizeX])
        for obj in self.objects:
            if obj.name == 'hero':
                matrix[obj.x][obj.y] = 2
            elif obj.name == 'goal':
                matrix[obj.x][obj.y] = 3
            else:
                matrix[obj.x][obj.y] = 1
        print(matrix.transpose())

        # a = np.zeros([self.sizeY, self.sizeX, 3])
        # a = np.ones([self.sizeY+2,self.sizeX+2,3])
        # a[1:-1, :, :] = 0
        # a[:, 0, 0] = 1
        # a[:, -1, 0] = 1
        # a[0, :, 1:3] = 0
        # a[-1, :, 1:3] = 0
        # hero = None
        # for item in self.objects:
        #     a[item.y+1:item.y+item.size+1,item.x+1:item.x+item.size+1,item.channel] = item.intensity
        #     # a[item.y:item.y + item.size, item.x:item.x + item.size, item.channel] = item.intensity
        #     # if item.name == 'hero':
        #     #     hero = item
        # # if self.partial == True:
        # #     a = a[hero.y:hero.y+3,hero.x:hero.x+3,:]
        #
        # # 0-fire 1-goal 2-hero
        # b = scipy.misc.imresize(a[:, :, 0], [84, 84, 1], interp='nearest')
        # c = scipy.misc.imresize(a[:, :, 1], [84, 84, 1], interp='nearest')
        # d = scipy.misc.imresize(a[:, :, 2], [84, 84, 1], interp='nearest')
        # a = np.stack([b, c, d], axis=2)
        # plt.imshow(a, interpolation="nearest")

    def step(self, action):
        penalty, dis_reward, done = self.moveChar(action)
        state = self.renderEnv()
        if done == True:
            return state, penalty, done, 0
        else:
            reward,done = self.checkGoal()

        if reward == None:
            print(done)
            print(reward)
            print(penalty)
            return state,(reward+penalty+dis_reward),done, 1 if reward == 1 else 0
        else:
            return state,(reward+penalty+dis_reward),done, 1 if reward == 1 else 0