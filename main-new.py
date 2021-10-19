import random
import copy
import time
import numpy
from itertools import permutations
import sys
import math
import tkinter  # //GUI模块
import threading
from functools import reduce

# 参数
'''
ALPHA:信息启发因子，值越大，则蚂蚁选择之前走过的路径可能性就越大
      ，值越小，则蚁群搜索范围就会减少，容易陷入局部最优
BETA:Beta值越大，蚁群越就容易选择局部较短路径，这时算法收敛速度会
     加快，但是随机性不高，容易得到局部的相对最优
'''

(ALPHA, BETA, RHO, Q) = (1.5, 0.5, 0.5, 100.0)
# 城市数，蚁群
(city_num, ant_num,hour_num) = (8, 20, 14)

## 创建坐标矩阵
distance_x = [377,468,169,227,186,1000,1591,1668]
distance_y=[599,421,318,596,515,450,538,425]


## 创建距离矩阵
walking_distance=[0,5,9,8,9,8,66,62,5,0,7,9,10,6,65,65,9,7,0,6,7,1,68,68,8,9,6,0,6,5,65,62,9,10,7,6,0,6,65,61,8,6,1,5,6,0,8,5,66,65,68,65,65,8,0,6,62,65,68,62,61,5,6,0]
walking_distance=numpy.array(walking_distance).reshape(city_num,city_num)
waiting_time=[10,10,20,20,30,30,80,40,30,60,70,30,10,10,5,5,20,40,45,50,50,55,70,60,45,20,15,5,10,10,5,20,45,45,50,55,35,30,55,15,15,10,45,45,75,60,40,60,50,10,30,20,10,10,15,45,60,60,55,85,100,60,70,80,45,30,10,20,10,60,70,70,20,10,10,10,20,40,70,70,70,70,70,70,60,60,80,80,75,120,130,90,105,120,90,90,45,60,30,30,40,70,60,70,120,90,105,105,85,90,40,30]
waiting_time=numpy.array(waiting_time).reshape(city_num,hour_num)

def return_distance(i,j,t):
    tempt=t+walking_distance[i][j]
    tempt=int(tempt/60)
    temp_time=walking_distance[i][j]+waiting_time[j][tempt]
    return temp_time


def permu():
    best_distance = 1000
    possible_list = list(permutations(range(city_num), city_num))

    for i in possible_list:
        temp_distance = waiting_time[i[0]][0]
        for j in range(city_num-1):
            arriving_time=temp_distance+walking_distance[i[j]][i[j + 1]]
            temp_distance = temp_distance + return_distance(i[j], i[j + 1],  temp_distance)

        if temp_distance <= best_distance:
            best_distance = temp_distance

    return(best_distance)


## 创建信息素矩阵
pheromone_graph=numpy.ones((city_num,city_num,hour_num))

class Ant(object):

    # 初始化
    def __init__(self, ID):

        self.ID = ID  # ID
        self.__clean_data()  # 随机初始化出生点

    # 初始数据
    def __clean_data(self):

        self.path = []  # 当前蚂蚁的路径
        self.total_distance = 0.0  # 当前路径的总距离
        self.move_count = 0  # 移动次数
        self.current_city = -1  # 当前停留的城市
        self.open_table_city = [True for i in range(city_num)]  # 探索城市的状态

        city_index = random.randint(0, city_num - 1)  # 随机初始出生点
        self.current_city = city_index
        self.path.append(city_index)
        self.open_table_city[city_index] = False
        self.move_count = 1
        self.total_distance=self.total_distance+waiting_time[self.current_city,int(self.total_distance/60)]

    # 选择下一个城市
    def __choice_next_city(self):

        next_city = -1
        select_citys_prob = [0.0 for i in range(city_num)]  # 存储去下个城市的概率
        total_prob = 0.0

        # 获取去下一个城市的概率
        for i in range(city_num):
            if self.open_table_city[i]:
                try:
                    # 计算概率：与信息素浓度成正比，与距离成反比
                    select_citys_prob[i] = pow(pheromone_graph[self.current_city][i][int(self.total_distance/60)], ALPHA) * pow(
                        (1.0 / return_distance(self.current_city,i,int(self.total_distance/60))), BETA)
                    total_prob += select_citys_prob[i]
                except ZeroDivisionError as e:
                    print('Ant ID: {ID}, current city: {current}, target city: {target}'.format(ID=self.ID,
                                                                                                current=self.current_city,
                                                                                                target=i))
                    sys.exit(1)

        # 轮盘选择城市
        if total_prob > 0.0:
            # 产生一个随机概率,0.0-total_prob
            temp_prob = random.uniform(0.0, total_prob)
            for i in range(city_num):
                if self.open_table_city[i]:
                    # 轮次相减
                    temp_prob -= select_citys_prob[i]
                    if temp_prob < 0.0:
                        next_city = i
                        break

        # 未从概率产生，顺序选择一个未访问城市
        # if next_city == -1:
        #     for i in range(city_num):
        #         if self.open_table_city[i]:
        #             next_city = i
        #             break

        if (next_city == -1):
            next_city = random.randint(0, city_num - 1)
            while ((self.open_table_city[next_city]) == False):  # if==False,说明已经遍历过了
                next_city = random.randint(0, city_num - 1)

        # 返回下一个城市序号
        return next_city


    # 移动操作
    def __move(self, next_city):

        self.path.append(next_city)
        self.open_table_city[next_city] = False
        self.total_distance += return_distance(self.current_city,next_city,self.total_distance)
        self.current_city = next_city
        self.move_count += 1

    # 搜索路径
    def search_path(self):

        # 初始化数据
        self.__clean_data()

        # 搜素路径，遍历完所有城市为止
        while self.move_count < city_num:
            # 移动到下一个城市
            next_city = self.__choice_next_city()
            self.__move(next_city)

class TSP(object):

    def __init__(self, root, width=2000, height=900, n=city_num):

        # 创建画布
        self.root = root
        self.width = width
        self.height = height
        # 城市数目初始化为city_num
        self.n = n
        # tkinter.Canvas
        self.canvas = tkinter.Canvas(
            root,
            width=self.width,
            height=self.height,
            bg="#EBEBEB",  # 背景白色
            xscrollincrement=1,
            yscrollincrement=1
        )



        self.canvas.pack(expand=tkinter.YES, fill=tkinter.BOTH)

        img=tkinter.PhotoImage(file="Disney.gif")
        self.canvas.create_image(10,10, anchor=tkinter.NW, image=img)


        self.title("ACO(n: Random Start e: Start Searching s: Stop Searching q: Exit")
        self.__r = 8

        self.__lock = threading.RLock()  # 线程锁

        self.__bindEvents()
        self.new()


    # 按键响应程序
    def __bindEvents(self):

        self.root.bind("q", self.quite)  # 退出程序
        self.root.bind("n", self.new)  # 初始化
        self.root.bind("e", self.search_path)  # 开始搜索
        self.root.bind("s", self.stop)  # 停止搜索

    # 更改标题
    def title(self, s):

        self.root.title(s)

    # 初始化
    def new(self, evt=None):

        # 停止线程
        self.__lock.acquire()
        self.__running = False
        self.__lock.release()

        self.clear()  # 清除信息
        self.img = tkinter.PhotoImage(file="Disney.gif")
        self.canvas.create_image(10, 10, anchor=tkinter.NW, image=self.img)
        self.nodes = []  # 节点坐标
        self.nodes2 = []  # 节点对象

        # 初始化城市节点
        for i in range(len(distance_x)):
            # 在画布上随机初始坐标
            x = distance_x[i]
            y = distance_y[i]
            self.nodes.append((x, y))
            # 生成节点椭圆，半径为self.__r
            node = self.canvas.create_oval(x - self.__r,
                                           y - self.__r, x + self.__r, y + self.__r,
                                           fill="#ff0000",  # 填充红色
                                           outline="#000000",  # 轮廓白色
                                           tags="node",
                                           )
            self.nodes2.append(node)
            # 显示坐标
            self.canvas.create_text(x, y - 10,  # 使用create_text方法在坐标（302，77）处绘制文字
                                    text='(' + str(x) + ',' + str(y) + ')',  # 所绘制文字的内容
                                    fill='black'  # 所绘制文字的颜色为灰色
                                    )

        # 顺序连接城市
        # self.line(range(city_num))

        # 初始城市之间的距离和信息素
        for i in range(city_num):
            for j in range(city_num):
                for t in range(hour_num):
                    pheromone_graph[i][j][t] = 1.0

        self.ants = [Ant(ID) for ID in range(ant_num)]  # 初始蚁群
        self.best_ant = Ant(-1)  # 初始最优解
        self.best_ant.total_distance = 1 << 31  # 初始最大距离
        self.iter = 1  # 初始化迭代次数

    # 将节点按order顺序连线
    def line(self, order):
        # 删除原线
        self.canvas.delete("line")

        def line2(i1, i2):
            p1, p2 = self.nodes[i1], self.nodes[i2]
            self.canvas.create_line(p1, p2, fill="red", activewidth=12, tags="line", dash=(4,4))
            return i2

        # order[-1]为初始值
        reduce(line2, order, order[-1])

    # 清除画布
    def clear(self):
        for item in self.canvas.find_all():
            self.canvas.delete(item)

    # 退出程序
    def quite(self, evt):
        self.__lock.acquire()
        self.__running = False
        self.__lock.release()
        self.root.destroy()
        print(u"\n程序已退出...")
        sys.exit()

    # 停止搜索
    def stop(self, evt):
        self.__lock.acquire()
        self.__running = False
        self.__lock.release()

    # 开始搜索
    def search_path(self, evt=None):

        # 开启线程
        self.__lock.acquire()
        self.__running = True
        self.__lock.release()

        while self.__running:
            # 遍历每一只蚂蚁
            for ant in self.ants:
                # 搜索一条路径
                ant.search_path()
                # 与当前最优蚂蚁比较
                if ant.total_distance < self.best_ant.total_distance:
                    # 更新最优解
                    self.best_ant = copy.deepcopy(ant)
            # 更新信息素
            self.__update_pheromone_gragh()
            print(u"迭代次数：", self.iter, u"最佳路径总距离：", int(self.best_ant.total_distance))
            # 连线
            self.line(self.best_ant.path)
            # 设置标题
            self.title("ACO(n:Random Start e:Start Searching s:Stop Searching q:Exit) Number of Iterations: %d Best Time %d" % (self.iter,self.best_ant.total_distance))
            # 更新画布
            self.canvas.update()
            self.iter += 1
            if self.iter<=20:
                time.sleep(0.3)


    # 更新信息素
    def __update_pheromone_gragh(self):

        # 获取每只蚂蚁在其路径上留下的信息素
        temp_pheromone = numpy.ones((city_num,city_num,hour_num))
        for ant in self.ants:
            temp_current_time=0
            temp_current_hour=0
            for i in range(1, city_num):
                start, end = ant.path[i - 1], ant.path[i]
                temp_current_time=temp_current_time + return_distance(start,end,temp_current_hour)
                temp_current_hour=int(temp_current_time/60)
                # 在路径上的每两个相邻城市间留下信息素，与路径总距离反比
                temp_pheromone[start][end][temp_current_hour] += Q / ant.total_distance
                temp_pheromone[end][start][temp_current_hour] = temp_pheromone[start][end][temp_current_hour]

        # 更新所有城市之间的信息素，旧信息素衰减加上新迭代信息素
        for i in range(city_num):
            for j in range(city_num):
                for t in range(hour_num):
                    pheromone_graph[i][j][t] = pheromone_graph[i][j][t] * RHO + temp_pheromone[i][j][t]

    # 主循环
    def mainloop(self):
        self.root.mainloop()


if __name__ == '__main__':
#    TSP(tkinter.Tk()).mainloop()
    TSP1=TSP(tkinter.Tk())
    img = tkinter.PhotoImage(file="Disney.gif")
    TSP1.canvas.create_image(10,10, anchor=tkinter.NW, image=img)
    TSP1.new()
    TSP1.mainloop()


