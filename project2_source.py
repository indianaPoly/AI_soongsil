import numpy as np
import csv
import random
import matplotlib.pyplot as plt
import pandas as pd
import math
from copy import deepcopy
from collections import deque

class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def distance(self, city):
        return np.sqrt((self.x - city.x) ** 2 + (self.y - city.y) ** 2)
    def __repr__(self):
        return f"({self.x}, {self.y})"
    @staticmethod
    def read_cities(file_name):
        cities = []
        with open(file_name, mode='r', newline='') as tsp:
            reader = csv.reader(tsp)
            for row in reader:
                if row:
                    x, y = map(float, row)
                    cities.append(City(x, y))
        return cities
# 총 거리 구하는 클래스
class Distance:
    @staticmethod
    def total_distance(route):
        return sum(route[i].distance(route[(i + 1) % len(route)]) for i in range(len(route)))
    @staticmethod
    def precompute_distances(cities):
        num_cities = len(cities)
        distance_matrix = np.zeros((num_cities, num_cities))
        for i in range(num_cities):
            for j in range(num_cities):
                if i != j:
                    coord_i = np.array([cities[i].x, cities[i].y])
                    coord_j = np.array([cities[j].x, cities[j].y])
                    distance_matrix[i][j] = np.linalg.norm(coord_i - coord_j)
        return distance_matrix
class Policy:
    @staticmethod
    def extract_policy(value_table):
        num_cities = value_table.shape[0]
        policy = np.zeros(num_cities, dtype=int)
        visited = set()
        current_city = 0
        for i in range(1, num_cities):
            visited.add(current_city)
            next_city = np.argmax(value_table[current_city])
            while next_city in visited:
                value_table[current_city][next_city] = -np.inf
                next_city = np.argmax(value_table[current_city])
            policy[i] = next_city
            current_city = next_city
        return policy
    @staticmethod
    def compute_total_distance(cities, policy):
        total_distance = 0
        for i in range(len(policy) - 1):
            total_distance += cities[policy[i]].distance(cities[policy[i + 1]])
        # Return to start city to complete the tour
        total_distance += cities[policy[-1]].distance(cities[policy[0]])
        return total_distance
class ValueIteration:
    @staticmethod
    def init_route(cities, population_size, generations, mutation_rate=0.4):
      # 경로 생성 함수
      def hybrid_initialization(cities, hybrid_factor=0.5):
        start_city = cities[0] # 시작 도시를 cities의 0번째 인덱스로 고정
        route = [start_city]
        remaining_cities = set(cities) - {start_city}
        while remaining_cities:
            current_city = route[-1]
            if random.random() < hybrid_factor:
                nearest_city = min(remaining_cities, key=lambda city: current_city.distance(current_city))
            else:
                nearest_city = random.choice(list(remaining_cities))
            route.append(nearest_city)
            remaining_cities.remove(nearest_city)
        # 경로를 순환으로 만듦
        return route
        route.append(start_city)
      # 경로 생성 함수
      def create_route(cities):
        return hybrid_initialization(cities)
      # 두 부모의 경로를 조합하여 자식 경로를 생성함.
      def crossover(parent1, parent2):
          start, end = sorted(random.sample(range(1, len(parent1) - 1), 2))
          child = [None] * len(parent1)
          child[start:end] = parent1[start:end]
          ptr = end
          for city in parent2:
              if city not in child:
                  if ptr >= len(child) - 1:
                      ptr = 1  # 첫 번째 도시와 마지막 도시는 고정
                  child[ptr] = city
                  ptr += 1
          child[0], child[-1] = parent1[0], parent1[0]  # 첫 번째 도시와 마지막 도시를 고정
          return child
      # 변이가 생긴 경우 새로운 경로를 생성함. 두 도시의 위치를 바꾸는 방식으로 변이 수행
      def mutate(route):
          if random.random() < mutation_rate:
              i, j = random.sample(range(1, len(route) - 1), 2)
              route[i], route[j] = route[j], route[i]
      # 현재 세대 중 가장 좋은 경로 선택함.
      def select(population, fitnesses, elite_size):
          elite_indices = np.argsort(fitnesses)[-elite_size:]
          elites = [population[i] for i in elite_indices]
          return elites
      # 여러개의 경로 생성
      population = [create_route(cities) for _ in range(population_size)]
      for generation in range(generations):
          # 경로의 총 거리를 기반으로 적합도를 계산함.
          fitnesses = [-Distance.total_distance(route) for route in population]
          min_fitness = min(fitnesses)
          # 적합도가 0보다 작거나 같은 경우에는 보정 수행.
          if min_fitness <= 0:
              fitnesses = [f + abs(min_fitness) + 1 for f in fitnesses]
          elites = select(population, fitnesses, elite_size=int(population_size * 0.1))
          new_population = elites.copy()
          while len(new_population) < population_size:
              parent1, parent2 = random.choices(population, weights=fitnesses, k=2)
              child = crossover(parent1, parent2)
              mutate(child)
              new_population.append(child)
          population = new_population
      best_route = max(population, key=lambda route: -Distance.total_distance(route))
      return best_route
    @staticmethod
    def init_value_table(cities, best_route, noise_factor=0.1, route_bonus=10):
      num_cities = len(cities)
      value_table = np.zeros((num_cities, num_cities))
      for i in range(num_cities):
          current_city = best_route[i]
          next_city = best_route[(i + 1) % num_cities]
          current_city_index = cities.index(current_city)
          next_city_index = cities.index(next_city)
          distance = current_city.distance(next_city)
          # Distance에  noise 추가
          noise = random.uniform(-noise_factor * distance, noise_factor * distance)
          value_with_noise = -distance + noise
          # 해당 경로가 best_route 안에 있다면, bouns 추가
          value_with_bonus = value_with_noise + route_bonus
          value_table[current_city_index][next_city_index] = value_with_bonus
      return value_table
    @staticmethod
    def value_iteration(cities, init_value_table, iterations, discount_factor=0.8, tolerance=1e-6):
        num_cities = len(cities)
        distance_matrix = Distance.precompute_distances(cities)
        value_table_history = []
        # Initialize value table with small random values
        value_table = init_value_table
        for iteration in range(iterations):
            new_value_table = np.zeros_like(value_table)
            for i in range(num_cities):
                for j in range(num_cities):
                    if i != j:
                        reward = -distance_matrix[i][j]
                        max_next_value = np.max(value_table[j])
                        new_value_table[i][j] = reward + discount_factor * max_next_value
            max_change = np.max(np.abs(new_value_table - value_table))
            value_table = new_value_table
            value_table_history.append(deepcopy(value_table))
            if max_change < tolerance:
                print(f"Converged after {iteration + 1} iterations.")
                break
        final_policy = Policy.extract_policy(value_table)
        final_distance = Policy.compute_total_distance(cities, final_policy)
        print(f"Final Total Distance: {final_distance}")
        return value_table, value_table_history

class QLearning:
    @staticmethod
    def value_table_to_q_table(value_table, distance_matrix, initial_q_value=0.1):
      q_table = np.full(value_table.shape, initial_q_value)

      # Value table에 따라 초기 Q-값 설정
      for state in range(value_table.shape[0]):
          for action in range(value_table.shape[1]):
              if value_table[state, action] == 1:
                  q_table[state, action] = 1.0
              else:
                  q_table[state, action] = -distance_matrix[state][action]

      return q_table

    def td_q_learning(cities, q_table, episodes, learning_rate=0.2, discount_factor=0.99, epsilon=0.1, epsilon_decay=0.99, min_epsilon=0.01):
      num_cities = len(cities)
      distance_matrix = Distance.precompute_distances(cities)
      q_table_history = []

      for episode in range(episodes):
          current_city = 0  # Always start from the first city
          visited = [current_city]

          while len(visited) < num_cities:
              if random.random() < epsilon:
                  next_city = random.choice([i for i in range(num_cities) if i not in visited])
              else:
                  next_city = np.argmax(q_table[current_city] + (np.isin(range(num_cities), visited) * -1e6))

              visited.append(next_city)

              reward = -distance_matrix[current_city][next_city]
              td_target = reward + discount_factor * np.max(q_table[next_city])
              td_error = td_target - q_table[current_city][next_city]
              q_table[current_city][next_city] += learning_rate * td_error

              current_city = next_city

          # Update the distance and reward for returning to the start city
          reward = -distance_matrix[current_city][0]
          td_target = reward + discount_factor * q_table[0][0]
          td_error = td_target - q_table[current_city][0]
          q_table[current_city][0] += learning_rate * td_error

          # Decay epsilon
          epsilon = max(min_epsilon, epsilon * epsilon_decay)

          # Save a copy of the Q-table for this episode
          q_table_history.append(deepcopy(q_table))

      return q_table, q_table_history

    def monte_carlo_q_learning(cities, q_table, episodes, learning_rate=0.2, discount_factor=0.99, epsilon=0.1, epsilon_decay=0.99, min_epsilon=0.01):
      num_cities = len(cities)
      distance_matrix = Distance.precompute_distances(cities)
      q_table_history = []

      for episode in range(episodes):
          current_city = 0  # Always start from the first city
          visited = [current_city]
          episode_states = []
          episode_rewards = []

          # Generate an episode
          while len(visited) < num_cities:
              if random.random() < epsilon:
                  next_city = random.choice([i for i in range(num_cities) if i not in visited])
              else:
                  next_city = np.argmax(q_table[current_city] + (np.isin(range(num_cities), visited) * -1e6))

              visited.append(next_city)
              episode_states.append((current_city, next_city))

              reward = -distance_matrix[current_city][next_city]
              episode_rewards.append(reward)

              current_city = next_city

          # Complete the tour by returning to the start city
          reward = -distance_matrix[current_city][0]
          episode_rewards.append(reward)
          episode_states.append((current_city, 0))

          # Update the Q-table using the episode
          G = 0  # Total reward
          for t in reversed(range(len(episode_states))):
              state = episode_states[t]
              reward = episode_rewards[t]
              G = reward + discount_factor * G
              q_table[state[0]][state[1]] += learning_rate * (G - q_table[state[0]][state[1]])

          # Decay epsilon
          epsilon = max(min_epsilon, epsilon * epsilon_decay)

          # Save a copy of the Q-table for this episode
          q_table_history.append(deepcopy(q_table))

      return q_table, q_table_history

class Draw:
    @staticmethod
    def plot_route(route):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        x = [city.x for city in route]
        y = [city.y for city in route]
        plt.plot(x + [x[0]], y + [y[0]], 'o-', label='Route')
        plt.plot(x[0], y[0], 'ro')
        plt.title("Traveling Salesman Path")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.tight_layout()
        plt.show()
    @staticmethod
    def compute_distance_history(cities, value_table_history):
      distance_history = []
      for value_table in value_table_history:
          policy = Policy.extract_policy(value_table)
          total_distance = Policy.compute_total_distance(cities, policy)
          distance_history.append(total_distance)
      return distance_history
    @staticmethod
    def plot_distance_convergence(distance_history):
      plt.figure(figsize=(10, 6))
      plt.plot(distance_history, label='Total Distance')
      plt.xlabel('Iteration')
      plt.ylabel('Total Distance')
      plt.title('Total Distance <Iterations>')
      plt.legend()
      plt.grid(True)
      plt.show()
    @staticmethod
    def plot_policy_route(cities, policy):
        route = [cities[city] for city in policy]
        route.append(route[0])  # Return to the start city
        Draw.plot_route(route)

def main():
    # CSV 파일에서 도시 읽기
    cities = City.read_cities('2024_AI_TSP.csv')
    
    # 초기 경로 생성
    init_route = ValueIteration.init_route(cities, population_size=100, generations=500)
    
    # 초기 가치 테이블 생성
    init_value_table = ValueIteration.init_value_table(cities, init_route)
    
    # 초기 정책 추출
    init_policy = Policy.extract_policy(init_value_table)
    
    # 초기 총 거리 계산
    init_total_distance = Policy.compute_total_distance(cities, init_policy)
    print(f"Initial Total Distance: {init_total_distance}")
    
    # 가치 반복 수행
    value_iteration_table, value_iteration_history = ValueIteration.value_iteration(
        cities, init_value_table, iterations=3
    )
    
    # 최종 정책 추출
    policy = Policy.extract_policy(value_iteration_table)
    
    # 최종 경로 생성
    route = [cities[idx] for idx in policy]
    
    # 최종 경로 시각화
    Draw.plot_route(route)
    # 거리 수렴 시각화
    distance_history = Draw.compute_distance_history(cities, value_iteration_history)
    Draw.plot_distance_convergence(distance_history)
    Draw.plot_policy_route(cities, policy)

    q_table = QLearning.value_table_to_q_table(value_iteration_table, Distance.precompute_distances(cities))
    q_learning_table, q_learning_history = QLearning.td_q_learning(cities, q_table, 500)
    policy = Policy.extract_policy(q_learning_table)

main()


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import namedtuple

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class QNet(nn.Module):

    def __init__(self, emb_dim, T=4, node_dim = 5):
        super(QNet, self).__init__()
        self.emb_dim = emb_dim
        self.T = T
        self.node_dim = node_dim

        self.theta1 = nn.Linear(self.node_dim, self.emb_dim, True)
        self.theta2 = nn.Linear(self.emb_dim, self.emb_dim, True)
        self.theta3 = nn.Linear(self.emb_dim, self.emb_dim, True)
        self.theta4 = nn.Linear(1, self.emb_dim, True)
        self.theta5 = nn.Linear(2*self.emb_dim, 1, True)
        self.theta6 = nn.Linear(self.emb_dim, self.emb_dim, True)
        self.theta7 = nn.Linear(self.emb_dim, self.emb_dim, True)

        self.layer = nn.Linear(self.emb_dim, self.emb_dim, True)

    def forward(self, xv, Ws):

        num_nodes = xv.shape[1]   # 전체 도시의 수
        batch_size = xv.shape[0]  # batch size


        # distance matrix의 값이 0인 곳은 0으로, 0 이상인 곳은 1로 채운 conn_matries
        # --> 대각 원소 = 0, 그 외의 원소 = 1
        conn_matrices = torch.where(Ws > 0, torch.ones_like(Ws), torch.zeros_like(Ws)).to(device)

        mu = torch.zeros(batch_size, num_nodes, self.emb_dim, device=device)

        # state 정보를 embedding
        s1 = self.theta1(xv)                     # (batch_size, num_nodes, 5) --> (batch_size, num_nodes, emb_dim)
        s1 = self.layer(F.relu(s1))              # (batch_size, num_nodes, emb_dim) --> (batch_size, num_nodes, emb_dim)

        # distance matrix 정보를 embedding
        s3_0 = Ws.unsqueeze(3)                   # (batch_size, num_nodes, num_nodes) --> (batch_size, num_nodes, num_nodes, 1)
        s3_1 = F.relu(self.theta4(s3_0))         # (batch_size, num_nodes, num_nodes, 1) --> (batch_size, num_nodes, num_nodes, emb_dim)
        s3_2 = torch.sum(s3_1, dim=1)            # (batch_size, num_nodes, num_nodes, emb_dim) --> (batch_size, num_nodes, emb_dim)
        s3 = self.theta3(s3_2)                   # (batch_size, num_nodes, emb_dim) --> (batch_size, num_nodes, emb_dim)


        # state 정보(s1)와 각 state에 대한 나머지 node들의 distance 정보(s3)를 함께 embedding
        for _ in range(self.T):
            s2 = self.theta2(conn_matrices.matmul(mu))    # state와 action이 동일한 경우 (대각 원소)를 제외하고 정보 융합
            mu = F.relu(s1 + s2 + s3)

        # 전체적인 state와 distance에 대한 정보를 모든 노드에 동일하게 제공
        global_state = self.theta6(torch.sum(mu, dim=1, keepdim=True).repeat(1, num_nodes, 1))

        # 각각의 state에 대한 정보
        local_action = self.theta7(mu)

        # 전체적인 정보와, 각각의 state에서의 정보를 함께 융합하여 Q-value 예측
        out = F.relu(torch.cat([global_state, local_action], dim=2))
        return self.theta5(out).squeeze(dim=2)

class QTrainer():
    def __init__(self, model, optimizer, lr_scheduler):
        # QNetwork 인스턴스
        self.model = model

        # 학습에 활용할 QNetwork 학습 구성요소
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = nn.MSELoss()


    def predict(self, state_tsr, W):
        # batch가 1인 인풋을 가정. inference 시 호출
        with torch.no_grad():
            estimated_q_value = self.model(state_tsr.unsqueeze(0), W.unsqueeze(0))

        return estimated_q_value[0]


    def get_best_action(self, state_tsr, state):
        """ 주어진 state에 대해 최적의 greedy action을 선택하는 단계.
            다음 노드(aciton)의 index와 추정된 q_value.
        """
        W = state.W
        estimated_q_value = self.predict(state_tsr, W)
        sorted_q_value_idx = estimated_q_value.argsort(descending=True)

        solution = state.partial_solution

        already_in = set(solution)
        for idx in sorted_q_value_idx.tolist():
            if (len(solution) == 0 or W[solution[-1], idx] > 0) and idx not in already_in:
                return idx, estimated_q_value[idx].item()


    def batch_update(self, states_tsrs, Ws, actions, targets):
        """ Batch단위의 (embedding of state, distance matrix, action, target_q_value)를 통해 Gradient를 통한 최적화를 수행하는 단계.
            states_tsrs: list of (single) state tensors
            Ws: list of W tensors
            actions: list of actions taken
            targets: list of targets (resulting estimated q_value after taking the actions)
        """
        Ws_tsr = torch.stack(Ws).to(device)
        xv = torch.stack(states_tsrs).to(device)
        self.optimizer.zero_grad()

        estimated_q_value = self.model(xv, Ws_tsr)[range(len(actions)), actions]


        loss = self.loss_fn(estimated_q_value, torch.tensor(targets, device=device))
        loss_val = loss.item()

        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        return loss_val

SEED = 1                     # A seed for the random number generator
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Graph
NR_NODES = 998               # Number of nodes N
EMBEDDING_DIMENSIONS = 5     # Embedding dimension D
EMBEDDING_ITERATIONS_T = 1   # Number of embedding iterations T

# Learning
NR_EPISODES = 100
MEMORY_CAPACITY = 10000
N_STEP_QL = 2                # Number of steps (n) in n-step Q-learning to wait before computing target reward estimate
BATCH_SIZE = 16

GAMMA = 0.9
INIT_LR = 5e-3
LR_DECAY_RATE = 1. - 2e-5    # learning rate decay

MIN_EPSILON = 0.1
EPSILON_DECAY_RATE = 6e-4    # epsilon decay

State = namedtuple('State', ('W', 'coords', 'partial_solution'))
Experience = namedtuple('Experience', ('state', 'state_tsr', 'action', 'reward', 'next_state', 'next_state_tsr'))


def state2tens(state):
    solution = set(state.partial_solution)
    sol_last_node = state.partial_solution[-1] if len(state.partial_solution) > 0 else -1
    sol_first_node = state.partial_solution[0] if len(state.partial_solution) > 0 else -1
    coords = state.coords
    nr_nodes = coords.shape[0]

    xv = [[(1 if i in solution else 0),           # 해당 노드를 방문 했는지 여부
           (1 if i == sol_first_node else 0),     # 해당 노드가 시작 노드인지 여부
           (1 if i == sol_last_node else 0),      # 해당 노드가 마지막 노드인지 여부
           coords[i,0],                           # 해당 노드의 x좌표
           coords[i,1]                            # 해당 노드의 y좌표
          ] for i in range(nr_nodes)]

    return torch.tensor(xv, dtype=torch.float32, requires_grad=False, device=device)


class Memory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.nr_inserts = 0

    def remember(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = experience
        self.position = (self.position + 1) % self.capacity
        self.nr_inserts += 1

    def sample_batch(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return min(self.nr_inserts, self.capacity)

def total_distance(solution, W):
    if len(solution) < 2:
        return 0

    total_dist = 0
    for i in range(len(solution) - 1):
        total_dist += W[solution[i], solution[i+1]].item()

    if len(solution) == W.shape[0]:
        total_dist += W[solution[-1], solution[0]].item()

    return total_dist


def is_state_final(state):
    return len(set(state.partial_solution)) == state.W.shape[0]


def get_next_neighbor_random(state):
    solution, W = state.partial_solution, state.W

    if len(solution) == 0:
        return random.choice(range(W.shape[0]))
    already_in = set(solution)
    candidates = list(filter(lambda n: n.item() not in already_in, W[solution[-1]].nonzero()))
    if len(candidates) == 0:
        return None
    return random.choice(candidates).item()


def get_distance_matrix(x, num_cities=998):
    x = torch.tensor(x)
    x1, x2 = x[:,0:1], x[:,1:2]
    d1 = x1 - (x1.T).repeat(num_cities,1)
    d2 = x2 - (x2.T).repeat(num_cities,1)
    distance_matrix = (d1**2 + d2**2)**0.5   # Euclidean Distance
    return distance_matrix.numpy()


def init_model(fname=None):
    Q_net = QNet(EMBEDDING_DIMENSIONS, T=EMBEDDING_ITERATIONS_T).to(device)
    optimizer = optim.Adam(Q_net.parameters(), lr=INIT_LR)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY_RATE)

    if fname is not None:
        checkpoint = torch.load(fname)
        Q_net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    Q_trainer = QTrainer(Q_net, optimizer, lr_scheduler)
    return Q_trainer, Q_net, optimizer, lr_scheduler

# TSP Data Load
coords = np.array(pd.read_csv('2024_AI_TSP.csv', header=None))

# make distance matrix
W_np = get_distance_matrix(coords)

# init Trainer, Model
Q_trainer, Q_net, optimizer, lr_scheduler = init_model()

# generate memory
memory = Memory(MEMORY_CAPACITY)


losses = []
path_lengths = []
found_solutions = dict()
current_min_med_length = float('inf')


for episode in range(NR_EPISODES):

    # tensor (distance matrix)
    W = torch.tensor(W_np, dtype=torch.float32, requires_grad=False, device=device)

    # start node = 0
    solution = [0]

    # current state
    current_state = State(partial_solution=solution, W=W, coords=coords)
    current_state_tsr = state2tens(current_state)


    # define state, state_tsrs(embedding), reward, action list
    states = [current_state]
    states_tsrs = [current_state_tsr]
    rewards = []
    actions = []


    # current value of epsilon
    epsilon = max(MIN_EPSILON, (1-EPSILON_DECAY_RATE)**episode)


    while not is_state_final(current_state):

        # select next node
        if epsilon >= random.random():
            next_node = get_next_neighbor_random(current_state)
        else:
            next_node, est_reward = Q_trainer.get_best_action(current_state_tsr, current_state)


        # append next node to solution
        next_solution = solution + [next_node]

        # calulate reward
        reward = -(total_distance(next_solution, W) - total_distance(solution, W))


        next_state = State(partial_solution=next_solution, W=W, coords=coords)
        next_state_tsr = state2tens(next_state)

        states.append(next_state)
        states_tsrs.append(next_state_tsr)
        rewards.append(reward)
        actions.append(next_node)


        if len(solution) >= N_STEP_QL:
            memory.remember(Experience(state=states[-N_STEP_QL],
                                       state_tsr=states_tsrs[-N_STEP_QL],
                                       action=actions[-N_STEP_QL],
                                       reward=sum(rewards[-N_STEP_QL:]),
                                       next_state=next_state,
                                       next_state_tsr=next_state_tsr))

        if is_state_final(next_state):
            for n in range(1, N_STEP_QL):
                memory.remember(Experience(state=states[-n],
                                           state_tsr=states_tsrs[-n],
                                           action=actions[-n],
                                           reward=sum(rewards[-n:]),
                                           next_state=next_state,
                                           next_state_tsr=next_state_tsr))


        current_state = next_state
        current_state_tsr = next_state_tsr
        solution = next_solution


        loss = None
        if len(memory) >= BATCH_SIZE:

            # sampling batch experience
            experiences = memory.sample_batch(BATCH_SIZE)

            batch_states_tsrs = [e.state_tsr for e in experiences]
            batch_Ws = [e.state.W for e in experiences]
            batch_actions = [e.action for e in experiences]
            batch_targets = []


            for i, experience in enumerate(experiences):
                target = experience.reward
                if not is_state_final(experience.next_state):
                    _, best_q_value = Q_trainer.get_best_action(experience.next_state_tsr, experience.next_state)
                    target += GAMMA * best_q_value
                batch_targets.append(target)

            loss = Q_trainer.batch_update(batch_states_tsrs, batch_Ws, batch_actions, batch_targets)
            losses.append(loss)

    length = total_distance(solution, W)
    path_lengths.append(length)

    if episode % 10 == 0:
        print('Ep %d. Loss = %.3f, length = %.3f, epsilon = %.4f, lr = %.4f' % (
            episode, (-1 if loss is None else loss), length, epsilon,
            Q_trainer.optimizer.param_groups[0]['lr']))
        found_solutions[episode] = (W.clone(), coords.copy(), [n for n in solution])

solution = [0]
current_state = State(partial_solution=solution, W=W, coords=coords)
current_state_tsr = state2tens(current_state)

while not is_state_final(current_state):
    next_node, est_reward = Q_trainer.get_best_action(current_state_tsr,
                                                    current_state)

    solution = solution + [next_node]
    current_state = State(partial_solution=solution, W=W, coords=coords)
    current_state_tsr = state2tens(current_state)

print("Final solution : ", str(solution))