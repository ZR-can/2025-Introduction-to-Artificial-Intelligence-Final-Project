import gym
from gym.spaces import Discrete, Dict, Box
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

class MultiAgentEnv(gym.Env):
    def __init__(self):
        # 定义环境地图
        self.map_file = r"E:\code\RL\xuance-master\xuance\environment\hmarl_env\map.jpg"  # 替换为你的地图路径
        self.map_image = Image.open(self.map_file)
        self.map_size = self.map_image.size  # 地图尺寸

        # 定义动作空间（简化为移动和攻击）
        self.action_space = Discrete(15)  # 0: 不动, 1-4: 上下左右移动, 5-9: 攻击不同类型敌人

        # 定义观察空间（假设每个智能体的状态由其位置和所处地形表示）
        self.observation_space = Dict({
            'agent_positions': Box(low=0, high=self.map_size[0], shape=(15, 2), dtype=np.int32),
            'terrain_map': Box(low=0, high=255, shape=self.map_size + (3,), dtype=np.uint8)  # RGB图像
        })

        # 初始化智能体位置
        self.agent_positions = np.zeros((15, 2), dtype=np.int32)  # 初始位置全部设为(0, 0)，需要根据地图随机生成位置

        # 初始化地图
        self.terrain_map = np.array(self.map_image)

    def reset(self):
        # 在每个episode开始时重置环境
        # 随机放置智能体的初始位置
        self.agent_positions = np.zeros((15, 2), dtype=np.int32)
        for i in range(15):
            x = random.randint(0, self.map_size[0] - 1)
            y = random.randint(0, self.map_size[1] - 1)
            self.agent_positions[i] = [x, y]

        # 返回初始观察
        observation = {
            'agent_positions': np.copy(self.agent_positions),
            'terrain_map': np.copy(self.terrain_map)
        }
        return observation

    def step(self, actions):
        # 示例中简化为不进行实际动作，只更新位置
        for i, action in enumerate(actions):
            if action < 5:
                # 移动动作
                direction = action - 1  # 1: 上, 2: 下, 3: 左, 4: 右
                if direction == 0:
                    self.agent_positions[i, 1] -= 1  # 向上移动
                elif direction == 1:
                    self.agent_positions[i, 1] += 1  # 向下移动
                elif direction == 2:
                    self.agent_positions[i, 0] -= 1  # 向左移动
                elif direction == 3:
                    self.agent_positions[i, 0] += 1  # 向右移动
            elif action < 15:
                target_agent_index = action - 5
                

                    # 更新观察状态
        observation = {
            'agent_positions': np.copy(self.agent_positions),
            'terrain_map': np.copy(self.terrain_map)
        }
        
        # 返回观察、奖励、终止状态和其他信息
        reward = 0.0  # 暂时设置为0
        done = False  # 暂时设置为False
        info = {}  # 暂时空字典
        return observation, reward, done, info

    def render(self, mode='human'):
        # 可视化环境
        plt.imshow(self.terrain_map)
        for i in range(15):
            plt.text(self.agent_positions[i, 0], self.agent_positions[i, 1], f'Agent {i}', color='red', ha='center')
            # 示例中显示子弹飞行轨迹的简化方式
            # 这里假设攻击动作为从智能体 i 到智能体 (i+5)%15 的攻击
            # 实际情况需要根据具体的逻辑来绘制子弹飞行轨迹
            plt.arrow(self.agent_positions[i, 0], self.agent_positions[i, 1], 
                      self.agent_positions[(i+5)%15, 0] - self.agent_positions[i, 0], 
                      self.agent_positions[(i+5)%15, 1] - self.agent_positions[i, 1], 
                      color='red', linestyle='dotted', linewidth=1, head_width=0.2)
        plt.show()
# 测试环境是否正常工作
if __name__ == "__main__":
    env = MultiAgentEnv()
    env.reset()
    env.render()