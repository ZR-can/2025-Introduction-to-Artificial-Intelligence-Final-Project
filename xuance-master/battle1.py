import os
import numpy as np
import torch
from xuance.environment import make_envs
from xuance.torch.agents import MAPPO_Agents
from xuance.common import load_config
from argparse import Namespace
import time

class DirectAdversarialEvaluator:
    def __init__(self, model_path_team1="F:\\best_model1.pth", model_path_team2="F:\\best_model2.pth"):
        """
        极简对抗评估系统
        使用默认模型路径或自定义路径运行
        Args:
            model_path_team1: 红方模型路径 (默认 "F:\\best_model1.pth")
            model_path_team2: 蓝方模型路径 (默认 "F:\\best_model2.pth")
        """
        # 自动检测环境类型
        self.env_type = self._detect_env_type(model_path_team1)
        self.env = self._init_environment()
        
        # 直接加载模型
        self.team1 = self._load_model(model_path_team1, 'team1')
        self.team2 = self._load_model(model_path_team2, 'team2')
        
        # 战斗记录
        self.stats = {'team1_wins': 0, 'team2_wins': 0, 'draws': 0}

    def _detect_env_type(self, model_path):
        """从模型路径推断环境类型"""
        model_path = model_path.lower()
        if 'football' in model_path:
            return 'football'
        elif 'sc2' in model_path or 'smac' in model_path:
            return 'sc2'
        elif 'mpe' in model_path or 'simple' in model_path:
            return 'mpe'
        else:
            raise ValueError("无法从模型路径推断环境类型，请确保路径包含football/sc2/mpe等关键字")

    def _init_environment(self):
        """根据环境类型初始化"""
        env_map = {
            'football': 'academy_3_vs_1_with_keeper',
            'sc2': '3m_vs_3z',
            'mpe': 'simple_adversary_v3'
        }
        return make_envs(env_id=env_map[self.env_type], render_mode="human")

    def _load_model(self, model_path, team_name):
        """加载模型并自动构建配置"""
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 创建最小化配置
        args = Namespace(
            agent='mappo',
            agent_keys=[team_name],
            device='cuda' if torch.cuda.is_available() else 'cpu',
            n_agents=3 if self.env_type in ['sc2', 'football'] else 2,
            dim_obs=128,  # 需根据实际环境调整
            dim_act=10,   # 需根据实际环境调整
            hidden_sizes=[64, 64],
            use_rnn=False
        )
        
        agent = MAPPO_Agents(args, self.env, args.device)
        agent.load_model(model_path)
        print(f"成功加载 {team_name} 模型: {model_path}")
        return agent

    def run_battle(self, max_steps=1000):
        """执行单场对抗"""
        obs = self.env.reset()
        done = False
        step = 0
        
        while not done and step < max_steps:
            # 获取双方动作
            team1_act = self._get_actions(obs, self.team1, 'team1')
            team2_act = self._get_actions(obs, self.team2, 'team2')
            
            # 合并动作
            actions = {**team1_act, **team2_act} if isinstance(team1_act, dict) else {
                'team1': team1_act,
                'team2': team2_act
            }
            
            # 环境执行
            next_obs, rewards, done, _ = self.env.step(actions)
            obs = next_obs
            step += 1
            
            # 胜负判定
            if done:
                if rewards['team1'] > rewards['team2']:
                    self.stats['team1_wins'] += 1
                    print("红方获胜！")
                elif rewards['team2'] > rewards['team1']:
                    self.stats['team2_wins'] += 1
                    print("蓝方获胜！")
                else:
                    self.stats['draws'] += 1
                    print("平局！")

    def _get_actions(self, obs, agent, team_key):
        """通用动作获取方法"""
        if self.env_type == 'sc2':
            return agent.action({
                'obs': np.array(obs[team_key]['obs']),
                'avail_actions': np.array(obs[team_key]['avail_actions'])
            })[0]
        else:  # MPE/Football
            return agent.action(np.array(obs[team_key]))[0]

    def run_tournament(self, n_battles=10):
        """运行对抗赛"""
        print(f"\n开始 {self.env_type.upper()} 对抗赛")
        print(f"红方模型: F:\\best_model1.pth")
        print(f"蓝方模型: F:\\best_model2.pth")
        
        for i in range(n_battles):
            print(f"\n第 {i+1} 场战斗开始...")
            self.run_battle()
            print(f"当前战绩: 红方 {self.stats['team1_wins']} 胜 | 蓝方 {self.stats['team2_wins']} 胜 | 平局 {self.stats['draws']}")
        
        print("\n最终结果:")
        print(f"红方胜率: {self.stats['team1_wins']/n_battles*100:.1f}%")
        print(f"蓝方胜率: {self.stats['team2_wins']/n_battles*100:.1f}%")

if __name__ == "__main__":
    # 直接使用默认路径运行
    evaluator = DirectAdversarialEvaluator()
    evaluator.run_tournament(n_battles=10)