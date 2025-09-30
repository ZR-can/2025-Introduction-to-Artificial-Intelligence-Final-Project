import os
import time
import numpy as np
from xuance import get_runner
from xuance.common import load_config

class SC2Inference:
    # 内置路径（直接使用您提供的两个路径）
    MODEL_PATHS = {
        "qmix": r"C:\Users\陈笑涵\OneDrive\桌面\xuance-master\examples\qmix\models\qmix\2m_vs_1z\seed_1_SatJul1217_33_062025",
        "mappo": r"C:\Users\陈笑涵\OneDrive\桌面\xuance-master\examples\mappo\mappo_sc2_configs"
    }

    def __init__(self, model_type="qmix"):
        """初始化推理器
        Args:
            model_type: qmix 或 mappo
        """
        if model_type not in self.MODEL_PATHS:
            raise ValueError(f"模型类型必须是 'qmix' 或 'mappo'，当前输入: {model_type}")
        
        self.model_type = model_type
        self.model_dir = self.MODEL_PATHS[model_type]
        self._check_files()
        self.config = self._load_config()
        self.runner = None

    def _check_files(self):
        """检查模型文件完整性"""
        required_files = {
            'config.yaml': '配置文件',
            'best_model.pth': '模型权重文件',
            'policy.pth': '模型权重文件（备选）'
        }
        
        # 检查至少有一个模型文件存在
        model_files = ['best_model.pth', 'policy.pth']
        self.model_file = None
        for f in model_files:
            if os.path.exists(os.path.join(self.model_dir, f)):
                self.model_file = f
                break
        
        if not self.model_file:
            raise FileNotFoundError(
                f"在目录 {self.model_dir} 中未找到模型文件\n"
                f"请确保存在以下文件之一:\n"
                + "\n".join(f"• {f}" for f in model_files)
            )

        # 检查配置文件
        if not os.path.exists(os.path.join(self.model_dir, "config.yaml")):
            raise FileNotFoundError(f"缺少配置文件: {os.path.join(self.model_dir, 'config.yaml')}")

    def _load_config(self):
        """加载配置文件"""
        config = load_config(os.path.join(self.model_dir, "config.yaml"))
        config.env_id = "2m_vs_1z"  # 固定测试场景
        config.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        return config

    def run(self, n_episodes=5, render=True, realtime=False, save_video=False):
        """运行推理
        Args:
            n_episodes: 测试回合数
            render: 是否实时渲染
            realtime: 是否星际2原速模式
            save_video: 是否保存视频
        """
        print(f"\n{'='*50}\n初始化 {self.model_type.upper()} 推理引擎\n{'='*50}")
        print(f"模型目录: {self.model_dir}")
        print(f"测试场景: {self.config.env_id}")
        print(f"设备: {self.config.device}")

        # 初始化runner
        self.runner = get_runner(
            method=self.model_type,
            env="marl",
            env_id=self.config.env_id,
            config=self.config,
            is_test=True
        )

        # 加载模型
        model_path = os.path.join(self.model_dir, self.model_file)
        self.runner.load_model(model_path)
        print(f"加载模型: {self.model_file}")

        # 设置SMAC实时模式
        if hasattr(self.runner.envs, 'env'):
            self.runner.envs.env.set_realtime(realtime)
            print(f"游戏速度: {'原速' if realtime else '加速'}")

        # 运行测试
        print(f"\n{'='*50}\n开始测试 (共{n_episodes}回合)\n{'='*50}")
        rewards = []
        start_time = time.time()
        
        for ep in range(1, n_episodes + 1):
            ep_reward = self.runner.test(
                render=render,
                save_video=save_video,
                video_path=f"{self.model_type}_ep{ep}.mp4" if save_video else None
            )
            rewards.append(ep_reward)
            print(f"回合 {ep}/{n_episodes} | 团队奖励: {ep_reward:.1f}")

        # 输出统计
        print(f"\n{'='*50}\n测试完成\n{'='*50}")
        print(f"平均奖励: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
        print(f"最大奖励: {np.max(rewards):.1f}")
        print(f"最小奖励: {np.min(rewards):.1f}")
        print(f"总耗时: {time.time()-start_time:.1f}秒")

if __name__ == "__main__":
    import torch
    
    # ============== 用户配置 ==============
    MODEL_TYPE = "qmix"  # 可选: "qmix" 或 "mappo"
    N_EPISODES = 5       # 测试回合数
    RENDER = True        # 是否实时渲染
    REALTIME = False     # 是否星际2原速模式
    SAVE_VIDEO = False   # 是否保存视频
    # ====================================
    
    try:
        infer = SC2Inference(MODEL_TYPE)
        infer.run(
            n_episodes=N_EPISODES,
            render=RENDER,
            realtime=REALTIME,
            save_video=SAVE_VIDEO
        )
    except Exception as e:
        print(f"\n{'!'*50}\n错误发生: {str(e)}\n{'!'*50}")
    finally:
        input("\n按Enter键退出...")