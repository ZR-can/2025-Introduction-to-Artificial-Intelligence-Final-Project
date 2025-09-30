import os
import sys
import time
import numpy as np

# 添加xuance源码路径到系统路径（确保能正确导入）
xuance_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, xuance_path)

try:
    # 开发版专用导入方式
    from xuance.configs import load_config
    from xuance import get_runner
except ImportError as e:
    raise ImportError(
        f"无法导入XuanCe模块，请确保：\n"
        f"1. 已从源码安装 (在{xuance_path}目录下运行 pip install -e .)\n"
        f"2. 错误详情: {str(e)}"
    )

class QMIX_Inference:
    def __init__(self, model_dir: str):
        """初始化推理器"""
        self.model_dir = os.path.abspath(model_dir)
        self._check_files()
        self.config = self._load_config()
        self.runner = None

    def _check_files(self):
        """检查必需文件"""
        required = {
            'config.yaml': '配置文件',
            'best_model.pth': '模型权重'
        }
        missing = []
        for f, desc in required.items():
            if not os.path.exists(os.path.join(self.model_dir, f)):
                missing.append(f"{desc}({f})")
        if missing:
            raise FileNotFoundError(
                f"在 {self.model_dir} 中缺少:\n" + 
                "\n".join(f"• {m}" for m in missing)
            )

    def _load_config(self):
        """加载配置文件（兼容开发版）"""
        try:
            return load_config(os.path.join(self.model_dir, "config.yaml"))
        except Exception as e:
            raise RuntimeError(
                f"加载配置文件失败:\n"
                f"1. 请检查config.yaml格式是否正确\n"
                f"2. 错误详情: {str(e)}"
            )

    def run(self, n_episodes=5, render=True):
        """运行推理"""
        print(f"\n{'='*50}\n初始化QMIX推理环境\n{'='*50}")
        print(f"环境ID: {self.config.env_id}")
        print(f"设备: {getattr(self.config, 'device', 'cuda:0')}")
        
        # 初始化runner
        self.runner = get_runner(
            method='qmix',
            env='marl',
            env_id=self.config.env_id,
            config=self.config,
            is_test=True
        )
        
        # 加载模型
        model_path = os.path.join(self.model_dir, "best_model.pth")
        self.runner.load_model(model_path)
        
        # 运行测试
        rewards = []
        print(f"\n{'='*50}\n开始测试 (共{n_episodes}回合)\n{'='*50}")
        for ep in range(1, n_episodes + 1):
            reward = self.runner.test(render=render)
            rewards.append(reward)
            print(f"回合 {ep}/{n_episodes} | 奖励: {reward:.2f}")
        
        print(f"\n{'='*50}\n测试完成\n{'='*50}")
        print(f"平均奖励: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        return rewards

if __name__ == "__main__":
    # ============== 配置区域 ==============
    MODEL_DIR = r"C:\Users\陈笑涵\OneDrive\桌面\xuance-master\examples\qmix\models\qmix\2m_vs_1z\seed_1_SatJul1217_33_062025"
    N_EPISODES = 3
    RENDER = True
    # ====================================
    
    try:
        infer = QMIX_Inference(MODEL_DIR)
        infer.run(n_episodes=N_EPISODES, render=RENDER)
    except Exception as e:
        print(f"\n{'!'*50}\n错误发生: {str(e)}\n{'!'*50}")
    finally:
        input("\n按Enter键退出...")