from utils.visualization import plot_bloch_sphere

if __name__ == '__main__':
    # 定义一个状态向量
    state = [0.4, 0.6, 0.9]
    # 绘制带有状态向量的布洛赫球
    plot_bloch_sphere(state)