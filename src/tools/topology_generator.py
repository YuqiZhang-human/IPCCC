#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
通用网络拓扑生成器

功能：
1. 根据配置生成不同类型的网络拓扑：
   - fat_tree   : x 元 y 层 fat-tree 风格拓扑（带 host）
   - clos       : CLOS 架构（多层，每层节点数自定义，相邻层全连接）
   - random     : 随机连通图（生成树 + 额外随机边）
   - mesh       : 网状网络，保证每个节点度数 ≥ d

2. 输出：
   - 邻接矩阵 txt 文件：0/1，空格分隔
   - 拓扑可视化 png 文件

使用方式：
- 直接运行本脚本，修改 CONFIG 中的参数即可，无需命令行参数。
"""

import os
import random
from typing import List, Dict, Tuple

import networkx as nx
import matplotlib.pyplot as plt


# ==========================
# 配置区：在这里改参数
# ==========================

CONFIG = {
    # 拓扑类型: "fat_tree" | "clos" | "random" | "mesh"
    "topology_type": "mesh",

    # 输出目录（相对于本脚本所在目录）
    "output_dir": "topology_output",

    # 随机种子（方便复现）
    "random_seed": 42,

    # ---- fat-tree 参数 ----
    "fat_tree": {
        "x_hosts_per_leaf": 2,     # 每个叶子交换机下挂的 host 数量
        "y_switch_layers": 3,      # 交换机层数（不含 host 层）
        "branching_factor": 2      # 每下一层交换机数量 = 上一层 * branching_factor
    },

    # ---- CLOS 参数 ----
    "clos": {
        # 每层的节点数，例如 [4, 8, 8, 4]
        "layer_sizes": [4, 8, 8, 4]
    },

    # ---- 随机网络参数 ----
    "random": {
        "node_count": 20,
        "edge_prob": 0.1           # 在生成树基础上额外随机加边的概率
    },

    # ---- 网状网络参数 ----
    "mesh": {
        "node_count": 12,
        "min_degree": 3            # 要求每个节点度数 ≥ min_degree（需 < node_count）
    }
}


# ==========================
# 工具函数
# ==========================

def ensure_dir(path: str) -> None:
    """确保目录存在"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def graph_to_adjacency_matrix(G: nx.Graph) -> List[List[int]]:
    """将无向图转为邻接矩阵（节点按升序排序）"""
    nodes = sorted(G.nodes())
    index_map = {node: idx for idx, node in enumerate(nodes)}
    n = len(nodes)
    mat = [[0] * n for _ in range(n)]

    for u, v in G.edges():
        i = index_map[u]
        j = index_map[v]
        mat[i][j] = 1
        mat[j][i] = 1

    return mat


def save_adjacency_matrix(
    adjacency: List[List[int]], filepath: str
) -> None:
    """将邻接矩阵保存为 txt 文件（0/1，空格分隔）"""
    with open(filepath, "w", encoding="utf-8") as f:
        for row in adjacency:
            line = " ".join(str(x) for x in row)
            f.write(line + "\n")


def draw_graph(
    G: nx.Graph,
    filepath: str,
    pos: Dict[int, Tuple[float, float]] = None,
    title: str = ""
) -> None:
    """绘制拓扑图并保存为 png"""
    plt.figure(figsize=(8, 6))

    if pos is None:
        # 默认弹簧布局
        pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx_nodes(G, pos, node_size=300, node_color="lightblue")
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.close()


# ==========================
# 拓扑生成器
# ==========================

class TopologyGenerator:
    def __init__(self, config: Dict):
        self.config = config
        seed = config.get("random_seed", None)
        if seed is not None:
            random.seed(seed)

    # ---- 总入口 ----
    def generate(self) -> Tuple[nx.Graph, List[List[int]], str]:
        ttype = self.config.get("topology_type", "fat_tree")
        ttype = ttype.lower()

        if ttype == "fat_tree":
            G, name, pos = self._generate_fat_tree()
        elif ttype == "clos":
            G, name, pos = self._generate_clos()
        elif ttype == "random":
            G, name, pos = self._generate_random()
        elif ttype == "mesh":
            G, name, pos = self._generate_mesh()
        else:
            raise ValueError(f"未知拓扑类型: {ttype}")

        adjacency = graph_to_adjacency_matrix(G)
        return G, adjacency, name

    # ---- 1. fat-tree ----
    def _generate_fat_tree(self) -> Tuple[nx.Graph, str, Dict[int, Tuple[float, float]]]:
        """
        生成一个简化的 fat-tree 风格拓扑：
        - y 层交换机（0:核心层 ... y-1:叶子交换机层）
        - 每层交换机数量依次乘以 branching_factor
        - 每层之间全连接
        - 每个叶子交换机接 x_hosts_per_leaf 个 host
        """
        params = self.config["fat_tree"]
        x_hosts_per_leaf = int(params["x_hosts_per_leaf"])
        y_layers = int(params["y_switch_layers"])
        branching_factor = int(params["branching_factor"])

        G = nx.Graph()

        switch_layers: List[List[int]] = []
        current_node_id = 0

        # 生成交换机层：从核心层到叶子层
        num_switches = 1
        for layer in range(y_layers):
            layer_nodes = list(range(current_node_id, current_node_id + num_switches))
            current_node_id += num_switches
            switch_layers.append(layer_nodes)
            num_switches *= branching_factor

        # 在交换机层之间建立全连接
        for layer in range(y_layers - 1):
            upper = switch_layers[layer]
            lower = switch_layers[layer + 1]
            for u in upper:
                for v in lower:
                    G.add_edge(u, v)

        # 叶子层挂 host
        leaf_switches = switch_layers[-1]
        host_nodes: List[int] = []
        for sw in leaf_switches:
            for _ in range(x_hosts_per_leaf):
                host = current_node_id
                current_node_id += 1
                G.add_edge(sw, host)
                host_nodes.append(host)

        total_nodes = G.number_of_nodes()
        name = f"fat_tree_y{y_layers}_x{x_hosts_per_leaf}_bf{branching_factor}_N{total_nodes}"

        # 简单分层布局：按 layer 排纵坐标
        pos: Dict[int, Tuple[float, float]] = {}
        # 交换机层
        for layer_idx, nodes in enumerate(switch_layers):
            n = len(nodes)
            for i, node in enumerate(sorted(nodes)):
                x = (i - (n - 1) / 2)  # 居中
                y = -layer_idx
                pos[node] = (x, y)
        # host 放在最下面一层
        host_y = -(y_layers + 1)
        n_hosts = len(host_nodes)
        for i, node in enumerate(sorted(host_nodes)):
            x = (i - (n_hosts - 1) / 2)
            pos[node] = (x, host_y)

        return G, name, pos

    # ---- 2. CLOS 架构 ----
    def _generate_clos(self) -> Tuple[nx.Graph, str, Dict[int, Tuple[float, float]]]:
        """
        生成 CLOS 架构：
        - layer_sizes = [n0, n1, ..., n_{L-1}]
        - 相邻层之间全连接
        """
        params = self.config["clos"]
        layer_sizes: List[int] = list(params["layer_sizes"])

        G = nx.Graph()
        layers: List[List[int]] = []

        current_node_id = 0
        for layer_idx, size in enumerate(layer_sizes):
            nodes = list(range(current_node_id, current_node_id + int(size)))
            current_node_id += int(size)
            layers.append(nodes)

        # 相邻层全连接
        L = len(layers)
        for layer_idx in range(L - 1):
            upper = layers[layer_idx]
            lower = layers[layer_idx + 1]
            for u in upper:
                for v in lower:
                    G.add_edge(u, v)

        total_nodes = G.number_of_nodes()
        name = "clos_" + "-".join(str(s) for s in layer_sizes) + f"_N{total_nodes}"

        # 简单分层布局
        pos: Dict[int, Tuple[float, float]] = {}
        for layer_idx, nodes in enumerate(layers):
            n = len(nodes)
            for i, node in enumerate(sorted(nodes)):
                x = (i - (n - 1) / 2)
                y = -layer_idx
                pos[node] = (x, y)

        return G, name, pos

    # ---- 3. 随机网络 ----
    def _generate_random(self) -> Tuple[nx.Graph, str, Dict[int, Tuple[float, float]]]:
        """
        随机连通图：
        - 先生成一棵随机生成树（保证连通）
        - 再在非树边上以概率 p 加边
        """
        params = self.config["random"]
        n = int(params["node_count"])
        p = float(params["edge_prob"])

        G = nx.Graph()
        G.add_nodes_from(range(n))

        # 随机生成树（确保连通）
        nodes = list(range(n))
        random.shuffle(nodes)
        for i in range(1, n):
            u = nodes[i]
            v = random.choice(nodes[:i])
            G.add_edge(u, v)

        # 在非树边上加随机边
        for u in range(n):
            for v in range(u + 1, n):
                if G.has_edge(u, v):
                    continue
                if random.random() < p:
                    G.add_edge(u, v)

        name = f"random_N{n}_p{p:.2f}"
        pos = nx.spring_layout(G, seed=42)
        return G, name, pos

    # ---- 4. 网状网络 ----
    def _generate_mesh(self) -> Tuple[nx.Graph, str, Dict[int, Tuple[float, float]]]:
        """
        网状网络：
        - 给定 N 和 min_degree
        - 先生成一棵随机生成树保证连通
        - 然后补边直到所有节点度数 ≥ min_degree（或达到尝试上限）
        """
        params = self.config["mesh"]
        n = int(params["node_count"])
        d_min = int(params["min_degree"])

        if d_min >= n:
            raise ValueError(f"min_degree ({d_min}) 必须小于节点数 ({n})")

        G = nx.Graph()
        G.add_nodes_from(range(n))

        # 1) 随机生成树
        nodes = list(range(n))
        random.shuffle(nodes)
        for i in range(1, n):
            u = nodes[i]
            v = random.choice(nodes[:i])
            G.add_edge(u, v)

        # 2) 补边：保证度数 ≥ d_min
        max_attempts = n * n * 10
        attempts = 0

        def all_ok():
            return all(G.degree(i) >= d_min for i in G.nodes())

        while not all_ok() and attempts < max_attempts:
            attempts += 1
            # 找出度数不足的节点
            low_deg_nodes = [i for i in G.nodes() if G.degree(i) < d_min]
            if not low_deg_nodes:
                break
            u = random.choice(low_deg_nodes)

            # 随机找一个可以连的节点 v
            candidates = [v for v in G.nodes() if v != u and not G.has_edge(u, v)]
            if not candidates:
                continue
            v = random.choice(candidates)
            G.add_edge(u, v)

        if not all_ok():
            print(f"警告：达到最大尝试次数，仍有节点度数 < {d_min}")

        name = f"mesh_N{n}_d{d_min}"
        pos = nx.spring_layout(G, seed=42)
        return G, name, pos


# ==========================
# 主函数
# ==========================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, CONFIG["output_dir"])
    ensure_dir(output_dir)

    gen = TopologyGenerator(CONFIG)
    G, adjacency, topo_name = gen.generate()

    print(f"生成拓扑: {topo_name}")
    print(f"节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")

    # 保存邻接矩阵
    adj_path = os.path.join(output_dir, f"{topo_name}_adjacency.txt")
    save_adjacency_matrix(adjacency, adj_path)
    print(f"邻接矩阵已保存到: {adj_path}")

    # 保存拓扑图
    img_path = os.path.join(output_dir, f"{topo_name}_topology.png")

    # 对 fat-tree / CLOS 使用自带 pos，其它用 spring_layout
    topology_type = CONFIG["topology_type"].lower()
    if topology_type in ("fat_tree", "clos"):
        # 上面生成函数已经返回了 pos，这里为了简单再算一次布局也可，
        # 但我们直接用 spring_layout 也没问题。
        pos = None
    else:
        pos = None

    draw_graph(G, img_path, pos=pos, title=topo_name)
    print(f"拓扑图已保存到: {img_path}")


if __name__ == "__main__":
    main()
