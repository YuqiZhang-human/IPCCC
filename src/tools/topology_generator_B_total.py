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
from typing import List


# ==========================
# 配置区：在这里改参数
# ==========================

CONFIG = {
    # 拓扑类型: "fat_tree" | "clos" | "random" | "mesh"
    "topology_type": "clos",

    # 输出目录（相对于本脚本所在目录）
    "output_dir": "topology_output",

    # 随机种子（方便复现）
    "random_seed": 42,

    # ---- fat-tree 参数 ----
    "fat_tree": {
        "x_hosts_per_leaf": 3,     # 每个叶子交换机下挂的 host 数量
        "y_switch_layers": 3,      # 交换机层数（不含 host 层）
        "branching_factor": 2      # 每下一层交换机数量 = 上一层 * branching_factor
    },

    # ---- CLOS 参数 ----
    "clos": {
        # 每层的节点数，例如 [4, 8, 8, 4]
        "layer_sizes": [4, 4, 4, 4, 4, 4]
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


def save_adjacency_matrix(adjacency: List[List[int]], filepath: str) -> None:
    """将邻接矩阵保存为 Python 列表字面量（list of lists）"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("[\n")
        for i, row in enumerate(adjacency):
            row_str = ", ".join(str(x) for x in row)
            comma = "," if i != len(adjacency) - 1 else ""
            f.write(f"    [{row_str}]{comma}\n")
        f.write("]\n")


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
    def generate(self) -> Tuple[nx.Graph, List[List[int]], str, Dict[int, Tuple[float, float]], Dict[str, any]]:
        """
        返回：
        - G: networkx 图
        - adjacency: 0/1 邻接矩阵（list[list[int]]）
        - name: 拓扑名
        - pos: 绘图坐标（fat-tree/clos 为分层坐标；random/mesh 可为 spring）
        - topo_meta: 拓扑元信息（用于分层带宽分配等）
            {
              "topology_type": str,
              "layers": List[List[int]],
              "edge_tiers": Dict[str, List[Tuple[int,int]]]
            }
        """
        ttype = self.config.get("topology_type", "fat_tree").lower()

        if ttype == "fat_tree":
            G, name, pos, topo_meta = self._generate_fat_tree()
        elif ttype == "clos":
            G, name, pos, topo_meta = self._generate_clos()
        elif ttype == "random":
            G, name, pos = self._generate_random()
            topo_meta = {"topology_type": "random", "layers": [], "edge_tiers": {}}
        elif ttype == "mesh":
            G, name, pos = self._generate_mesh()
            topo_meta = {"topology_type": "mesh", "layers": [], "edge_tiers": {}}
        else:
            raise ValueError(f"未知拓扑类型: {ttype}")

        adjacency = graph_to_adjacency_matrix(G)
        return G, adjacency, name, pos, topo_meta


    # ---- 1. fat-tree ----
    def _generate_fat_tree(self) -> Tuple[nx.Graph, str, Dict[int, Tuple[float, float]], Dict[str, any]]:
        """
        简化 fat-tree 风格拓扑（你现有实现）：
        - y 层交换机（0:核心层 ... y-1:叶子层）
        - 层间全连接
        - 每个叶子层节点接 x_hosts_per_leaf 个 host

        这里新增：
        - layers: [switch_layer0, ..., switch_layer(y-1), host_layer]
        - edge_tiers:
            tier_sw_0_1, tier_sw_1_2, ..., tier_sw_(y-2)_(y-1), tier_leaf_host
        """
        params = self.config["fat_tree"]
        x_hosts_per_leaf = int(params["x_hosts_per_leaf"])
        y_layers = int(params["y_switch_layers"])
        branching_factor = int(params["branching_factor"])

        G = nx.Graph()
        switch_layers: List[List[int]] = []
        edge_tiers: Dict[str, List[Tuple[int, int]]] = {}

        current_node_id = 0
        num_switches = 1

        # 生成交换机层
        for layer in range(y_layers):
            layer_nodes = list(range(current_node_id, current_node_id + num_switches))
            switch_layers.append(layer_nodes)
            G.add_nodes_from(layer_nodes)
            current_node_id += num_switches
            num_switches *= branching_factor

        # 层间全连接，并记录 tier
        for layer in range(y_layers - 1):
            tier_name = f"tier_sw_{layer}_{layer+1}"
            edge_tiers[tier_name] = []
            for u in switch_layers[layer]:
                for v in switch_layers[layer + 1]:
                    G.add_edge(u, v)
                    a, b = (u, v) if u < v else (v, u)
                    edge_tiers[tier_name].append((a, b))

        # 生成 host，并连接叶子层
        host_nodes: List[int] = []
        edge_tiers["tier_leaf_host"] = []

        leaf_switches = switch_layers[-1]
        for leaf in leaf_switches:
            for _ in range(x_hosts_per_leaf):
                host_id = current_node_id
                current_node_id += 1
                host_nodes.append(host_id)
                G.add_node(host_id)
                G.add_edge(leaf, host_id)

                a, b = (leaf, host_id) if leaf < host_id else (host_id, leaf)
                edge_tiers["tier_leaf_host"].append((a, b))

        # 分层坐标（你原本的分层 pos 逻辑）
        pos: Dict[int, Tuple[float, float]] = {}
        y_gap = 1.5
        for layer_idx, layer_nodes in enumerate(switch_layers):
            y = (y_layers - 1 - layer_idx) * y_gap
            x_gap = 1.0
            for j, node in enumerate(layer_nodes):
                pos[node] = (j * x_gap, y)

        # host 放在最底层
        host_y = -1.0 * y_gap
        for j, node in enumerate(host_nodes):
            pos[node] = (j * 0.4, host_y)

        name = f"fat_tree_x{x_hosts_per_leaf}_y{y_layers}_bf{branching_factor}"

        topo_meta = {
            "topology_type": "fat_tree",
            "layers": switch_layers + [host_nodes],
            "edge_tiers": edge_tiers,
        }
        return G, name, pos, topo_meta

    # ---- 2. CLOS 架构 ----
    def _generate_clos(self) -> Tuple[nx.Graph, str, Dict[int, Tuple[float, float]], Dict[str, any]]:
        """
        CLOS 架构：
        - 多层，每层节点数由 layer_sizes 指定
        - 相邻层全连接

        新增：
        - layers: 每层节点列表
        - edge_tiers: tier_0_1, tier_1_2, ...
        """
        layer_sizes = self.config["clos"]["layer_sizes"]
        layer_sizes = [int(x) for x in layer_sizes]
        L = len(layer_sizes)

        G = nx.Graph()
        layers: List[List[int]] = []
        edge_tiers: Dict[str, List[Tuple[int, int]]] = {}

        current_node_id = 0
        for i, sz in enumerate(layer_sizes):
            layer_nodes = list(range(current_node_id, current_node_id + sz))
            layers.append(layer_nodes)
            G.add_nodes_from(layer_nodes)
            current_node_id += sz

        # 相邻层全连接，并记录 tier
        for i in range(L - 1):
            tier_name = f"tier_{i}_{i+1}"
            edge_tiers[tier_name] = []
            for u in layers[i]:
                for v in layers[i + 1]:
                    G.add_edge(u, v)
                    a, b = (u, v) if u < v else (v, u)
                    edge_tiers[tier_name].append((a, b))

        # 分层坐标
        pos: Dict[int, Tuple[float, float]] = {}
        y_gap = 1.5
        for i, layer_nodes in enumerate(layers):
            y = (L - 1 - i) * y_gap
            x_gap = 1.0
            for j, node in enumerate(layer_nodes):
                pos[node] = (j * x_gap, y)

        name = "clos_" + "_".join(map(str, layer_sizes))

        topo_meta = {
            "topology_type": "clos",
            "layers": layers,
            "edge_tiers": edge_tiers,
        }
        return G, name, pos, topo_meta

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

# --- main() ---
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, CONFIG["output_dir"])
    ensure_dir(output_dir)

    gen = TopologyGenerator(CONFIG)
    G, adjacency, topo_name, pos, topo_meta = gen.generate()

    print(f"生成拓扑: {topo_name}")
    print(f"节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")

    adj_path = os.path.join(output_dir, f"{topo_name}_adjacency.txt")
    save_adjacency_matrix(adjacency, adj_path)
    print(f"邻接矩阵已保存到: {adj_path}")

    img_path = os.path.join(output_dir, f"{topo_name}_topology.png")

    topology_type = CONFIG["topology_type"].lower()
    if topology_type in ("random", "mesh"):
        pos = None  # 保持 random/mesh 为普通图（spring）

    draw_graph(G, img_path, pos=pos, title=topo_name)
    print(f"拓扑图已保存到: {img_path}")



if __name__ == "__main__":
    main()
