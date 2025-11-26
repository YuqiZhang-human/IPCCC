# shortest_path_all_node.py
# -*- coding: utf-8 -*-
"""
基于分层图最短路的单用户成本最小化部署算法（多链部署版本）

目标：
- 在给定的 test_case（预处理后的测试数据）下，
  对单条推理链（模块序列）构建分层图；
- 使用 DP / 最短路找到「单人成本」最小的可行路径：
    - 这里的「可行」是指：
        在当前剩余资源下，max_users > 0；
    - 单人成本 = 部署成本(显存+算力) + 通信成本(链路带宽)；
- 在此基础上，执行多链部署：
    - 每次部署一条最优链，扣除资源；
    - 循环，直到无法再部署一条完整的新链。

注意：
- 这里不考虑“更贵但资源更均衡的路径”，
  即如果 DP 在某个 (layer, node) 只保留了一条最低成本路径，
  这条路径不可行但另一条更贵路径可能可行，这种情况被忽略。
  这是你明确表示可以接受的简化。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math
import copy


@dataclass
class PathResult:
    """表示单条链路部署方案（针对单用户）的结果。"""
    per_user_deploy_cost: float
    per_user_comm_cost: float
    per_user_total_cost: float
    max_users: int
    deployment: List[int]  # 长度 = module_count，每个元素是节点 index


class ShortestPathOptimizer:
    def __init__(self, test_case: Dict[str, Any]):
        """
        test_case 由外部预处理得到，字段约定：

        - "test_data_id": 测试数据 ID（可选）
        - "node_count": 节点数量 N
        - "module_count": 模块数量 M
        - "computation_capacity": List[[compute_cap, mem_cap], ...]，节点资源容量
        - "resource_demands": List[[compute_demand_per_user, mem_demand_per_user], ...]
        - "data_sizes": List[boundary_data_mb_per_sec_per_user]，长度 M-1
        - "bandwidth_matrix": N×N 矩阵，元素为 MB/s
        - "distance_matrix": N×N 矩阵（可选），不提供则默认 hop=1
        - "node_costs": List[[gpu_cost, memory_cost], ...]
        - "bandwidth_cost": 链路带宽单价，$/ (MB/s * month) 或等效
        - "profit_per_user": 每用户收益，$/month
        """
        self.test_case = test_case
        self.test_data_id: int = int(test_case.get("test_data_id", -1))

        # 基本规模
        self.node_count: int = int(test_case["node_count"])
        self.module_count: int = int(test_case["module_count"])

        # 资源能力（初始）
        # computation_capacity[i] = [compute_cap_i, mem_cap_i]
        self.initial_computation_capacity: List[List[float]] = [
            [float(c[0]), float(c[1])] for c in test_case["computation_capacity"]
        ]

        # 当前剩余资源（会随着多链部署被扣减）
        self.remaining_computation_capacity: List[List[float]] = copy.deepcopy(
            self.initial_computation_capacity
        )

        # 带宽矩阵（MB/s）
        self.initial_bandwidth_matrix: List[List[float]] = [
            [float(x) for x in row] for row in test_case["bandwidth_matrix"]
        ]
        self.remaining_bandwidth_matrix: List[List[float]] = copy.deepcopy(
            self.initial_bandwidth_matrix
        )

        # 距离矩阵（若未提供，则默认距离为 1）
        if "distance_matrix" in test_case and test_case["distance_matrix"] is not None:
            self.distance_matrix: List[List[float]] = [
                [float(x) for x in row] for row in test_case["distance_matrix"]
            ]
        else:
            self.distance_matrix = [
                [0.0 if i == j else 1.0 for j in range(self.node_count)]
                for i in range(self.node_count)
            ]

        # 每个模块的 per-user 需求
        # resource_demands[i] = [compute_demand, mem_demand]
        self.resource_demands: List[List[float]] = [
            [float(d[0]), float(d[1])] for d in test_case["resource_demands"]
        ]

        # 边界数据量（MB/s per user）
        self.data_sizes: List[float] = [float(x) for x in test_case.get("data_sizes", [])]

        # 节点成本（gpu_cost, mem_cost）
        # node_costs[i] = [gpu_cost_i, mem_cost_i]
        node_costs_raw = test_case.get("node_costs", [])
        self.node_gpu_costs: List[float] = []
        self.node_mem_costs: List[float] = []
        for c in node_costs_raw:
            # 若传进来的是 dict 也兼容一下
            if isinstance(c, dict):
                self.node_gpu_costs.append(float(c.get("gpu_cost", 0.0)))
                self.node_mem_costs.append(float(c.get("memory_cost", 0.0)))
            else:
                self.node_gpu_costs.append(float(c[0]))
                self.node_mem_costs.append(float(c[1]))

        # 带宽单价、收益
        self.bandwidth_cost: float = float(test_case.get("bandwidth_cost", 0.0))
        self.profit_per_user: float = float(test_case.get("profit_per_user", 0.0))

        # 调试开关（必要时可以人为打开）
        self.debug: bool = False

    # ================================================================
    # 对外主入口：多链部署
    # ================================================================
    def shortest_path_deployment(self) -> Optional[Dict[str, Any]]:
        """
        多链部署：
        - 使用当前剩余资源，找一条单人成本最小的「可行链」；
        - 扣除资源，累加成本和用户数；
        - 循环，直到不存在任何可行链。
        """
        # 每次调用从初始资源开始
        self.remaining_computation_capacity = copy.deepcopy(self.initial_computation_capacity)
        self.remaining_bandwidth_matrix = copy.deepcopy(self.initial_bandwidth_matrix)

        total_cost = 0.0
        total_deploy_cost = 0.0
        total_comm_cost = 0.0
        total_profit = 0.0
        total_users = 0

        used_nodes_flags = [0] * self.node_count
        modules_assigned_count = [0] * self.node_count
        chain_count = 0

        while True:
            path_res = self._find_best_feasible_chain()
            if path_res is None:
                # 没有任何可行链，结束
                break

            max_users = path_res.max_users
            if max_users <= 0:
                # 虽然找到了路径，但连一个用户都不支持
                break

            per_user_dep = path_res.per_user_deploy_cost
            per_user_comm = path_res.per_user_comm_cost
            per_user_total = path_res.per_user_total_cost
            deployment = path_res.deployment

            # 累加成本
            chain_users = max_users
            chain_deploy_cost = per_user_dep * chain_users
            chain_comm_cost = per_user_comm * chain_users
            chain_cost = per_user_total * chain_users
            chain_profit = self.profit_per_user * chain_users - chain_cost

            total_deploy_cost += chain_deploy_cost
            total_comm_cost += chain_comm_cost
            total_cost += chain_cost
            total_profit += chain_profit
            total_users += chain_users
            chain_count += 1

            # 资源扣减 + 使用统计
            self._apply_chain_consumption(deployment, chain_users)
            for m, node in enumerate(deployment):
                used_nodes_flags[node] = 1
                modules_assigned_count[node] += 1

        if chain_count == 0:
            return None

        used_nodes_count = sum(used_nodes_flags)
        if used_nodes_count > 0:
            avg_modules_per_node = sum(modules_assigned_count) / float(used_nodes_count)
        else:
            avg_modules_per_node = 0.0

        return {
            "total_cost": total_cost,
            "total_deploy_cost": total_deploy_cost,
            "total_comm_cost": total_comm_cost,
            "total_profit": total_profit,
            "total_users": total_users,
            "used_nodes": used_nodes_count,
            "avg_modules_per_node": avg_modules_per_node,
            "chain_count": chain_count,
        }

    # ================================================================
    # 单链：在当前剩余资源下，找到「可行路径中单人成本最小」的部署方案
    # ================================================================
    def _find_best_feasible_chain(self) -> Optional[PathResult]:
        N = self.node_count
        M = self.module_count
        if N <= 0 or M <= 0:
            return None

        INF = 1e30

        # dp[layer][node] = 走到第 layer 个模块部署在 node 上的最小单人成本
        dp = [[INF] * N for _ in range(M)]
        prev_node = [[-1] * N for _ in range(M)]

        # ---------------------------
        # 初始化第一层（模块 0）
        # ---------------------------
        comp_demand_0, mem_demand_0 = self._get_module_demand(0)
        for node in range(N):
            comp_cap, mem_cap = self._get_node_capacity(node)
            if comp_demand_0 <= comp_cap and mem_demand_0 <= mem_cap:
                dep_cost = self._deploy_cost_per_user(module_idx=0, node_idx=node)
                dp[0][node] = dep_cost
                prev_node[0][node] = -1  # 源头

        # 如果第一层就完全不可行，直接结束
        if all(c >= INF for c in dp[0]):
            return None

        # ---------------------------
        # 自底向上 DP
        # ---------------------------
        for layer in range(1, M):
            comp_demand, mem_demand = self._get_module_demand(layer)
            boundary_idx = layer - 1  # 与上一层之间的 boundary index
            boundary_data = self._get_data_size(boundary_idx)

            for cur_node in range(N):
                # 先检查当前模块能否单用户放在 cur_node 上
                comp_cap_cur, mem_cap_cur = self._get_node_capacity(cur_node)
                if comp_demand > comp_cap_cur or mem_demand > mem_cap_cur:
                    continue

                dep_cost_cur = self._deploy_cost_per_user(module_idx=layer, node_idx=cur_node)

                best_cost_for_cur = INF
                best_prev_node = -1

                for prev in range(N):
                    if dp[layer - 1][prev] >= INF:
                        continue

                    # 若跨节点，需要链路
                    comm_cost = 0.0
                    if prev != cur_node and boundary_data > 0:
                        bw = self._get_link_bandwidth(prev, cur_node)
                        if bw <= 0:
                            # 无带宽，无法连接
                            continue
                        dist = self.distance_matrix[prev][cur_node]
                        comm_cost = boundary_data * self.bandwidth_cost * dist

                    new_cost = dp[layer - 1][prev] + dep_cost_cur + comm_cost
                    if new_cost < best_cost_for_cur:
                        best_cost_for_cur = new_cost
                        best_prev_node = prev

                if best_prev_node != -1:
                    dp[layer][cur_node] = best_cost_for_cur
                    prev_node[layer][cur_node] = best_prev_node

        # ---------------------------
        # 在所有可能终点中，筛选「可行且单人成本最小」的路径
        # ---------------------------
        best_path: Optional[PathResult] = None

        last_layer = M - 1
        for last_node in range(N):
            if dp[last_layer][last_node] >= INF:
                continue

            # 回溯出完整部署路径
            deployment = [0] * M
            cur_node = last_node
            for layer in range(last_layer, -1, -1):
                deployment[layer] = cur_node
                cur_node = prev_node[layer][cur_node]

            # 计算该路径下最大可支持用户数
            max_users = self._calculate_max_users_for_deployment(deployment)
            if max_users <= 0:
                continue  # 不可行，跳过

            # 计算 per-user 部署成本和通信成本
            dep_cost_per_user = self._deploy_cost_for_deployment_per_user(deployment)
            comm_cost_per_user = self._comm_cost_for_deployment_per_user(deployment)
            total_cost_per_user = dep_cost_per_user + comm_cost_per_user

            if best_path is None or total_cost_per_user < best_path.per_user_total_cost:
                best_path = PathResult(
                    per_user_deploy_cost=dep_cost_per_user,
                    per_user_comm_cost=comm_cost_per_user,
                    per_user_total_cost=total_cost_per_user,
                    max_users=max_users,
                    deployment=deployment,
                )

        return best_path

    # ================================================================
    # 资源扣减：根据某条链和其用户数，扣除节点和链路资源
    # ================================================================
    def _apply_chain_consumption(self, deployment: List[int], users: int) -> None:
        """
        部署一条链后，扣减剩余算力 / 显存 / 带宽。
        """
        N = self.node_count
        M = self.module_count

        # 节点资源扣减
        node_comp_usage = [0.0] * N
        node_mem_usage = [0.0] * N
        for m_idx, node in enumerate(deployment):
            comp_d, mem_d = self._get_module_demand(m_idx)
            node_comp_usage[node] += comp_d * users
            node_mem_usage[node] += mem_d * users

        for node in range(N):
            comp_cap, mem_cap = self.remaining_computation_capacity[node]
            comp_cap -= node_comp_usage[node]
            mem_cap -= node_mem_usage[node]
            # 不做特别 clamp，只要逻辑上 max_users 的计算保证不会越界
            self.remaining_computation_capacity[node] = [comp_cap, mem_cap]

        # 链路带宽扣减
        for boundary_idx in range(M - 1):
            from_node = deployment[boundary_idx]
            to_node = deployment[boundary_idx + 1]
            if from_node == to_node:
                continue

            data_size = self._get_data_size(boundary_idx)
            if data_size <= 0:
                continue

            traffic = data_size * users  # MB/s * users

            bw_ft = self.remaining_bandwidth_matrix[from_node][to_node]
            bw_tf = self.remaining_bandwidth_matrix[to_node][from_node]

            bw_ft -= traffic
            bw_tf -= traffic

            self.remaining_bandwidth_matrix[from_node][to_node] = bw_ft
            self.remaining_bandwidth_matrix[to_node][from_node] = bw_tf

    # ================================================================
    # 工具函数：单用户部署成本 / 通信成本 / 资源限制
    # ================================================================
    def _get_node_capacity(self, node_idx: int) -> Tuple[float, float]:
        """
        返回当前剩余资源：
        - compute_cap: 节点可用算力
        - mem_cap: 节点可用显存（GB）
        """
        comp_cap, mem_cap = self.remaining_computation_capacity[node_idx]
        return float(comp_cap), float(mem_cap)

    def _get_module_demand(self, module_idx: int) -> Tuple[float, float]:
        """
        返回模块的 per-user 需求：
        - compute_demand
        - mem_demand
        """
        comp_d, mem_d = self.resource_demands[module_idx]
        return float(comp_d), float(mem_d)

    def _get_data_size(self, boundary_idx: int) -> float:
        if 0 <= boundary_idx < len(self.data_sizes):
            return float(self.data_sizes[boundary_idx])
        return 0.0

    def _get_link_bandwidth(self, from_node: int, to_node: int) -> float:
        return float(self.remaining_bandwidth_matrix[from_node][to_node])

    def _deploy_cost_per_user(self, module_idx: int, node_idx: int) -> float:
        """
        单用户在 node_idx 上部署 module_idx 的成本：
        mem_demand * mem_cost + compute_demand * gpu_cost
        """
        comp_d, mem_d = self._get_module_demand(module_idx)
        gpu_cost = self.node_gpu_costs[node_idx]
        mem_cost = self.node_mem_costs[node_idx]
        return comp_d * gpu_cost + mem_d * mem_cost

    def _deploy_cost_for_deployment_per_user(self, deployment: List[int]) -> float:
        total = 0.0
        for m_idx, node in enumerate(deployment):
            total += self._deploy_cost_per_user(m_idx, node)
        return total

    def _comm_cost_for_deployment_per_user(self, deployment: List[int]) -> float:
        total = 0.0
        M = self.module_count
        for boundary_idx in range(M - 1):
            from_node = deployment[boundary_idx]
            to_node = deployment[boundary_idx + 1]
            if from_node == to_node:
                continue
            data_size = self._get_data_size(boundary_idx)
            if data_size <= 0:
                continue
            dist = self.distance_matrix[from_node][to_node]
            total += data_size * self.bandwidth_cost * dist
        return total

    def _calculate_max_users_for_deployment(self, deployment: List[int]) -> int:
        """
        在当前剩余资源下，计算指定部署方案可支持的最大用户数：
        - 节点算力 / 显存约束
        - 链路带宽约束
        """
        N = self.node_count
        M = self.module_count

        if len(deployment) != M:
            return 0

        # 节点资源使用统计（per user）
        node_comp_usage = [0.0] * N
        node_mem_usage = [0.0] * N
        for m_idx, node in enumerate(deployment):
            comp_d, mem_d = self._get_module_demand(m_idx)
            node_comp_usage[node] += comp_d
            node_mem_usage[node] += mem_d

        node_limits: List[int] = []

        # 节点算力 / 显存限制
        for node in range(N):
            comp_cap, mem_cap = self._get_node_capacity(node)
            comp_use = node_comp_usage[node]
            mem_use = node_mem_usage[node]

            if comp_use <= 0 and mem_use <= 0:
                continue

            if comp_use > 0 and comp_cap > 0:
                users_by_compute = comp_cap / comp_use
            else:
                users_by_compute = float("inf")

            if mem_use > 0 and mem_cap > 0:
                users_by_memory = mem_cap / mem_use
            else:
                users_by_memory = float("inf")

            node_limit = min(users_by_compute, users_by_memory)
            if node_limit <= 0:
                return 0

            node_limits.append(int(math.floor(node_limit)))

        # 链路带宽限制
        link_limits: List[int] = []
        for boundary_idx in range(M - 1):
            from_node = deployment[boundary_idx]
            to_node = deployment[boundary_idx + 1]
            if from_node == to_node:
                continue

            bandwidth = self._get_link_bandwidth(from_node, to_node)
            if bandwidth <= 0:
                return 0

            data_size = self._get_data_size(boundary_idx)
            if data_size <= 0:
                continue

            users_by_bandwidth = bandwidth / data_size
            if users_by_bandwidth <= 0:
                return 0

            link_limits.append(int(math.floor(users_by_bandwidth)))

        all_limits = node_limits + link_limits
        if not all_limits:
            return 0

        max_users = max(0, min(all_limits))
        return max_users
