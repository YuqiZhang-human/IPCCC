#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于分层图最短路的单用户成本最小化多链部署算法

特性：
- 单条链内部：允许一个节点部署多个模块（multi-function）
- 多条链之间：允许复用节点，但每次都会根据剩余算力 / 显存 / 带宽重新计算最大用户数
- 单用户成本建模：
    * 每层 i 决策：模块 i 部署在哪个物理节点 y
    * a_{i-1,x} -> a_{i,y} 的边权：
        - 部署成本（显存 + 算力）/用户
        - 若 x != y，则叠加链路带宽成本 /用户
- 多链部署：
    * 在当前剩余资源下，找到“单用户总成本最小”的一条路径（链）
    * 计算这条链可支持的最大用户数 max_users（受算力 / 显存 / 带宽限制）
    * 扣减资源（remaining_computation_capacity / remaining_bandwidth_matrix）
    * 累计成本 / 利润 / 用户数，直到无法再部署新的链

返回：
- shortest_path_deployment() 返回 8 元组：
    (total_cost, total_deploy_cost, total_comm_cost,
     total_profit, total_users, used_nodes_count,
     avg_modules_per_node, chain_count)
"""

import json
import math
import copy
from typing import Any, Dict, List, Optional, Tuple


class ShortestPathOptimizer:
    """基于分层图最短路的部署优化器（多链 + 节点可复用）"""

    def __init__(self, test_data: Dict[str, Any]) -> None:
        """
        Args:
            test_data: 由 test_experiment1.py 构造的字典，主要字段包括：
                - node_count, module_count
                - computation_capacity: n×2 [[compute_TFLOPs/s, memory_GB], ...]
                - resource_demands: m×2 [[compute_TFLOPs/s per user, memory_GB per user], ...]
                - data_sizes: 长度 m-1，单用户边界数据速率 (MB/s)
                - bandwidth_matrix: n×n，链路带宽 (MB/s)
                - gpu_cost, memory_cost, bandwidth_cost
                - profit_per_user
                - node_costs: n×2，每节点 [gpu_cost_node, mem_cost_node]
                - distance_matrix: n×n，节点间距离（跳数，用于通信成本）
        """
        self.test_data_id = test_data.get("test_data_id", 0)

        self.node_count = int(test_data["node_count"])
        self.module_count = int(test_data["module_count"])

        self.computation_capacity = self._parse_array(test_data["computation_capacity"])
        self.resource_demands = self._parse_array(test_data["resource_demands"])
        self.data_sizes = self._parse_array(test_data["data_sizes"])
        self.bandwidth_matrix = self._parse_array(test_data["bandwidth_matrix"])

        self.gpu_cost = float(test_data.get("gpu_cost", 0.0))
        self.memory_cost = float(test_data.get("memory_cost", 0.0))
        self.bandwidth_cost = float(test_data.get("bandwidth_cost", 0.0))
        self.profit_per_user = float(test_data.get("profit_per_user", 0.0))

        node_costs = test_data.get("node_costs")
        distance_matrix = test_data.get("distance_matrix")

        self.node_costs = self._parse_array(node_costs) if node_costs is not None else None
        self.distance_matrix = self._parse_array(distance_matrix) if distance_matrix is not None else None

        # 若未提供 per-node 成本，则使用全局成本参数
        if self.node_costs is None:
            self.node_costs = [[self.gpu_cost, self.memory_cost] for _ in range(self.node_count)]

        # 若未提供距离矩阵，则默认距离为 1（对角线为 0）
        if self.distance_matrix is None:
            self.distance_matrix = [
                [0 if i == j else 1 for j in range(self.node_count)]
                for i in range(self.node_count)
            ]

        # 备份初始资源，用于多链部署迭代时重置
        self.initial_computation_capacity = copy.deepcopy(self.computation_capacity)
        self.initial_bandwidth_matrix = copy.deepcopy(self.bandwidth_matrix)

        # 当前剩余资源
        self.remaining_computation_capacity = copy.deepcopy(self.computation_capacity)
        self.remaining_bandwidth_matrix = copy.deepcopy(self.bandwidth_matrix)

        # 可选：记录所有单链方案
        self.all_solutions: List[Tuple] = []

    # ----------------------------------------------------------------------
    # 通用工具函数
    # ----------------------------------------------------------------------
    def _parse_array(self, value):
        """将可能是 JSON 字符串的字段解析为 Python 列表"""
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                return json.loads(text)
            except Exception:
                try:
                    return eval(text)
                except Exception:
                    raise ValueError(f"无法解析数组字段: {value}")
        return value

    def get_node_capacity(self, node: int) -> Tuple[float, float]:
        """获取节点当前剩余的 [算力, 显存] 容量"""
        if node < 0 or node >= self.node_count:
            return 0.0, 0.0
        return (
            float(self.remaining_computation_capacity[node][0]),
            float(self.remaining_computation_capacity[node][1]),
        )

    def get_module_demand(self, module: int) -> Tuple[float, float]:
        """获取单个模块的单用户 [算力需求, 显存需求]"""
        if module < 0 or module >= self.module_count:
            return 0.0, 0.0
        return (
            float(self.resource_demands[module][0]),
            float(self.resource_demands[module][1]),
        )

    def get_link_bandwidth(self, from_node: int, to_node: int) -> float:
        """获取两个节点之间的当前剩余带宽 (MB/s)"""
        if (
            from_node < 0 or from_node >= self.node_count
            or to_node < 0 or to_node >= self.node_count
        ):
            return 0.0
        return float(self.remaining_bandwidth_matrix[from_node][to_node])

    def get_link_distance(self, from_node: int, to_node: int) -> float:
        """获取两个节点之间的距离（跳数）"""
        if (
            from_node < 0 or from_node >= self.node_count
            or to_node < 0 or to_node >= self.node_count
        ):
            return 1.0
        return float(self.distance_matrix[from_node][to_node])

    def get_data_size(self, boundary_index: int) -> float:
        """获取边界 boundary_index 上单用户数据速率 (MB/s)"""
        if boundary_index < 0 or boundary_index >= len(self.data_sizes):
            return 0.0
        return float(self.data_sizes[boundary_index])

    # ----------------------------------------------------------------------
    # 资源极限与成本计算（与 MultiFunctionOptimizer 保持一致）
    # ----------------------------------------------------------------------
    def calculate_max_users_for_deployment(self, deployment: List[int]) -> int:
        """
        在当前剩余资源下，计算给定部署方案能支持的最大用户数。

        限制：
        - 节点算力 / 显存
        - 链路带宽
        """
        n = self.node_count
        m = self.module_count

        if len(deployment) != m:
            return 0

        # 节点 per-user 资源使用
        node_compute_usage = [0.0] * n
        node_memory_usage = [0.0] * n

        for module_idx, node in enumerate(deployment):
            comp_demand, mem_demand = self.get_module_demand(module_idx)
            node_compute_usage[node] += comp_demand
            node_memory_usage[node] += mem_demand

        node_limits: List[int] = []

        # 节点算力 / 显存限制
        for node in range(n):
            comp_cap, mem_cap = self.get_node_capacity(node)
            comp_use = node_compute_usage[node]
            mem_use = node_memory_usage[node]

            if comp_use <= 0 and mem_use <= 0:
                continue

            # 算力限制
            if comp_use > 0 and comp_cap > 0:
                users_by_compute = comp_cap / comp_use
            else:
                users_by_compute = float("inf")

            # 显存限制
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

        for boundary_idx in range(m - 1):
            from_node = deployment[boundary_idx]
            to_node = deployment[boundary_idx + 1]

            if from_node == to_node:
                continue

            bandwidth = self.get_link_bandwidth(from_node, to_node)
            if bandwidth <= 0:
                return 0

            data_size = self.get_data_size(boundary_idx)
            if data_size <= 0:
                continue

            users_by_bandwidth = bandwidth / data_size
            if users_by_bandwidth <= 0:
                return 0

            link_limits.append(int(math.floor(users_by_bandwidth)))

        all_limits = node_limits + link_limits
        if not all_limits:
            return 0

        return max(0, min(all_limits))

    def calculate_costs_for_deployment(
        self, deployment: List[int], user_count: int
    ) -> Tuple[float, float, float, float]:
        """
        计算在当前视角下，该部署方案在 user_count 用户时的：
        - 总成本 total_cost
        - 部署成本 deploy_cost（算力+显存）
        - 通信成本 comm_cost
        - 利润 profit
        """
        n = self.node_count
        m = self.module_count

        if user_count <= 0 or len(deployment) != m:
            return 0.0, 0.0, 0.0, 0.0

        # per-user 节点资源使用
        node_compute_usage = [0.0] * n
        node_memory_usage = [0.0] * n

        for module_idx, node in enumerate(deployment):
            comp_demand, mem_demand = self.get_module_demand(module_idx)
            node_compute_usage[node] += comp_demand
            node_memory_usage[node] += mem_demand

        deploy_cost = 0.0
        for node in range(n):
            comp_use = node_compute_usage[node] * user_count
            mem_use = node_memory_usage[node] * user_count

            gpu_cost_node, mem_cost_node = self.node_costs[node]

            deploy_cost += comp_use * float(gpu_cost_node)
            deploy_cost += mem_use * float(mem_cost_node)

        comm_cost = 0.0
        for boundary_idx in range(m - 1):
            from_node = deployment[boundary_idx]
            to_node = deployment[boundary_idx + 1]
            if from_node == to_node:
                continue

            data_size = self.get_data_size(boundary_idx)
            if data_size <= 0:
                continue

            distance = self.get_link_distance(from_node, to_node)
            comm_cost += data_size * distance * self.bandwidth_cost * user_count

        total_cost = deploy_cost + comm_cost
        profit = self.profit_per_user * user_count - total_cost

        return total_cost, deploy_cost, comm_cost, profit

    def apply_chain_consumption(self, deployment: List[int], user_count: int) -> None:
        """
        将一条链的资源消耗从“剩余资源”中扣除（允许节点复用，只减少 capacity）
        """
        if user_count <= 0:
            return

        n = self.node_count
        m = self.module_count

        node_compute_usage = [0.0] * n
        node_memory_usage = [0.0] * n

        for module_idx, node in enumerate(deployment):
            comp_demand, mem_demand = self.get_module_demand(module_idx)
            node_compute_usage[node] += comp_demand
            node_memory_usage[node] += mem_demand

        # 扣减节点算力 / 显存
        for node in range(n):
            comp_cap, mem_cap = self.remaining_computation_capacity[node]
            comp_cap = float(comp_cap) - node_compute_usage[node] * user_count
            mem_cap = float(mem_cap) - node_memory_usage[node] * user_count
            if comp_cap < 0.0:
                comp_cap = 0.0
            if mem_cap < 0.0:
                mem_cap = 0.0
            self.remaining_computation_capacity[node][0] = comp_cap
            self.remaining_computation_capacity[node][1] = mem_cap

        # 扣减链路带宽
        for boundary_idx in range(m - 1):
            from_node = deployment[boundary_idx]
            to_node = deployment[boundary_idx + 1]
            if from_node == to_node:
                continue

            data_size = self.get_data_size(boundary_idx)
            if data_size <= 0:
                continue

            consume = data_size * user_count
            bw_ft = float(self.remaining_bandwidth_matrix[from_node][to_node]) - consume
            bw_tf = float(self.remaining_bandwidth_matrix[to_node][from_node]) - consume
            if bw_ft < 0.0:
                bw_ft = 0.0
            if bw_tf < 0.0:
                bw_tf = 0.0
            self.remaining_bandwidth_matrix[from_node][to_node] = bw_ft
            self.remaining_bandwidth_matrix[to_node][from_node] = bw_tf

    # ----------------------------------------------------------------------
    # 单条链：分层图最短路（单用户成本最小）
    # ----------------------------------------------------------------------
    def find_best_single_chain(
        self,
    ) -> Optional[Tuple[float, float, float, float, int, int, float, List[int]]]:
        """
        在当前剩余资源下，使用分层图最短路（DP）找出：
        - 单用户总成本最小的一条链（部署方案 + 节点序列）
        - 然后计算该链可支持的最大用户数和总体成本/利润

        返回:
            (cost, deploy_cost, comm_cost, profit,
             user_count, used_nodes_count, avg_modules_per_node, deployment)
        或 None（无可行路径）
        """
        n = self.node_count
        m = self.module_count

        if n <= 0 or m <= 0:
            return None

        # 预取 per-module demand，避免多次取值
        comp_demands = []
        mem_demands = []
        for i in range(m):
            c, mem = self.get_module_demand(i)
            comp_demands.append(c)
            mem_demands.append(mem)

        # DP 数组: dp[i][y] = 模块 0..i 部署完且第 i 个模块在节点 y 时的单用户最小总成本
        INF = float("inf")
        dp = [[INF] * n for _ in range(m)]
        prev_node = [[-1] * n for _ in range(m)]

        # 层 0 初始化：只考虑部署成本
        for y in range(n):
            comp_cap_y, mem_cap_y = self.get_node_capacity(y)
            c_d = comp_demands[0]
            m_d = mem_demands[0]

            # 至少保证这个节点对“单个用户”的单个模块是可行的（不然根本部署不了 1 个用户）
            if (c_d > 0 and comp_cap_y <= 0) or (m_d > 0 and mem_cap_y <= 0):
                continue

            gpu_cost_y, mem_cost_y = self.node_costs[y]
            deploy_cost_per_user = c_d * float(gpu_cost_y) + m_d * float(mem_cost_y)

            dp[0][y] = deploy_cost_per_user
            prev_node[0][y] = -1

        # 逐层 DP：i = 1..m-1
        for i in range(1, m):
            data_size_prev = self.get_data_size(i - 1)  # boundary i-1 -> i 的单用户数据量
            for y in range(n):
                comp_cap_y, mem_cap_y = self.get_node_capacity(y)
                c_d = comp_demands[i]
                m_d = mem_demands[i]

                # 同样要求该节点至少有能力承载该模块的单用户资源
                if (c_d > 0 and comp_cap_y <= 0) or (m_d > 0 and mem_cap_y <= 0):
                    continue

                gpu_cost_y, mem_cost_y = self.node_costs[y]
                deploy_cost_per_user = c_d * float(gpu_cost_y) + m_d * float(mem_cost_y)

                best_cost_for_y = INF
                best_prev = -1

                for x in range(n):
                    if dp[i - 1][x] >= INF:
                        continue

                    # 计算 x -> y 的链路成本（单用户）
                    link_cost = 0.0
                    if x != y and data_size_prev > 0:
                        bw_xy = self.get_link_bandwidth(x, y)
                        # 至少保证 1 个用户的通信是可行的（BW>=data_size_prev）
                        if bw_xy < data_size_prev:
                            continue
                        dist_xy = self.get_link_distance(x, y)
                        link_cost = data_size_prev * dist_xy * self.bandwidth_cost

                    cand_cost = dp[i - 1][x] + link_cost + deploy_cost_per_user
                    if cand_cost < best_cost_for_y:
                        best_cost_for_y = cand_cost
                        best_prev = x

                dp[i][y] = best_cost_for_y
                prev_node[i][y] = best_prev

        # 终层选择：选择 dp[m-1][y] 最小的 y
        last_layer = m - 1
        best_final_cost = INF
        best_last_node = -1

        for y in range(n):
            if dp[last_layer][y] < best_final_cost:
                best_final_cost = dp[last_layer][y]
                best_last_node = y

        if best_last_node == -1 or best_final_cost >= INF:
            return None

        # 回溯得到部署路径 deployment
        deployment = [0] * m
        cur_node = best_last_node
        for i in range(last_layer, -1, -1):
            deployment[i] = cur_node
            cur_node = prev_node[i][cur_node]

        # 在当前剩余资源下，计算这条路径可支持的最大用户数
        max_users = self.calculate_max_users_for_deployment(deployment)
        if max_users <= 0:
            return None

        # 计算部署成本 / 通信成本 / 利润
        total_cost, deploy_cost, comm_cost, profit = self.calculate_costs_for_deployment(
            deployment, max_users
        )
        used_nodes_count = len(set(deployment))
        avg_mods_per_node = m / used_nodes_count if used_nodes_count > 0 else 0.0

        return (
            total_cost,
            deploy_cost,
            comm_cost,
            profit,
            max_users,
            used_nodes_count,
            avg_mods_per_node,
            deployment,
        )

    # ----------------------------------------------------------------------
    # 多链循环部署（直到无法再部署一条完整链）
    # ----------------------------------------------------------------------
    def shortest_path_deployment(self):
        """
        从初始资源出发，循环执行：
        1. 用分层图最短路求出“单用户成本最小”的路径（链）
        2. 计算该链可支持的最大用户数
        3. 扣除对应资源占用
        4. 累计整体指标

        直到：再也找不到任何可行链（或该链最大用户数 <= 0）为止。

        返回 8 元组：
            (total_cost, total_deploy_cost, total_comm_cost,
             total_profit, total_users, used_nodes_count,
             avg_modules_per_node, chain_count)
        """
        # 重置剩余资源
        self.remaining_computation_capacity = copy.deepcopy(self.initial_computation_capacity)
        self.remaining_bandwidth_matrix = copy.deepcopy(self.initial_bandwidth_matrix)

        total_cost = 0.0
        total_deploy_cost = 0.0
        total_comm_cost = 0.0
        total_profit = 0.0
        total_users = 0
        used_nodes_union = set()
        chain_count = 0

        while True:
            best = self.find_best_single_chain()
            if best is None:
                break

            (
                cost,
                deploy_cost,
                comm_cost,
                profit,
                user_count,
                used_nodes_count,
                avg_mods_per_node,
                deployment,
            ) = best

            if user_count <= 0:
                break

            total_cost += cost
            total_deploy_cost += deploy_cost
            total_comm_cost += comm_cost
            total_profit += profit
            total_users += user_count
            used_nodes_union.update(deployment)
            chain_count += 1

            # 扣减资源（节点可复用，只减容量）
            self.apply_chain_consumption(deployment, user_count)

        if chain_count == 0:
            return None

        used_nodes_count = len(used_nodes_union)
        avg_modules_per_node = (
            self.module_count * chain_count / used_nodes_count
            if used_nodes_count > 0
            else 0.0
        )

        return (
            total_cost,
            total_deploy_cost,
            total_comm_cost,
            total_profit,
            total_users,
            used_nodes_count,
            avg_modules_per_node,
            chain_count,
        )
