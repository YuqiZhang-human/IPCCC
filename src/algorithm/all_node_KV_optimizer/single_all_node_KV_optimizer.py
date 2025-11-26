#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
单功能部署优化器（单链节点不重复 + 多链可复用 + 权重复用 + KV 不复用）

特性对齐说明：

1. 资源模型（真实化简版）：
   - resource_demands[module] = [compute_per_user, kv_memory_per_user_GB]
       * 第二个维度只表示 KV 显存，随用户数线性增长
   - weight_memory_demands[module] = 模块权重显存(GB)，一次加载，多链复用
   - 每个节点 j 的显存约束（在实现中体现在 remaining_computation_capacity / remaining_memory）：
       mem_used_j = sum_{已加载模块} weight_mem[m]
                    + sum_{已部署链 k} u_k * sum_{m 属于链 k 且在 j} kv_mem_per_user[m]
       <= mem_cap_j

   我们通过“扣减剩余显存 + 记录 modules_loaded_per_node”来隐式维护上式。

2. 单条链部署（single-function in-chain）：
   - 一条链 K 个模块 m_0,...,m_{K-1}，要求：
       * 同一条链内部，每个模块必须部署在不同节点上
       * 多条链之间，节点可以复用，只要资源足够
   - 部署策略：
       对当前模块 m_i：
           枚举所有可行节点 j（当前链中未用过 & 有剩余资源 & 链路可行）：
              计算局部指标：
                 users_local  : 看当前模块 + 与上一模块链路，在该节点上能支持多少用户
                 cost_local   : 本模块的增量成本（算力+KV+新增权重）+ 与前一模块之间通信
                 profit_local : users_local * profit_per_user - cost_local
           然后根据不同 objective 贪心选节点：
               - min_cost   : cost_local 最小
               - max_profit : profit_local 最大
               - min_profit : profit_local 最小（最差情况）
               - max_users  : users_local 最大
   - 全链完成后：
       使用权重复用 + KV 不复用的全链资源模型重新计算：
           max_users、total_cost、profit 等真实指标。

3. 多链部署（单策略）：
   - 对某个 objective：
       * 从初始资源状态开始（remaining_* ← initial_*，modules_loaded_per_node 清空）
       * while True:
           - 按该 objective 调用一次“单链贪心部署”
           - 若本链不可行 / user_count <= 0 → 停止
           - 否则：累计该链的 cost / profit / users，并从剩余资源中扣除本链消耗
       * 得到该策略下的 8 元组：
           (total_cost, total_deploy_cost, total_comm_cost,
            total_profit, total_users, used_nodes_count,
            avg_modules_per_node, chain_count)

4. 四种策略完全独立：
   - optimize_for_profit() 内部依次运行：
       - min_cost
       - max_profit
       - min_profit
       - max_users
     每次都从初始资源开始独立多链部署。

5. 兼容性：
   - 保留 single_func_deployment()：
       * 返回 max_profit 策略的 8 元组，兼容你原先的调用方式。
"""

import math
import copy
import json
from typing import Any, Dict, List, Optional, Tuple


class SingleFunctionOptimizer:
    """单功能部署优化器（单链节点不重复 + 多链可复用 + 权重复用 + KV 不复用）"""

    def __init__(self, test_data: Dict[str, Any]) -> None:
        """
        Args:
            test_data: 字典，包含以下键（由上游构造）：
                - node_count: 节点数量 n
                - module_count: 模块数量 m
                - computation_capacity: n×2，[[compute_TFLOPs_per_s, memory_GB], ...]
                - resource_demands: m×2，[[compute_per_user, kv_mem_GB_per_user], ...]
                - weight_memory_demands: 长度为 m，模块权重显存(GB)，可选，没有则视为 0
                - data_sizes: 长度 m-1，每个边界的单用户数据速率 (MB/s per user)
                - bandwidth_matrix: n×n，链路带宽 (MB/s)
                - gpu_cost: GPU 成本系数（通常设为 0，算力仅作为约束）
                - memory_cost: 显存成本系数（$/GB/month）
                - bandwidth_cost: 带宽成本系数（$/ (MB/s·month)）
                - profit_per_user: 单用户月收益（$/user/month）
                - node_costs: n×2，每个节点 [gpu_cost, memory_cost]（可选）
                - distance_matrix: n×n，节点间距离（跳数，用于通信成本）
                - test_data_id: （可选）调试用 ID
        """
        # 调试 ID
        self.test_data_id = int(test_data.get("test_data_id", 0))

        # 基础规模
        self.node_count = int(test_data["node_count"])
        self.module_count = int(test_data["module_count"])

        # 解析数组型数据
        self.computation_capacity = self._parse_array(test_data["computation_capacity"])
        self.resource_demands = self._parse_array(test_data["resource_demands"])
        self.data_sizes = self._parse_array(test_data["data_sizes"])
        self.bandwidth_matrix = self._parse_array(test_data["bandwidth_matrix"])

        # 成本与收益
        self.gpu_cost = float(test_data.get("gpu_cost", 0.0))
        self.memory_cost = float(test_data.get("memory_cost", 0.0))
        self.bandwidth_cost = float(test_data.get("bandwidth_cost", 0.0))
        self.profit_per_user = float(test_data.get("profit_per_user", 0.0))

        # 节点特定成本与距离矩阵
        node_costs = test_data.get("node_costs")
        distance_matrix = test_data.get("distance_matrix")

        self.node_costs = self._parse_array(node_costs) if node_costs is not None else None
        self.distance_matrix = (
            self._parse_array(distance_matrix) if distance_matrix is not None else None
        )

        # 若未提供 per-node 成本，则使用全局成本
        if self.node_costs is None:
            self.node_costs = [[self.gpu_cost, self.memory_cost] for _ in range(self.node_count)]

        # 若未提供距离矩阵，默认 1（对角线为 0）
        if self.distance_matrix is None:
            self.distance_matrix = [
                [0 if i == j else 1 for j in range(self.node_count)]
                for i in range(self.node_count)
            ]

        # 权重显存（静态，一次加载，多链复用）
        raw_weight_mem = test_data.get("weight_memory_demands")
        if raw_weight_mem is not None:
            parsed = self._parse_array(raw_weight_mem)
            if len(parsed) != self.module_count:
                raise ValueError("weight_memory_demands 长度必须等于 module_count")
            self.weight_memory_demands = [float(x) for x in parsed]
        else:
            # 兼容旧配置：若未提供，视为 0
            self.weight_memory_demands = [0.0 for _ in range(self.module_count)]

        # sanity check：单链节点不重复 => 节点数 < 模块数 时一般无解
        if self.node_count < self.module_count:
            print(
                f"警告: 节点数量({self.node_count})少于模块数量({self.module_count})，"
                "single-function 模型下每条链需要至少 m 个节点，可能无法找到可行解"
            )

        # 初始资源（备份）
        self.initial_computation_capacity = copy.deepcopy(self.computation_capacity)
        self.initial_bandwidth_matrix = copy.deepcopy(self.bandwidth_matrix)

        # 剩余资源（在多链部署过程中动态更新）
        self.remaining_computation_capacity = copy.deepcopy(self.computation_capacity)
        self.remaining_bandwidth_matrix = copy.deepcopy(self.bandwidth_matrix)

        # 节点是否仍有资源可用（算力>0 或 显存>0）
        self.node_available: List[bool] = [
            (
                float(self.remaining_computation_capacity[i][0]) > 0.0
                or float(self.remaining_computation_capacity[i][1]) > 0.0
            )
            for i in range(self.node_count)
        ]

        # 当前策略下：每个节点已加载的模块权重集合（多链之间权重复用）
        self.modules_loaded_per_node: List[set] = [set() for _ in range(self.node_count)]

    # ----------------------------------------------------------------------
    # 工具函数
    # ----------------------------------------------------------------------
    def _parse_array(self, value):
        """将可能为 JSON 字符串的数组转换为 Python 列表"""
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            try:
                return json.loads(value)
            except Exception:
                try:
                    return eval(value)
                except Exception:
                    raise ValueError(f"无法解析数组字段: {value}")
        return value

    def get_node_capacity(self, node: int) -> Tuple[float, float]:
        """获取节点剩余的 [算力, 显存] 容量"""
        if node < 0 or node >= self.node_count:
            return 0.0, 0.0
        return (
            float(self.remaining_computation_capacity[node][0]),
            float(self.remaining_computation_capacity[node][1]),
        )

    def get_module_demand(self, module: int) -> Tuple[float, float]:
        """获取单个模块的 [算力需求/用户, KV 显存需求/用户]"""
        if module < 0 or module >= self.module_count:
            return 0.0, 0.0
        return (
            float(self.resource_demands[module][0]),
            float(self.resource_demands[module][1]),
        )

    def get_module_weight_memory(self, module: int) -> float:
        """获取模块权重显存 (GB)"""
        if module < 0 or module >= self.module_count:
            return 0.0
        return float(self.weight_memory_demands[module])

    def get_link_bandwidth(self, from_node: int, to_node: int) -> float:
        """获取两个节点之间的剩余带宽 (MB/s)"""
        if (
            from_node < 0
            or from_node >= self.node_count
            or to_node < 0
            or to_node >= self.node_count
        ):
            return 0.0
        return float(self.remaining_bandwidth_matrix[from_node][to_node])

    def get_link_distance(self, from_node: int, to_node: int) -> float:
        """获取两个节点之间的距离（跳数，用于通信成本）"""
        if (
            from_node < 0
            or from_node >= self.node_count
            or to_node < 0
            or to_node >= self.node_count
        ):
            return 1.0
        return float(self.distance_matrix[from_node][to_node])

    def get_data_size(self, boundary_index: int) -> float:
        """获取相邻两个模块之间的单用户数据速率 (MB/s)"""
        if boundary_index < 0 or boundary_index >= len(self.data_sizes):
            return 0.0
        return float(self.data_sizes[boundary_index])

    # ----------------------------------------------------------------------
    # 全链：在当前剩余资源 + 已加载权重状态下，计算最大用户 & 成本
    # ----------------------------------------------------------------------
    def calculate_max_users_for_deployment(self, deployment: List[int]) -> int:
        """
        在当前剩余资源 + self.modules_loaded_per_node 状态下，
        使用“权重复用 + KV 不复用”模型计算给定部署方案的最大用户数。
        """
        n = self.node_count
        m = self.module_count

        if len(deployment) != m:
            return 0

        # 1. 逐节点统计 per-user 需求 & 本链新增权重
        node_compute_per_user = [0.0] * n
        node_kv_per_user = [0.0] * n
        node_new_weight = [0.0] * n

        for module_idx, node in enumerate(deployment):
            comp_demand, kv_demand = self.get_module_demand(module_idx)
            node_compute_per_user[node] += comp_demand
            node_kv_per_user[node] += kv_demand

            weight_mem = self.get_module_weight_memory(module_idx)
            if module_idx not in self.modules_loaded_per_node[node]:
                node_new_weight[node] += weight_mem

        limits: List[int] = []

        # 2. 节点算力 & 显存限制
        for node in range(n):
            comp_cap, mem_cap = self.get_node_capacity(node)
            comp_use = node_compute_per_user[node]
            kv_use = node_kv_per_user[node]
            new_weight = node_new_weight[node]

            if comp_use <= 0 and kv_use <= 0 and new_weight <= 0:
                continue

            if new_weight > mem_cap + 1e-9:
                return 0

            comp_limit = float("inf")
            mem_limit = float("inf")

            if comp_use > 0:
                if comp_cap <= 0:
                    return 0
                comp_limit = comp_cap / comp_use

            if kv_use > 0:
                avail_mem = mem_cap - new_weight
                if avail_mem <= 0:
                    return 0
                mem_limit = avail_mem / kv_use

            node_limit = min(comp_limit, mem_limit)
            if node_limit <= 0:
                return 0

            limits.append(int(math.floor(node_limit)))

        # 3. 链路带宽限制
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

            limits.append(int(math.floor(users_by_bandwidth)))

        if not limits:
            return 0

        return max(0, min(limits))

    def calculate_costs_for_deployment(
        self, deployment: List[int], user_count: int
    ) -> Tuple[float, float, float, float]:
        """
        使用“权重复用 + KV 不复用”模型计算单条链在当前资源视角下的成本与利润：
        - total_cost
        - deploy_cost  (算力 + 显存成本)
        - comm_cost
        - profit
        """
        n = self.node_count
        m = self.module_count

        if user_count <= 0 or len(deployment) != m:
            return 0.0, 0.0, 0.0, 0.0

        node_compute_per_user = [0.0] * n
        node_kv_per_user = [0.0] * n
        node_new_weight = [0.0] * n

        for module_idx, node in enumerate(deployment):
            comp_demand, kv_demand = self.get_module_demand(module_idx)
            node_compute_per_user[node] += comp_demand
            node_kv_per_user[node] += kv_demand

            weight_mem = self.get_module_weight_memory(module_idx)
            if module_idx not in self.modules_loaded_per_node[node]:
                node_new_weight[node] += weight_mem

        deploy_cost = 0.0
        for node in range(n):
            comp_use = node_compute_per_user[node] * user_count
            kv_use = node_kv_per_user[node] * user_count
            weight_use = node_new_weight[node]

            gpu_cost_node, mem_cost_node = self.node_costs[node]
            deploy_cost += comp_use * float(gpu_cost_node)
            deploy_cost += (kv_use + weight_use) * float(mem_cost_node)

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

    # ----------------------------------------------------------------------
    # 扣减资源（用于多链部署）
    # ----------------------------------------------------------------------
    def apply_chain_consumption(self, deployment: List[int], user_count: int) -> None:
        """
        将一条已经部署的链对应的资源占用，从“剩余资源”中扣除，
        并在 modules_loaded_per_node 中标记本次新增的权重。
        """
        if user_count <= 0:
            return

        n = self.node_count
        m = self.module_count

        node_compute_per_user = [0.0] * n
        node_kv_per_user = [0.0] * n
        node_new_weight = [0.0] * n

        for module_idx, node in enumerate(deployment):
            comp_demand, kv_demand = self.get_module_demand(module_idx)
            node_compute_per_user[node] += comp_demand
            node_kv_per_user[node] += kv_demand

            weight_mem = self.get_module_weight_memory(module_idx)
            if module_idx not in self.modules_loaded_per_node[node]:
                node_new_weight[node] += weight_mem
                self.modules_loaded_per_node[node].add(module_idx)

        # 扣减节点算力 & 显存
        for node in range(n):
            comp_cap, mem_cap = self.remaining_computation_capacity[node]
            comp_cap = float(comp_cap) - node_compute_per_user[node] * user_count
            mem_cap = (
                float(mem_cap)
                - node_kv_per_user[node] * user_count
                - node_new_weight[node]
            )
            if comp_cap < 0.0:
                comp_cap = 0.0
            if mem_cap < 0.0:
                mem_cap = 0.0

            self.remaining_computation_capacity[node][0] = comp_cap
            self.remaining_computation_capacity[node][1] = mem_cap
            self.node_available[node] = (comp_cap > 0.0 or mem_cap > 0.0)

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
    # 单条链：按模块贪心部署（single-function + 四种策略）
    # ----------------------------------------------------------------------
    def _deploy_single_chain_greedy(
        self, objective: str
    ) -> Optional[Tuple[float, float, float, float, int, int, float, List[int]]]:
        """
        在当前剩余资源状态下，使用“按模块贪心 + 单链节点不重复”的策略，
        按给定 objective 部署一条新链。

        objective:
            - "min_cost"
            - "max_profit"
            - "min_profit"
            - "max_users"

        返回:
            (total_cost, deploy_cost, comm_cost, profit,
             user_count, used_nodes_count, avg_modules_per_node, deployment)
            或 None（本策略在当前资源下无法再部署一条可行链）
        """
        n = self.node_count
        m = self.module_count

        if n <= 0 or m <= 0:
            return None

        available_nodes_count = sum(1 for v in self.node_available if v)
        if available_nodes_count < m:
            if self.test_data_id <= 5:
                print(
                    f"[test_id={self.test_data_id}] objective={objective}, "
                    f"可用节点数 {available_nodes_count} < 模块数 {m}，无法部署单链"
                )
            return None

        deployment = [-1] * m
        used_nodes_chain = [False] * n  # 单链内节点独占

        for module_idx in range(m):
            best_candidate = None  # (score_tuple, node, users_local, local_cost, local_profit)

            for node in range(n):
                # 节点必须仍有资源且在本条链中未被使用
                if not self.node_available[node]:
                    continue
                if used_nodes_chain[node]:
                    continue

                comp_demand, kv_demand = self.get_module_demand(module_idx)
                weight_mem = self.get_module_weight_memory(module_idx)
                comp_cap, mem_cap = self.get_node_capacity(node)

                if comp_demand > 0 and comp_cap <= 0:
                    continue

                is_new_weight = (module_idx not in self.modules_loaded_per_node[node])
                static_weight_add = weight_mem if is_new_weight else 0.0

                if static_weight_add > mem_cap + 1e-9:
                    continue

                comp_limit = float("inf")
                mem_limit = float("inf")

                if comp_demand > 0:
                    comp_limit = comp_cap / comp_demand

                if kv_demand > 0:
                    avail_mem = mem_cap - static_weight_add
                    if avail_mem <= 0:
                        continue
                    mem_limit = avail_mem / kv_demand

                link_limit = float("inf")
                if module_idx > 0:
                    prev_node = deployment[module_idx - 1]
                    if prev_node != node:
                        bw = self.get_link_bandwidth(prev_node, node)
                        if bw <= 0:
                            continue
                        data_size = self.get_data_size(module_idx - 1)
                        if data_size > 0:
                            link_limit = bw / data_size

                limits = []
                if comp_limit < float("inf"):
                    limits.append(comp_limit)
                if mem_limit < float("inf"):
                    limits.append(mem_limit)
                if link_limit < float("inf"):
                    limits.append(link_limit)

                if not limits:
                    users_local = 1
                else:
                    users_local = int(math.floor(min(limits)))
                    if users_local <= 0:
                        continue

                gpu_cost_node, mem_cost_node = self.node_costs[node]
                comp_use_local = comp_demand * users_local
                kv_use_local = kv_demand * users_local

                deploy_cost_local = comp_use_local * float(gpu_cost_node)
                deploy_cost_local += (kv_use_local + static_weight_add) * float(mem_cost_node)

                comm_cost_local = 0.0
                if module_idx > 0:
                    prev_node = deployment[module_idx - 1]
                    if prev_node != node:
                        data_size = self.get_data_size(module_idx - 1)
                        if data_size > 0:
                            distance = self.get_link_distance(prev_node, node)
                            comm_cost_local = (
                                data_size * distance * self.bandwidth_cost * users_local
                            )

                total_cost_local = deploy_cost_local + comm_cost_local
                profit_local = self.profit_per_user * users_local - total_cost_local

                if objective == "min_cost":
                    score = (total_cost_local, -profit_local, -users_local)
                elif objective == "max_profit":
                    score = (-profit_local, -users_local, total_cost_local)
                elif objective == "min_profit":
                    score = (profit_local, total_cost_local, users_local)
                elif objective == "max_users":
                    score = (-users_local, total_cost_local, -profit_local)
                else:
                    continue

                if best_candidate is None or score < best_candidate[0]:
                    best_candidate = (
                        score,
                        node,
                        users_local,
                        total_cost_local,
                        profit_local,
                    )

            if best_candidate is None:
                if self.test_data_id <= 5:
                    print(
                        f"[test_id={self.test_data_id}] objective={objective}, "
                        f"在部署模块 {module_idx} 时无可行节点，停止单链部署"
                    )
                return None

            _, chosen_node, users_local, local_cost, local_profit = best_candidate
            deployment[module_idx] = chosen_node
            used_nodes_chain[chosen_node] = True

            if self.test_data_id <= 3:
                print(
                    f"[test_id={self.test_data_id}] objective={objective}, "
                    f"模块 {module_idx} 选择节点 {chosen_node} "
                    f"(局部 users={users_local}, local_cost={local_cost:.4f}, local_profit={local_profit:.4f})"
                )

        # 完整链部署后，用全量资源模型计算真实 max_users & 成本
        max_users = self.calculate_max_users_for_deployment(deployment)
        if max_users <= 0:
            if self.test_data_id <= 5:
                print(
                    f"[test_id={self.test_data_id}] objective={objective}, "
                    f"完整链部署后 max_users={max_users}，视为不可行"
                )
            return None

        total_cost, deploy_cost, comm_cost, profit = self.calculate_costs_for_deployment(
            deployment, max_users
        )

        used_nodes_count = len(set(deployment))
        avg_mods_per_node = (
            self.module_count / used_nodes_count if used_nodes_count > 0 else 0.0
        )

        if self.test_data_id <= 3:
            print(
                f"[test_id={self.test_data_id}] objective={objective}, "
                f"完整链：users={max_users}, total_cost={total_cost:.4f}, profit={profit:.4f}, "
                f"used_nodes={used_nodes_count}, avg_mods_per_node={avg_mods_per_node:.3f}"
            )

        return (
            total_cost,
            deploy_cost,
            comm_cost,
            profit,
            max_users,
            used_nodes_count,
            avg_mods_per_node,
            deployment.copy(),
        )

    # ----------------------------------------------------------------------
    # 多链循环：在指定 objective 下吃干网络资源
    # ----------------------------------------------------------------------
    def deploy_until_exhaustion(self, objective: str):
        """
        在指定 objective 下，从初始资源出发，循环部署多条单功能链，
        直到无法再部署任何一条可行链。

        返回 8 元组:
            (total_cost, total_deploy_cost, total_comm_cost,
             total_profit, total_users, used_nodes_count,
             avg_modules_per_node, chain_count)
        或 None（完全无法部署）
        """
        # 重置剩余资源和状态
        self.remaining_computation_capacity = copy.deepcopy(self.initial_computation_capacity)
        self.remaining_bandwidth_matrix = copy.deepcopy(self.initial_bandwidth_matrix)
        self.modules_loaded_per_node = [set() for _ in range(self.node_count)]
        self.node_available = [
            (
                float(self.remaining_computation_capacity[i][0]) > 0.0
                or float(self.remaining_computation_capacity[i][1]) > 0.0
            )
            for i in range(self.node_count)
        ]

        total_cost = 0.0
        total_deploy_cost = 0.0
        total_comm_cost = 0.0
        total_profit = 0.0
        total_users = 0
        used_nodes_union = set()
        chain_count = 0

        if self.test_data_id <= 3:
            print(f"\n[test_id={self.test_data_id}] === 开始 objective={objective} 的多链部署 ===")

        while True:
            best = self._deploy_single_chain_greedy(objective)
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

            if self.test_data_id <= 3:
                print(
                    f"[test_id={self.test_data_id}] objective={objective}, "
                    f"部署第 {chain_count} 条链：users={user_count}, "
                    f"cost={cost:.4f}, profit={profit:.4f}"
                )

            self.apply_chain_consumption(deployment, user_count)

        if chain_count == 0:
            if self.test_data_id <= 3:
                print(
                    f"[test_id={self.test_data_id}] objective={objective}, "
                    f"完全无法部署任何链"
                )
            return None

        used_nodes_count = len(used_nodes_union)
        avg_modules_per_node = (
            self.module_count * chain_count / used_nodes_count
            if used_nodes_count > 0
            else 0.0
        )

        if self.test_data_id <= 3:
            print(
                f"[test_id={self.test_data_id}] === objective={objective} 多链部署结束："
                f"chain_count={chain_count}, total_users={total_users}, "
                f"total_profit={total_profit:.4f}, total_cost={total_cost:.4f} ==="
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

    # ----------------------------------------------------------------------
    # 对外主接口：四种策略
    # ----------------------------------------------------------------------
    def optimize_for_profit(self):
        """
        与 multi 优化器保持一致，对同一网络 / 配置，
        分别在以下四种目标下从初始资源出发吃干榨净：

        1. min_cost   : 总成本最小（每次模块选局部成本最小）
        2. max_profit : 总利润最大（每次模块选局部利润最大）
        3. min_profit : 总利润最小（每次模块选局部利润最小，最差情况）
        4. max_users  : 总用户量最大（每次模块选局部用户数最大）

        返回:
            (min_cost_plan, max_profit_plan, min_profit_plan, max_users_plan)

        每个 plan 为 8 元组：
            (total_cost, total_deploy_cost, total_comm_cost,
             total_profit, total_users, used_nodes_count,
             avg_modules_per_node, chain_count)
        """
        min_cost_plan = self.deploy_until_exhaustion("min_cost")
        max_profit_plan = self.deploy_until_exhaustion("max_profit")
        min_profit_plan = self.deploy_until_exhaustion("min_profit")
        max_users_plan = self.deploy_until_exhaustion("max_users")

        return (min_cost_plan, max_profit_plan, min_profit_plan, max_users_plan)

    # ----------------------------------------------------------------------
    # 兼容老接口：single_func_deployment() 仍然返回一个 8 元组
    # ----------------------------------------------------------------------
    def single_func_deployment(self):
        """
        保留原来的接口形式：
        - 返回一个 8 元组（用于老代码），语义：max_profit 策略下的多链部署结果。

        等价于：
            _, max_profit_plan, _, _ = self.optimize_for_profit()
            return max_profit_plan
        """
        _min_cost, max_profit_plan, _min_profit, _max_users = self.optimize_for_profit()
        return max_profit_plan
