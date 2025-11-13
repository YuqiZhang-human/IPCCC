#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
单功能部署优化器
基于分支定界法实现，每个节点最多只能部署一个功能
完全基于物理限制计算最大用户量，没有随机因素
"""

import os
import sys
import json
import numpy as np
import math
from collections import defaultdict
import copy

class SingleFunctionOptimizer:
    """
    单功能部署优化器类，每个节点最多只能部署一个功能
    基于分支定界法实现
    """
    
    def __init__(self, test_data):
        """
        初始化单功能部署优化器
        
        Args:
            test_data (dict): 测试数据字典，包含如下字段：
                - node_count: 节点数量
                - module_count: 模块数量
                - computation_capacity: 计算能力列表 n×2的矩阵，第1列是算力，第2列是存储
                - resource_demands: 资源需求列表 m×2的矩阵，第1列是算力，第2列是存储
                - data_sizes: 数据大小列表，长度为m-1
                - bandwidth_matrix: 带宽矩阵 n×n的矩阵
                - gpu_cost: GPU成本系数
                - memory_cost: 内存成本系数
                - bandwidth_cost: 带宽成本系数
                - profit_per_user: 每个用户的利润
                - node_costs: 每个节点的GPU和内存成本 n×2的矩阵
                - distance_matrix: 节点间距离矩阵 n×n的矩阵
        """
        # 保存test_data_id用于调试
        self.test_data_id = test_data.get('test_data_id', 0)
        
        # 基础配置
        self.node_count = int(test_data['node_count'])
        self.module_count = int(test_data['module_count'])
        
        # 处理可能是字符串的输入
        self.computation_capacity = self._parse_array(test_data['computation_capacity'])
        self.resource_demands = self._parse_array(test_data['resource_demands'])
        self.data_sizes = self._parse_array(test_data['data_sizes'])
        self.bandwidth_matrix = self._parse_array(test_data['bandwidth_matrix'])
        
        # 成本参数
        self.gpu_cost = float(test_data['gpu_cost'])
        self.memory_cost = float(test_data['memory_cost'])
        self.bandwidth_cost = float(test_data['bandwidth_cost'])
        self.profit_per_user = float(test_data['profit_per_user'])
        
        # 节点特定的成本和距离
        self.node_costs = self._parse_array(test_data.get('node_costs', None))
        self.distance_matrix = self._parse_array(test_data.get('distance_matrix', None))
        
        # 如果未提供节点特定成本，使用全局成本
        if self.node_costs is None:
            self.node_costs = [[self.gpu_cost, self.memory_cost] for _ in range(self.node_count)]
        
        # 如果未提供距离矩阵，使用默认距离1
        if self.distance_matrix is None:
            self.distance_matrix = [[1 if i != j else 0 for j in range(self.node_count)] for i in range(self.node_count)]
        
        # 检查节点数量是否足够
        if self.node_count < self.module_count:
            print(f"警告: 节点数量({self.node_count})少于模块数量({self.module_count})，单功能部署可能无法找到解")
        
        # 初始化最优解
        self.best_deployment = None  # 最优部署方案
        self.best_cost = float('inf')  # 最小成本
        self.best_profit = float('-inf')  # 最大利润
        self.best_user_count = 0  # 最大用户数
        
        # 记录所有满足条件的解
        self.all_solutions = []
    
    def _parse_array(self, data):
        """解析数据，如果是JSON字符串则转换为列表"""
        if data is None:
            return None
            
        if isinstance(data, str):
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                return data
        return data
    
    def get_node_capacity(self, node):
        """获取节点的计算和存储能力"""
        if node < 0 or node >= self.node_count:
            return 0, 0
        return self.computation_capacity[node][0], self.computation_capacity[node][1]
    
    def get_module_demand(self, module):
        """获取模块的计算和存储需求"""
        if module < 0 or module >= self.module_count:
            return 0, 0
        return self.resource_demands[module][0], self.resource_demands[module][1]
    
    def get_link_bandwidth(self, from_node, to_node):
        """获取两个节点之间的带宽"""
        if from_node < 0 or from_node >= self.node_count or to_node < 0 or to_node >= self.node_count:
            return 0
        return self.bandwidth_matrix[from_node][to_node]
    
    def get_link_distance(self, from_node, to_node):
        """获取两个节点之间的距离"""
        if from_node < 0 or from_node >= self.node_count or to_node < 0 or to_node >= self.node_count:
            return 1  # 默认距离为1
        return self.distance_matrix[from_node][to_node]
    
    def get_data_size(self, module):
        """获取模块输出的数据大小"""
        if module < 0 or module >= self.module_count - 1:
            return 0
        return self.data_sizes[module]
    
    def get_node_costs(self, node):
        """获取节点的GPU和内存成本"""
        if node < 0 or node >= self.node_count:
            return self.gpu_cost, self.memory_cost  # 默认使用全局成本
        return self.node_costs[node][0], self.node_costs[node][1]
    
    def calculate_max_users(self, deployment):
        """
        计算给定部署方案的最大用户数量
        基于木桶原理，找到所有资源限制中的最小值
        并向下取整以确保返回整数用户数
        
        Args:
            deployment (list): 部署方案，每个元素是节点索引
            
        Returns:
            int: 最大用户数量(整数)
        """
        # 1. 计算节点资源限制
        node_resource_limits = []
        
        # 统计每个节点的资源使用情况
        node_compute_usage = defaultdict(float)
        node_storage_usage = defaultdict(float)
        
        for module_idx, node_idx in enumerate(deployment):
            compute_demand, storage_demand = self.get_module_demand(module_idx)
            node_compute_usage[node_idx] += compute_demand
            node_storage_usage[node_idx] += storage_demand
        
        # 计算每个节点的最大用户数
        for node_idx in set(deployment):
            compute_capacity, storage_capacity = self.get_node_capacity(node_idx)
            
            # 如果算力需求大于0，计算基于算力的最大用户数
            if node_compute_usage[node_idx] > 0:
                compute_limit = math.floor(compute_capacity / node_compute_usage[node_idx])
                node_resource_limits.append(compute_limit)
            
            # 如果存储需求大于0，计算基于存储的最大用户数
            if node_storage_usage[node_idx] > 0:
                storage_limit = math.floor(storage_capacity / node_storage_usage[node_idx])
                node_resource_limits.append(storage_limit)
        
        # 2. 计算链路带宽限制
        link_bandwidth_limits = []
        
        # 检查每对相邻模块
        for i in range(self.module_count - 1):
            from_node = deployment[i]
            to_node = deployment[i + 1]
            
            # 如果相邻模块部署在同一节点上，不需要考虑带宽
            if from_node == to_node:
                continue
            
            # 检查带宽是否为0
            bandwidth = self.get_link_bandwidth(from_node, to_node)
            if bandwidth <= 0:
                # 如果带宽为0，不能支持任何用户
                return 0
            
            # 计算基于带宽的最大用户数
            data_size = self.get_data_size(i)
            if data_size > 0:  # 避免除以零
                link_limit = math.floor(bandwidth / data_size)
                link_bandwidth_limits.append(link_limit)
        
        # 3. 木桶原理：系统整体最大用户量 = 所有限制中的最小值
        all_limits = node_resource_limits + link_bandwidth_limits
        if not all_limits:
            return 0  # 如果没有有效限制，返回0
        
        return min(all_limits)  # 最小值已经是取整后的结果
    
    def calculate_total_cost(self, deployment, user_count):
        """
        计算给定部署方案的总成本
        
        Args:
            deployment (list): 部署方案，每个元素是节点索引
            user_count (float): 用户数量
            
        Returns:
            tuple: (total_cost, compute_storage_cost, communication_cost) 总成本、计算存储成本、通信成本
        """
        # 1. 计算计算和存储成本
        compute_storage_cost = 0
        
        # 统计每个节点的资源使用情况
        node_compute_usage = defaultdict(float)
        node_storage_usage = defaultdict(float)
        
        for module_idx, node_idx in enumerate(deployment):
            compute_demand, storage_demand = self.get_module_demand(module_idx)
            node_compute_usage[node_idx] += compute_demand * user_count
            node_storage_usage[node_idx] += storage_demand * user_count
        
        # 计算每个节点的成本
        for node_idx, compute_usage in node_compute_usage.items():
            gpu_cost, memory_cost = self.get_node_costs(node_idx)
            node_cost = (compute_usage * gpu_cost + 
                         node_storage_usage[node_idx] * memory_cost)
            compute_storage_cost += node_cost
        
        # 2. 计算通信成本
        communication_cost = 0
        
        # 检查每对相邻模块
        for i in range(self.module_count - 1):
            from_node = deployment[i]
            to_node = deployment[i + 1]
            
            # 如果相邻模块部署在同一节点上，不需要考虑通信成本
            if from_node == to_node:
                continue
            
            # 计算通信成本 = 数据大小 * 带宽成本 * 距离 * 用户数
            data_size = self.get_data_size(i)
            distance = self.get_link_distance(from_node, to_node)
            
            communication_cost += data_size * self.bandwidth_cost * distance * user_count
        
        # 3. 总成本 = 计算存储成本 + 通信成本
        total_cost = compute_storage_cost + communication_cost
        
        return total_cost, compute_storage_cost, communication_cost
    
    def is_deployment_feasible(self, deployment):
        """
        检查部署方案是否可行
        
        Args:
            deployment: 部署方案，每个元素是节点索引
            
        Returns:
            bool: 是否可行
        """
        # 检查长度
        if len(deployment) != self.module_count:
            return False
        
        # 检查节点使用情况
        node_usage = {}  # 节点使用情况
        
        # 检查每个节点的资源使用
        for module_idx, node_idx in enumerate(deployment):
            # 检查节点索引是否有效
            if node_idx < 0 or node_idx >= self.node_count:
                return False
            
            # 获取节点容量
            compute_capacity, storage_capacity = self.get_node_capacity(node_idx)
            
            # 获取模块资源需求
            compute_demand, storage_demand = self.get_module_demand(module_idx)
            
            # 如果节点还没有记录，初始化
            if node_idx not in node_usage:
                node_usage[node_idx] = [0, 0]  # [计算使用, 存储使用]
            
            # 更新节点资源使用
            node_usage[node_idx][0] += compute_demand
            node_usage[node_idx][1] += storage_demand
            
            # 检查资源是否超过容量
            if node_usage[node_idx][0] > compute_capacity or node_usage[node_idx][1] > storage_capacity:
                return False
        
        # 检查每个相邻模块之间的带宽
        for i in range(self.module_count - 1):
            from_node = deployment[i]
            to_node = deployment[i + 1]
            
            # 如果相邻模块部署在同一节点上，不需要考虑带宽
            if from_node == to_node:
                continue
            
            # 获取链路带宽
            bandwidth = self.get_link_bandwidth(from_node, to_node)
            
            # 检查带宽是否为0（即没有连接）
            if bandwidth <= 0:
                return False
        
        # 对于单功能部署，每个节点最多只能部署一个功能
        used_nodes = set(deployment)
        if len(used_nodes) != self.module_count:
            return False
        
        return True
    
    def search_all_deployments(self):
        """
        使用深度优先搜索方法搜索所有可行的部署方案
        使用分支定界法进行剪枝
        """
        # 清空之前的结果
        self.all_solutions = []
        
        # 使用栈进行深度优先搜索
        stack = [([], set(), 0)]  # (当前部署方案, 已使用节点集合, 当前模块索引)
        
        while stack:
            current_deployment, used_nodes, module_idx = stack.pop()
            
            # 如果已经部署了所有模块，评估该方案
            if module_idx == self.module_count:
                # 计算最大用户数量
                max_users = self.calculate_max_users(current_deployment)
                
                if max_users > 0:  # 如果方案可行
                    # 计算成本
                    total_cost, compute_storage_cost, communication_cost = self.calculate_total_cost(current_deployment, max_users)
                    # 计算利润
                    profit = max_users * self.profit_per_user - total_cost
                    
                    # 保存方案
                    self.all_solutions.append((total_cost, profit, max_users, current_deployment))
                
                continue
            
            # 尝试将当前模块部署到每个未使用的节点
            for node_idx in range(self.node_count):
                # 单功能部署：如果节点已经被使用，跳过
                if node_idx in used_nodes:
                    continue
                
                # 创建新的部署方案和已使用节点集合
                new_deployment = current_deployment + [node_idx]
                new_used_nodes = used_nodes.copy()
                new_used_nodes.add(node_idx)
                
                # 剪枝判断：如果已经不可行，就不继续探索
                if not self._is_partial_feasible(new_deployment, module_idx):
                    continue
                
                # 将新方案加入栈中
                stack.append((new_deployment, new_used_nodes, module_idx + 1))
        
        # 根据需要对所有方案进行排序
        if self.all_solutions:
            self.all_solutions.sort(key=lambda x: (x[0], -x[2]))  # 先按成本升序，再按用户数降序
    
    def _is_partial_feasible(self, partial_deployment, current_module):
        """
        检查部分部署方案是否可行
        
        Args:
            partial_deployment: 部分部署方案
            current_module: 当前已部署到的模块索引
            
        Returns:
            bool: 是否可行
        """
        # 边界检查
        if not partial_deployment or current_module < 0:
            return True
        
        # 检查最后部署的模块
        latest_module = current_module
        node_idx = partial_deployment[latest_module]
        
        # 检查节点资源是否足够
        compute_capacity, storage_capacity = self.get_node_capacity(node_idx)
        compute_demand, storage_demand = self.get_module_demand(latest_module)
        
        # 检查资源是否足够
        if compute_demand > compute_capacity or storage_demand > storage_capacity:
            return False
        
        # 如果不是第一个模块，检查与前一个模块的连通性
        if latest_module > 0:
            prev_node = partial_deployment[latest_module - 1]
            
            # 如果不在同一节点，检查带宽
            if prev_node != node_idx and self.get_link_bandwidth(prev_node, node_idx) <= 0:
                return False
        
        return True
    
    def single_func_deployment(self):
        """
        单功能部署优化
        
        Returns:
            tuple: (cost, compute_storage_cost, communication_cost, profit, user_count, 
                   used_nodes_count, avg_modules_per_node) 
                   如果找到有效方案，否则返回None
        """
        # 搜索所有可行的部署方案
        self.search_all_deployments()
        
        # 如果没有找到任何解
        if not self.all_solutions:
            return None
        
        # 为每个方案计算详细成本
        detailed_solutions = []
        for solution in self.all_solutions:
            cost, profit, user_count, deployment = solution
            _, compute_storage_cost, communication_cost = self.calculate_total_cost(deployment, user_count)
            
            # 计算使用的节点数量
            used_nodes_count = len(set(deployment))
            
            # 计算平均每节点模块数
            avg_modules_per_node = self.module_count / used_nodes_count if used_nodes_count > 0 else 0
            
            detailed_solutions.append((cost, compute_storage_cost, communication_cost, profit, 
                                      user_count, used_nodes_count, avg_modules_per_node, deployment))
        
        # 找到最佳方案：先按用户量降序，再按成本升序,按利润降序
        detailed_solutions.sort(key=lambda x: (-x[3],-x[4], x[0]))
        best_solution = detailed_solutions[0]
        
        # 返回成本、计算存储成本、通信成本、利润、用户数、使用节点数、平均每节点模块数
        return best_solution[:7] 