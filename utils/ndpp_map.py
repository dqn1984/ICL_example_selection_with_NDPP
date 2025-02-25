import numpy as np
import logging

logger = logging.getLogger(__name__)

def greedy_map_ndpp(B, C, num_to_choose):
    # 确保C是方阵
    if C.shape[0] != C.shape[1]:
        raise ValueError('C should be a square matrix')
    # 确保B的列数和C的行数相等
    if B.shape[1] != C.shape[0]:
        raise ValueError('Number of rows in B and C should be equal')
    # 确保选择的数量不超过B的秩
    num_rank = B.shape[1]
    if num_to_choose > num_rank:
        raise ValueError('num_to_choose should be less than or equal to the rank of B')

    # 计算核矩阵L = B * C * B'的对角线
    L = np.dot(B, np.matmul(C, B.T))
    marginal_gain = np.diagonal(L)

    # 寻找最大化边际增益的项并添加到输出集合
    max_marginal_gain = np.max(marginal_gain)
    item_argmax = np.argmax(marginal_gain)
    chosen_set = [item_argmax]

    # 初始化用于更新边际增益的矩阵
    P = np.array([]).reshape(0, B.shape[1])
    Q = np.array([]).reshape(0, B.shape[1])

    for i in range(1, num_to_choose):
        b_argmax = B[item_argmax, :]
        if len(chosen_set) == 1:
            p = b_argmax / max_marginal_gain
            q = b_argmax
        else:
            p = (b_argmax - np.matmul(np.matmul(np.matmul(b_argmax, C.T), Q.T), P)) / max_marginal_gain
            q = b_argmax - np.matmul(np.matmul(np.matmul(b_argmax, C), P.T), Q)

        P = np.vstack((P, p))
        Q = np.vstack((Q, q))

        # 更新边际增益
        test = np.array(np.dot(np.matmul(np.matmul(B, C), p.T), np.matmul(np.matmul(B, C.T), q.T)))
        marginal_gain = marginal_gain - test
        if len(chosen_set) >= num_to_choose:
            break

        # 寻找下一个最大化边际增益的项并添加到输出集合
        max_marginal_gain = np.max(marginal_gain)
        item_argmax = np.argmax(marginal_gain)
        marginal_copy = np.copy(marginal_gain)
        while item_argmax in chosen_set:
            marginal_copy[item_argmax] = float('-inf')
            item_argmax = np.argmax(marginal_copy)
        max_marginal_gain = np.max(marginal_copy)
        chosen_set.append(item_argmax)

    return chosen_set
