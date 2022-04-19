
# 存放历史模型副本，client根据索引获取，单机模拟情况下每个client复制一次将十分浪费内存。
# class global_history(object):