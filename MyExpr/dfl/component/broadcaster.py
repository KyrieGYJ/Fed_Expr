

class Broadcaster(object):

    def __init__(self):
        # neighbors_weight_dict
        self.receive = None
        self.send = None
        self.recorder = None

    def register_recorder(self, recorder):
        self.recorder = recorder

    def use(self, strategy):
        description = "Broadcaster use strategy:{:s}"
        print(description.format(strategy))
        if strategy == "flood":
            self.send = self.send_to_neighbors_flood
        self.receive = self.receive_from_neighbors

    # todo 后续要利用上topology_weight（类似dfl论文里的参考pagerank）
    def send_to_neighbors_flood(self, sender_id, model):
        client_dic = self.recorder.client_dic
        topology = self.recorder.topology_manager.get_symmetric_neighbor_list(sender_id)
        for receiver_id in client_dic.keys():
            if topology[receiver_id] != 0 and receiver_id != sender_id:
                self.receive_from_neighbors(sender_id, model, receiver_id, topology[receiver_id])

    def receive_from_neighbors(self, sender_id, model, receiver_id, topology_weight):
        receiver = self.recorder.client_dic[receiver_id]
        # 调用receiver的方法，显示收到了某个client的数据。。（相当于钩子函数）
        receiver.response(sender_id)
        receiver.neighbors_weight_dict[sender_id] = model
        receiver.neighbors_topology_weight_dict[sender_id] = topology_weight
