
class logger(object):

    def __init__(self, holder, name="default"):
        self.holder = holder
        self.name = name

    def log_with_name(self, content, condition=False):
        if condition:
            print(f"[{self.name}]-{content}")
