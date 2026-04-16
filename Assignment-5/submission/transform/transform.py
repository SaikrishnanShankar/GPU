class SaSSTransform:
    def __init__(self, name):
        self.name = name

    def apply(self, module):
        self.module = module

