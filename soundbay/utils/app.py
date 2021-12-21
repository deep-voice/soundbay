class App:
    '''
    Class to be used as a global params and states handler across the project
    '''
    class _App:
        def __init__(self, args):
            self.args = args
            self.states = {}

    @classmethod
    def init(cls, args):
        App.inner = App._App(args)

    def __getattr__(self, item):
        return getattr(self.inner, item)


app = App()
