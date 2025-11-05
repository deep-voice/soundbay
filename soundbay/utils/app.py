class App:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._args = None
            cls._instance.states = {}
        return cls._instance

    def init(self, args):
        if self._args is not None:
            raise RuntimeError("App already initialized — args are immutable")
        self._args = args

    @property
    def args(self):
        if self._args is None:
            raise RuntimeError("App not initialized. Call app.init(args) first.")
        return self._args

    def __repr__(self):
        return f"<App args={self._args} states={self.states}>"
