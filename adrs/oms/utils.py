class aobject(object):
    """
    Async implementation of python's `object`.
    This allows to define Python Classes with `async def __init__(self)`
    """

    async def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        await instance.__init__(*args, **kwargs)
        return instance

    async def __init__(self):
        pass
