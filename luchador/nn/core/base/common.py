from __future__ import absolute_import


class CopyMixin(object):
    """Provide copy method which creates a copy of initialed instance"""
    def copy(self):
        """Create and initialize new instance with the same argument

        Typical usage is
        ```
        class Foo(CopyMixin, object):
            def __init__(self, arg1, arg2):
                self._store_args({arg1: arg1, arg2: arg2})

            def _validate_args(self, args):
                if not isinstance(args['arg1'], int):
                    raise ValueError('`arg1` must be int.')

        obj1 = Foo(arg1=1, arg2=None)
        obj2 = obj1.copy()  # object initialized with the
                            # same constructor argument as obj1
        ```
        """
        return type(self)(**self.args)

    def _store_args(self, args):
        """Store initializer arguments to args

        See `copy` method for detail

        Args:
            args (dict): Arguments passed to __init__.
            This will be used to recreate a subclass instance.

        """
        self._validate_args(args)
        self.args = args

    def _validate_args(self, args):
        """Validate arguments

        See `copy` method for detail

        Args:
            args (dict): Arguments passed to __init__.
        """
        pass

    def __repr__(self):
        return "{{'name': '{}', 'args': {}}}".format(
            self.__class__.__name__, self.args)

    ###########################################################################
    # Shallow equality
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.args == other.args
        return NotImplemented

    def __ne__(self, other):
        return not self.__eq__(other)

    ###########################################################################
    def serialize(self):
        """Serialize object configuration (constructor arguments)"""
        args = {}
        for key, val in self.args.items():
            args[key] = val.serialize() if hasattr(val, 'serialize') else val
        return {
            'name': self.__class__.__name__,
            'args': args
        }
