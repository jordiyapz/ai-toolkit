from jordiyapz_ai_toolkit.utils import timeit


class Pipeline:
    def __init__(self, origin=None, pipes=[]):
        self._origin = origin
        self._stash = {}
        self._cache = origin
        self._pipeline = pipes
        self._last_compile_i = -1

    def __repr__(self):
        pipes = '' if not self._pipeline else ',\n    pipes=(\n        {})'.format(
            ',\n        '.join(('{}({})'.format(fn.__name__,
                                                ', '.join(
                                                    args) if args else '',
                                                'skipable=True' if skipable else '')
                               for fn, args, skipable in self._pipeline)))
        return 'Pipeline({}{})'.format(type(self._origin), pipes)

    def set_origin(self, origin):
        self._origin = origin
        self._cache = origin
        return self

    def set_cache(self, cache, index=None):
        self._cache = cache
        if index is not None:
            self._last_compile_i = index
        return self

    def reset(self):
        self._cache = self._origin
        self._last_compile_i = -1
        return self

    def add_pipe(self, order=None, args=None, skipable=False):
        '''
            add_pipe(i)
                return lambda (prev_data, args, stash): new_data
            'fn' must always return immutables
        '''

        last_i = len(self._pipeline)
        if order is None:
            order = last_i
        elif order < 0 or order > last_i:
            raise IndexError('Index must be in range 0 to {}'.format(last_i))

        def decorator(fn, args_inner=None):
            arguments = args_inner if args_inner is not None else args
            if order == last_i:
                self._pipeline.append((fn, arguments, skipable))
            else:
                self._pipeline[order] = (fn, arguments, skipable)
            return fn

        return decorator

    def add_stash(self, key, value):
        self._stash[key] = value
        return self

    def compile(self, cache=True, stop=None, skip=False,
                skip_index=[], verbose=2):
        current = self._cache

        stop_i = stop+1 if stop is not None else len(self._pipeline)
        start_i = self._last_compile_i + 1
        if start_i > stop_i:
            start_i = 0
            current = self._origin

        for i, pipe in enumerate(self._pipeline[start_i:stop_i]):
            fn, args, skipable = pipe
            if (skip and skipable) or i in skip_index:
                continue
            if verbose > 0:
                if verbose > 1:
                    name = fn.__name__
                    fn = timeit(fn)
                    fn.__name__ = name
                print('Executing[{}]: {}...'.format(start_i+i, fn.__name__))
            current = fn(current, args, self._stash)

        if cache:
            self._last_compile_i = stop_i-1
            self._cache = current

        return current

    def peek(self, stop=None, cache=False, skip=False,
             skip_index=[], verbose=2):
        if stop and stop >= len(self._pipeline):
            raise IndexError('Index must be in range 0 to {}'
                             .format(len(self._pipeline)-1))

        current = self.compile(cache, stop, skip, skip_index, verbose)
        return current
