import sys
import types
import os.path as osp
from copy import copy

import importlib

class Config(object):

    def __init__(self, cfg_dict=None, filename=None):
        
        if cfg_dict is None:
            cfg_dict = dict()

        elif not isinstance(cfg_dict, dict):
            raise TypeError('cfg_dict must be a dict, but '
                            f'got {type(cfg_dict)}')

        super().__setattr__('_filename', filename)

        text = ''
        if filename is not None:
            if isinstance(filename, list):
                for file in filename:
                    with open(file, 'r') as f:
                        text += f.read() 
            else:                   
                with open(filename, 'r') as f:
                    text = f.read()

        super().__setattr__('_text', text)
        
        # set attributes from cfg_dict
        for k, v in cfg_dict.items():
            if isinstance(v, dict):
                setattr(self, k, Config(v))
            else:   
                setattr(self, k, v)

    @staticmethod
    def fromfile(filename, import_custom_modules=False):
        if isinstance(filename, list):
            cfg_dict = {}
            for filename in filename: 
                cfg_dict.update(Config._file2dict(filename) )

        else: 
            cfg_dict = Config._file2dict(filename)
        
        if import_custom_modules:
            import_modules_from_strings(**cfg_dict['custom_imports'])

        return Config(cfg_dict, filename=filename)

    @staticmethod
    def _merge_a_into_b(a, b):
        """merge dict ``a`` into dict ``b`` (non-inplace).
        Values in ``a`` will overwrite ``b``. ``b`` is copied first to avoid
        in-place modifications.
        Args:
            a (dict): The source dict to be merged into ``b``.
            b (dict): The origin dict to be fetch keys from ``a``.
        Returns:
            dict: The modified dict of ``b`` using ``a``.
        """
        b = copy(b)
        for k, v in a.items():
            # print(k, v, k in b)
            if isinstance(v, dict):
                if k in b:
                    b[k] = Config._merge_a_into_b(v, b[k])
                else:
                    b[k] = v
            else:
                b[k] = v
        return b

    @staticmethod
    def _file2dict(config_file):

        config_file = osp.abspath(config_file)

        if not osp.exists(config_file):
            raise Exception(f'{config_file} not exits')

        config_dir = osp.dirname(config_file)
        sys.path.insert(0, config_dir)

        config_name = osp.basename(config_file)
        module_name = osp.splitext(config_name)[0]

        # reload for latest changes
        if module_name in sys.modules:
            modules = importlib.reload(sys.modules[module_name])
        else:
            modules = importlib.import_module(module_name)

        sys.path.pop(0)

        cfg_dict = {
            name: value for name, value in modules.__dict__.items()
            if not name.startswith('__')
            and not isinstance(value, types.ModuleType)
            and not isinstance(value, types.FunctionType)
        }

        if '_base_' in cfg_dict:
            config_dir = osp.dirname(config_file)
            base_files = cfg_dict.pop('_base_')

            cfg_dict_list = list()
            for f in base_files:
                _cfg_dict = Config._file2dict(osp.join(config_dir, f))
                cfg_dict_list.append(_cfg_dict)

            base_cfg_dict = dict()
            for c in cfg_dict_list:
                duplicate_keys = base_cfg_dict.keys() & c.keys()
                if len(duplicate_keys) > 0:
                    raise KeyError('Duplicate key is not allowed among bases. '
                                   f'Duplicate keys: {duplicate_keys}')
                base_cfg_dict.update(c)

            cfg_dict = Config._merge_a_into_b(cfg_dict, base_cfg_dict)

        return cfg_dict


    @property
    def serialize(self, ):
        return self._serialize(self.__dict__)

    def _serialize(self, self_dict):

        _dict = {}
        for k, v in self_dict.items():

            if k.startswith('__') or k.startswith('_'):
                continue

            if isinstance(v, Config):
                _dict[k] = self._serialize(v.__dict__)

            else:
                _dict[k] = v

        return _dict 


def import_modules_from_strings(imports, allow_failed_imports=False):
    """Import modules from the given list of strings.
    Args:
        imports (list | str | None): The given module names to be imported.
        allow_failed_imports (bool): If True, the failed imports will return
            None. Otherwise, an ImportError is raise. Default: False.
    Returns:
        list[module] | module | None: The imported modules.
    Examples:
        >>> osp, sys = import_modules_from_strings(
        ...     ['os.path', 'sys'])
        >>> import os.path as osp_
        >>> import sys as sys_
        >>> assert osp == osp_
        >>> assert sys == sys_
    """
    if not imports:
        return
    single_import = False
    if isinstance(imports, str):
        single_import = True
        imports = [imports]
    if not isinstance(imports, list):
        raise TypeError(
            f'custom_imports must be a list but got type {type(imports)}')
    imported = []
    for imp in imports:
        if not isinstance(imp, str):
            raise TypeError(
                f'{imp} is of type {type(imp)} and cannot be imported.')
        try:
            imported_tmp = import_module(imp)
        except ImportError:
            if allow_failed_imports:
                warnings.warn(f'{imp} failed to import and is ignored.',
                              UserWarning)
                imported_tmp = None
            else:
                raise ImportError
        imported.append(imported_tmp)
    if single_import:
        imported = imported[0]
    return imported