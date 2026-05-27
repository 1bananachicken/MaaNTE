from typing import Any, TypeGuard, get_args, get_origin


def type_match[T](val: Any, t: type[T]) -> TypeGuard[T]:
    """
    检查值是否匹配给定的类型（支持泛型）

    支持的类型包括：
    - 基本类型: int, str, list, dict, tuple, set
    - 泛型类型: list[str], dict[str, int], tuple[int, ...]
    - 联合类型: int | str, Union[int, str]
    - 可选类型: Optional[str]
    - 嵌套泛型: list[list[str]], dict[str, list[int]]

    Args:
        val: 要检查的值
        t: 目标类型，可以是普通类型或泛型

    Returns:
        bool: 值是否匹配目标类型

    """
    t_org = get_origin(t)

    # 如果不是泛型类型，直接使用 isinstance
    if t_org is None:
        return isinstance(val, t)

    # 首先检查是否是 b_org 的实例
    if not isinstance(val, t_org):
        return False

    # 获取类型参数
    args = get_args(t)
    if not args:  # 没有类型参数，如 List
        return True

    # 根据不同的原始类型进行检查
    if t_org is list:
        # list[T] 检查
        if len(args) == 1:
            elem_type = args[0]
            return all(type_match(item, elem_type) for item in val)

    elif t_org is tuple:
        # tuple[T1, T2, ...] 或 tuple[T, ...] 检查
        if len(args) == 2 and args[1] is ...:  # 可变长度元组
            elem_type = args[0]
            return all(type_match(item, elem_type) for item in val)
        # 固定长度元组
        if len(val) != len(args):
            return False
        return all(type_match(item, t) for item, t in zip(val, args, strict=False))

    elif t_org is dict:
        # dict[K, V] 检查
        if len(args) == 2:
            key_type, value_type = args
            return all(
                type_match(k, key_type) and type_match(v, value_type)
                for k, v in val.items()
            )

    elif t_org is set:
        # set[T] 检查
        if len(args) == 1:
            elem_type = args[0]
            return all(type_match(item, elem_type) for item in val)

    elif t_org is frozenset:
        # frozenset[T] 检查
        if len(args) == 1:
            elem_type = args[0]
            return all(type_match(item, elem_type) for item in val)

    elif hasattr(t_org, "__name__") and t_org.__name__ == "Union":
        # Union[T1, T2, ...] 或 T1 | T2 检查
        return any(type_match(val, t) for t in args)

    return True
