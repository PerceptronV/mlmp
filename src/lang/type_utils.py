import inspect
from typing import get_type_hints, get_origin, get_args
from typing import TypeVar, Callable, Union, Optional


CallableOrig = get_origin(Callable)
type TypeType = Union[TypeVar, type]

def analyse_function_types(
    fn: Callable
) -> tuple[tuple[str], tuple[TypeType], TypeType]:

    var_types = get_type_hints(fn)
    signature = inspect.signature(fn)
    arg_names = tuple(p.name for p in signature.parameters.values())

    try:
        arg_types = tuple(var_types[a] for a in arg_names)
    except KeyError:
        raise TypeError(f"Missing type hint for arguments ({
            ', '.join(a for a in arg_names if a not in var_types)
        }) of {fn.__name__}")

    try:
        ret_type = var_types['return']
    except KeyError:
        raise TypeError(f"Missing type hint for return value of {fn.__name__}")
    
    return arg_names, arg_types, ret_type

def get_base_type(type_: TypeType) -> TypeType:
    origin = get_origin(type_)
    return origin if origin is not None else type_

def isvariable(type_: TypeType) -> bool:
    return isinstance(type_, TypeVar)

def isatomic(type_: TypeType) -> bool:
    return not (isvariable(type_) or has_args(type_))

def has_args(type_: TypeType) -> bool:
    return len(get_args(type_)) > 0


class SubstitutionTable:
    def __init__(self, substitutions: Optional[dict[TypeVar, TypeType]] = None, copy: bool = True):
        if substitutions is None:
            self.table = {}
        elif copy:
            self.table = substitutions.copy()
        else:
            self.table = substitutions
    
    def __str__(self) -> str:
        return str(self.table)
    
    def __repr__(self) -> str:
        return f"SubstitutionTable({self.table})"

    def __getitem__(self, type_key: TypeType) -> TypeType:
        type_value = self.table.get(type_key, type_key)
        # loop until fixed point
        if type_value != type_key:
            type_value = self.__getitem__(type_value)
            # cache result for faster lookup
            self.table[type_key] = type_value
        return type_value
    
    def compatible(self, type1: TypeType, type2: TypeType) -> bool:
        type1 = self.__getitem__(type1)
        type2 = self.__getitem__(type2)
        if (not isvariable(type1)) and (not isvariable(type2)):
            return type1 == type2
        else:
            return True

    def __setitem__(self, type_key: TypeType, type_value: TypeType):
        if not isvariable(type_key):
            raise TypeError(f"Cannot substitute a non-variable: {type_key}")
        
        k = self.__getitem__(type_key)   # current substitution for type_key
        v = self.__getitem__(type_value) # current substitution for type_value

        if not self.compatible(k, v):
            raise TypeError(f"Incompatible types: {type_key} -> {k} and {type_value} -> {v}")

        self.table[type_key] = v
    
    def link(self, type1: TypeType, type2: TypeType):
        if not self.compatible(type1, type2):
            raise TypeError(f"Incompatible types: {type1} and {type2}")
        
        v1 = self.__getitem__(type1)
        v2 = self.__getitem__(type2)
        
        if isvariable(v1):
            self.table[type1] = v2
        elif isvariable(v2):
            self.table[type2] = v1
        elif v1 != v2:
            raise TypeError(f"Cannot link different non-variables: {type1} -> {v1} != {type2} -> {v2}")
    
    def copy(self) -> 'SubstitutionTable':
        return SubstitutionTable(self.table)

    def update(self, other: 'SubstitutionTable'):
        """
        Safely update this substitution table with entries from another.

        Uses the existing link mechanism to ensure type compatibility
        when merging substitution tables.

        Args:
            other: Another SubstitutionTable to merge into this one
        """
        for type_key, type_value in other.table.items():
            if type_key in self.table:
                # If key already exists, link the values to ensure compatibility
                self.link(self.table[type_key], type_value)
            else:
                # New key, add directly
                self.table[type_key] = type_value


def substitute_type_vars(
    type_: TypeType,
    substitutions: SubstitutionTable
) -> TypeType:
    # If type_ is a Python list (happens with Callable parameter lists), we can't process it
    # This should not be called directly on such lists
    if isinstance(type_, list):
        raise TypeError(f"substitute_type_vars called on Python list: {type_}. This is likely a bug.")

    base_type = substitutions[get_base_type(type_)]
    args = get_args(type_)
    if len(args) > 0:
        # Special handling for Callable: args are ([arg_types...], return_type)
        # where the first element is a list, not a type
        if base_type == CallableOrig and isinstance(args[0], list):
            param_types = [substitute_type_vars(a, substitutions) for a in args[0]]
            return_type = substitute_type_vars(args[1], substitutions)
            return base_type[[*param_types], return_type]
        else:
            new_args = [substitute_type_vars(a, substitutions) for a in args]
            return base_type[*new_args]
    else:
        return base_type

def separate_type(
    type_: TypeType,
    substitutions: SubstitutionTable
) -> tuple[TypeType, list[TypeType]]:
    type_value = substitute_type_vars(type_, substitutions)
    base_type = get_base_type(type_value)
    args = get_args(type_value)
    return base_type, args

def matchable(
    type1: TypeType,
    type2: TypeType,
    substitutions: SubstitutionTable,
    update: bool = True, # whether to update the substitutions table
    strict: bool = True # whether to enforce argument matching
) -> bool:

    if not update:
        substitutions = substitutions.copy()

    # Get fully resolved types and their components
    resolved1 = substitute_type_vars(type1, substitutions)
    resolved2 = substitute_type_vars(type2, substitutions)
    base_type1, args1 = get_base_type(resolved1), get_args(resolved1)
    base_type2, args2 = get_base_type(resolved2), get_args(resolved2)

    # In non-strict mode, when one side is a type variable, link it to the
    # full resolved type of the other side (preserving type arguments)
    if not strict:
        if isvariable(base_type1):
            try:
                substitutions[base_type1] = resolved2
                return True
            except TypeError:
                return False

        if isvariable(base_type2):
            try:
                substitutions[base_type2] = resolved1
                return True
            except TypeError:
                return False

    if not substitutions.compatible(base_type1, base_type2):
        return False

    # In non-strict mode, skip arg length checking
    # This allows more flexible unification for type inference
    if strict and len(args1) != len(args2):
        return False

    # Handle Callable specially: args are ([param_types...], return_type)
    if base_type1 == CallableOrig and base_type2 == CallableOrig:
        if len(args1) != 2 or len(args2) != 2:
            return False
        # Match parameter lists
        param_list1, ret1 = args1
        param_list2, ret2 = args2
        if not isinstance(param_list1, list) or not isinstance(param_list2, list):
            return False
        if len(param_list1) != len(param_list2):
            return False
        for p1, p2 in zip(param_list1, param_list2):
            if not matchable(p1, p2, substitutions, strict=strict):
                return False
        # Match return types
        if not matchable(ret1, ret2, substitutions, strict=strict):
            return False
    else:
        # Regular type matching
        for a1, a2 in zip(args1, args2):
            if not matchable(a1, a2, substitutions, strict=strict):
                return False

    try:
        substitutions.link(base_type1, base_type2)
        return True
    except TypeError as e:
        print(e)
        return False
    
def get_free_types(type_: TypeType, substitutions: SubstitutionTable):
    type_ = substitute_type_vars(type_, substitutions)
    frees = set()

    def _check(t):
        base = get_base_type(t)
        args = get_args(t)
        if isvariable(base):
            frees.update([base])
        # Handle Callable specially: args are ([param_types...], return_type)
        if base == CallableOrig:
            assert len(args) == 2 and isinstance(args[0], list), f"Callable must have exactly 2 elements: {args}"
            for param_type in args[0]:
                _check(param_type)
            _check(args[1])  # return type
        else:
            for a in args:
                _check(a)

    _check(type_)
    return frees


if __name__ == "__main__":
    t1 = TypeVar("t1")
    t2 = TypeVar("t2")
    t3 = TypeVar("t3")

    subs = SubstitutionTable()
    print(matchable(int, t1, subs), subs)

    print(matchable(list[t1], list[t2], subs), subs)

    print(matchable(t2, float, subs), subs)

    print(get_free_types(list[t1, t2, t3], subs))

    print(matchable(t3, list[float], subs), subs)
