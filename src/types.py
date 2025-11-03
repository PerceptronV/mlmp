"""
Type System for the Functional Programming Language

This module defines the type system and provides type representations
that match the language specification.

Type System:
    - Int: Integer type (0-99)
    - Bool: Boolean type  
    - [t1]: List type (list of t1)
    - t1 → t2: Function type (function from t1 to t2)
    - Type variables: t1, t2, etc. for polymorphism
"""

from dataclasses import dataclass
from typing import Dict, Optional, Set
from abc import ABC, abstractmethod


class Type(ABC):
    """Base class for all types."""
    
    @abstractmethod
    def __str__(self) -> str:
        """String representation of the type."""
        pass
    
    @abstractmethod
    def __eq__(self, other) -> bool:
        """Check type equality."""
        pass
    
    @abstractmethod
    def occurs_in(self, var: 'TypeVar') -> bool:
        """Check if a type variable occurs in this type."""
        pass
    
    @abstractmethod
    def substitute(self, substitutions: Dict['TypeVar', 'Type']) -> 'Type':
        """Apply type variable substitutions."""
        pass
    
    @abstractmethod
    def free_type_vars(self) -> Set['TypeVar']:
        """Get free type variables in this type."""
        pass


@dataclass(frozen=True)
class IntType(Type):
    """Integer type."""
    
    def __str__(self) -> str:
        return "Int"
    
    def __eq__(self, other) -> bool:
        return isinstance(other, IntType)
    
    def __hash__(self) -> int:
        return hash("Int")
    
    def occurs_in(self, var: 'TypeVar') -> bool:
        return False
    
    def substitute(self, substitutions: Dict['TypeVar', Type]) -> Type:
        return self
    
    def free_type_vars(self) -> Set['TypeVar']:
        return set()


@dataclass(frozen=True)
class BoolType(Type):
    """Boolean type."""
    
    def __str__(self) -> str:
        return "Bool"
    
    def __eq__(self, other) -> bool:
        return isinstance(other, BoolType)
    
    def __hash__(self) -> int:
        return hash("Bool")
    
    def occurs_in(self, var: 'TypeVar') -> bool:
        return False
    
    def substitute(self, substitutions: Dict['TypeVar', Type]) -> Type:
        return self
    
    def free_type_vars(self) -> Set['TypeVar']:
        return set()


@dataclass(frozen=True)
class ListType(Type):
    """List type: [elem_type]"""
    elem_type: Type
    
    def __str__(self) -> str:
        return f"[{self.elem_type}]"
    
    def __eq__(self, other) -> bool:
        return isinstance(other, ListType) and self.elem_type == other.elem_type
    
    def __hash__(self) -> int:
        return hash(("List", self.elem_type))
    
    def occurs_in(self, var: 'TypeVar') -> bool:
        return self.elem_type.occurs_in(var)
    
    def substitute(self, substitutions: Dict['TypeVar', Type]) -> Type:
        return ListType(self.elem_type.substitute(substitutions))
    
    def free_type_vars(self) -> Set['TypeVar']:
        return self.elem_type.free_type_vars()


@dataclass(frozen=True)
class FunctionType(Type):
    """Function type: param_type → return_type"""
    param_type: Type
    return_type: Type
    
    def __str__(self) -> str:
        # Add parentheses for nested functions
        param_str = str(self.param_type)
        if isinstance(self.param_type, FunctionType):
            param_str = f"({param_str})"
        return f"{param_str} → {self.return_type}"
    
    def __eq__(self, other) -> bool:
        return (isinstance(other, FunctionType) and
                self.param_type == other.param_type and
                self.return_type == other.return_type)
    
    def __hash__(self) -> int:
        return hash(("Function", self.param_type, self.return_type))
    
    def occurs_in(self, var: 'TypeVar') -> bool:
        return self.param_type.occurs_in(var) or self.return_type.occurs_in(var)
    
    def substitute(self, substitutions: Dict['TypeVar', Type]) -> Type:
        return FunctionType(
            self.param_type.substitute(substitutions),
            self.return_type.substitute(substitutions)
        )
    
    def free_type_vars(self) -> Set['TypeVar']:
        return self.param_type.free_type_vars() | self.return_type.free_type_vars()


@dataclass(frozen=True)
class TypeVar(Type):
    """Type variable for polymorphism: t1, t2, etc."""
    name: str
    
    def __str__(self) -> str:
        return self.name
    
    def __eq__(self, other) -> bool:
        return isinstance(other, TypeVar) and self.name == other.name
    
    def __hash__(self) -> int:
        return hash(("TypeVar", self.name))
    
    def occurs_in(self, var: 'TypeVar') -> bool:
        return self == var
    
    def substitute(self, substitutions: Dict['TypeVar', Type]) -> Type:
        if self in substitutions:
            return substitutions[self]
        return self
    
    def free_type_vars(self) -> Set['TypeVar']:
        return {self}


# Common type instances
INT = IntType()
BOOL = BoolType()


def list_of(elem_type: Type) -> ListType:
    """Create a list type."""
    return ListType(elem_type)


def func(param_type: Type, return_type: Type) -> FunctionType:
    """Create a function type."""
    return FunctionType(param_type, return_type)


class TypeScheme:
    """
    Polymorphic type scheme with quantified type variables.
    
    For example: ∀t1. t1 → t1 (identity function)
    """
    
    def __init__(self, quantified: Set[TypeVar], type_: Type):
        """
        Create a type scheme.
        
        Args:
            quantified: Set of quantified type variables
            type_: The type
        """
        self.quantified = quantified
        self.type = type_
    
    def __str__(self) -> str:
        if not self.quantified:
            return str(self.type)
        vars_str = " ".join(str(v) for v in sorted(self.quantified, key=str))
        return f"∀{vars_str}. {self.type}"
    
    def instantiate(self, fresh_vars: Dict[TypeVar, TypeVar]) -> Type:
        """
        Instantiate the type scheme with fresh type variables.
        
        Args:
            fresh_vars: Mapping from quantified vars to fresh vars
            
        Returns:
            Instantiated type
        """
        substitutions = {var: fresh_vars.get(var, var) for var in self.quantified}
        return self.type.substitute(substitutions)


class TypeError(Exception):
    """Exception raised for type errors."""
    
    def __init__(self, message: str, location: Optional[str] = None):
        self.message = message
        self.location = location
        if location:
            super().__init__(f"Type error at {location}: {message}")
        else:
            super().__init__(f"Type error: {message}")


if __name__ == "__main__":
    # Example types
    print("Type System Examples:")
    print("=" * 60)
    
    # Basic types
    print(f"Int type: {INT}")
    print(f"Bool type: {BOOL}")
    
    # List types
    int_list = list_of(INT)
    print(f"List of Int: {int_list}")
    
    nested_list = list_of(list_of(INT))
    print(f"Nested list: {nested_list}")
    
    # Function types
    int_to_int = func(INT, INT)
    print(f"Int → Int: {int_to_int}")
    
    curried = func(INT, func(INT, INT))
    print(f"Curried: {curried}")
    
    # Polymorphic types
    t1 = TypeVar("t1")
    identity = func(t1, t1)
    print(f"Identity: {identity}")
    
    map_type = func(func(t1, TypeVar("t2")), func(list_of(t1), list_of(TypeVar("t2"))))
    print(f"Map: {map_type}")
    
    # Type scheme
    scheme = TypeScheme({t1}, identity)
    print(f"Polymorphic identity: {scheme}")

