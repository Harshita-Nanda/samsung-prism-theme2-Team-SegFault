"""Pydantic data contract (schema.py) for the Smart Guided Troubleshooting Engine.

Identical to the official contract EXCEPT two fixes in APIResponse, both required
for the file to work on Pydantic v2:

1. `Dict[str, any]` -> `Dict[str, Any]`
   `any` is Python's builtin function, not a type. Pydantic v2 raises
   PydanticSchemaGenerationError the moment the class is defined, so the original
   file cannot even be imported.
2. `min_items` / `max_items` -> `min_length` / `max_length`
   The old names are deprecated in Pydantic v2 and will be removed.
"""
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BaseDeeplink(BaseModel):
    deeplink: str


class Deeplink(BaseDeeplink):
    description: str
    message: Optional[str] = ""
    classes: Optional[Dict[str, str]] = None
    originalType: Optional[str] = None


class Condition(str, Enum):
    greater = "greater"
    equal = "equal"
    less = "less"


class ResultTypes(str, Enum):
    boolean = "boolean"
    intNum = "integer"
    string = "str"
    floatNum = "float"


class actionCategory(str, Enum):
    auto = "auto"
    manual = "manual"
    critical = "critical"


class ValidationDeepLink(BaseDeeplink):
    key: str
    resultType: Optional[ResultTypes] = None
    condition: Optional[Condition] = None
    value: Optional[str] = None


class StepGroup(BaseModel):
    steps: List[str]
    validationDeeplink: Optional[ValidationDeepLink] = None
    actionableDeeplink: Optional[Deeplink] = None


class Action(BaseModel):
    actionName: str
    description: str
    stepGroups: List[StepGroup]
    category: Optional[actionCategory] = actionCategory.manual


class Goal(BaseModel):
    goal: str
    title: str
    actions: List[Action]
    score: float


class ContextDeeplinkResponse(BaseModel):
    """RAG response containing a list of Goal objects."""

    contexts: List[Goal] = []


class APIResponse(BaseModel):
    query: str
    query_variations: List[str] = Field(..., min_length=8, max_length=10)
    response: ContextDeeplinkResponse
    meta: Dict[str, Any]
