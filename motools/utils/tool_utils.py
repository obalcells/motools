from typing import Union
from inspect_ai.tool import ToolDef as InspectToolDef

# Valid schema for HF tokenizer tools:
#  {
#   'type': 'function',
#   'function': {
#       'name': 'add',
#       'description': 'Add two integers and return the result.',
#       'parameters': {
#           'a': {'type': 'integer', 'description': 'First integer.'},
#           'b': {'type': 'integer', 'description': 'Second integer.'}
#       }
#   }
# }

def tools_as_json_schemas(tools: list[Union[dict, InspectToolDef]]):
    tools_as_schemas = []

    for tool_def in tools:
        if isinstance(tool_def, dict):
            tools_as_schemas.append(tool_def)
            continue

        if isinstance(tool_def.parameters, dict):
            parameters = tool_def.parameters
        else:
            parameters = tool_def.parameters.model_dump(exclude_none=True)

        tools_as_schemas.append({
            'type': 'function',
            'function': {
                'name': tool_def.name,
                'description': tool_def.description,
                'parameters': parameters,
            }
        })

    return tools_as_schemas

# class ToolParams(BaseModel):
#     """Description of tool parameters object in JSON Schema format."""

#     type: Literal["object"] = Field(default="object")
#     """Params type (always 'object')"""

#     properties: dict[str, ToolParam] = Field(default_factory=dict)
#     """Tool function parameters."""

#     required: list[str] = Field(default_factory=list)
#     """List of required fields."""

#     additionalProperties: bool = Field(default=False)
#     """Are additional object properties allowed? (always `False`)"""
