import json
import os
import sys

# Add the project root to sys.path so we can import clintrials
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from clintrials.core.schema import WinRatioSchema, CRMSchema, EffToxSchema, FieldInfo
from typing import get_args, get_origin, Annotated

def get_field_type_info(annotation, field_info):
    info = {"type": "string"}
    
    origin = get_origin(annotation)
    
    if origin is Annotated:
        args = get_args(annotation)
        base_type = args[0]
        
        # Check for our specific Annotated aliases
        is_prob = False
        is_pos_int = False
        for arg in args[1:]:
            if arg == "Probability":
                is_prob = True
            elif arg == "PositiveInt":
                is_pos_int = True
                
        if base_type is int or is_pos_int:
            info["type"] = "integer"
        elif base_type is float or is_prob:
            info["type"] = "number"
            
        if is_prob:
            info["minimum"] = 0.0
            info["maximum"] = 1.0
            
        if is_pos_int:
            info["exclusiveMinimum"] = 0
            
    elif origin in (list, tuple) or getattr(origin, "__origin__", None) in (list, tuple):
        info["type"] = "array"
        args = get_args(annotation)
        if args:
            info["items"] = get_field_type_info(args[0], FieldInfo())
    elif origin is type(None) or origin is getattr(sys.modules.get('typing'), 'Union', None): # Union / Optional
        args = get_args(annotation)
        # simplistic handling of Optional
        non_none_args = [arg for arg in args if arg is not type(None)]
        if non_none_args:
            info = get_field_type_info(non_none_args[0], field_info)
    elif annotation is int:
        info["type"] = "integer"
    elif annotation is float:
        info["type"] = "number"
    elif annotation is bool:
        info["type"] = "boolean"
        
    # Apply field info constraints
    if field_info:
        if field_info.ge is not None:
            info["minimum"] = field_info.ge
        if field_info.le is not None:
            info["maximum"] = field_info.le
        if field_info.gt is not None:
            info["exclusiveMinimum"] = field_info.gt
        if field_info.lt is not None:
            info["exclusiveMaximum"] = field_info.lt
        if field_info.description is not None:
            info["description"] = field_info.description
            
    return info

def generate_schema_for_class(cls):
    schema = {
        "type": "object",
        "properties": {},
        "required": []
    }
    
    for name, field_info in cls.model_fields.items():
        prop = get_field_type_info(field_info.annotation, field_info)
        
        import dataclasses
        if field_info.default is not dataclasses.MISSING and field_info.default is not None:
            prop["default"] = field_info.default
            
        # If it doesn't have a default and it's not Optional, it might be required
        if field_info.default is dataclasses.MISSING:
            origin = get_origin(field_info.annotation)
            is_optional = False
            if origin is getattr(sys.modules.get('typing'), 'Union', None):
                args = get_args(field_info.annotation)
                if type(None) in args:
                    is_optional = True
            if not is_optional:
                schema["required"].append(name)
                
        schema["properties"][name] = prop
        
    return schema

def main():
    schemas = {
        "WinRatioSchema": generate_schema_for_class(WinRatioSchema),
        "CRMSchema": generate_schema_for_class(CRMSchema),
        "EffToxSchema": generate_schema_for_class(EffToxSchema)
    }
    
    out_path = os.path.join(os.path.dirname(__file__), "..", "hub", "schema.json")
    with open(out_path, "w") as f:
        json.dump(schemas, f, indent=2)
        
    print(f"Serialized schemas to {out_path}")

if __name__ == "__main__":
    main()
