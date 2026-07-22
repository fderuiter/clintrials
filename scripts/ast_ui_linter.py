# ruff: noqa: D100, D103, T201
import argparse
import ast
import sys

RESTRICTED_WIDGETS = {
    "button", "download_button", "link_button", "page_link", "checkbox", "toggle",
    "radio", "selectbox", "multiselect", "slider", "select_slider", "text_input",
    "number_input", "text_area", "date_input", "time_input", "file_uploader",
    "camera_input", "color_picker", "chat_input", "data_editor"
}

def get_base_name_and_attr(node):
    if isinstance(node, ast.Name):
        return node.id, None
    elif isinstance(node, ast.Attribute):
        base, _ = get_base_name_and_attr(node.value)
        return base, node.attr
    return None, None

def check_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return False

    try:
        tree = ast.parse(content, filename=filepath)
    except SyntaxError as e:
        print(f"Syntax error in {filepath}: {e}")
        return False

    st_aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == 'streamlit' or alias.name.startswith('streamlit.'):
                    st_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith('streamlit'):
                for alias in node.names:
                    st_aliases.add(alias.asname or alias.name)

    if not st_aliases:
        # Streamlit not imported, probably fine, but what if they use specific imports?
        # Actually, if we handled ImportFrom, they are in st_aliases. Wait, if they do `from streamlit import button as b`, then `b` is in st_aliases. But they would use it as `b()`, which is a Name, not Attribute.
        pass

    errors = []

    # Check for direct function calls if imported via ImportFrom
    # Also check Attribute calls
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                base, attr = get_base_name_and_attr(node.func)
                if base in st_aliases and attr in RESTRICTED_WIDGETS:
                    errors.append((node.lineno, attr))
            elif isinstance(node.func, ast.Name):
                # if they imported a restricted widget directly: from streamlit import button
                name = node.func.id
                if name in st_aliases and name in RESTRICTED_WIDGETS:
                    errors.append((node.lineno, name))

    if errors:
        for lineno, attr in errors:
            print(f"{filepath}:{lineno} - Prohibited direct widget attribute call '{attr}' detected. Use centralized widget factory.")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="AST-based UI Linter for Streamlit")
    parser.add_argument('files', nargs='+', help='Files to lint')
    args = parser.parse_args()

    all_passed = True
    for filepath in args.files:
        if not check_file(filepath):
            all_passed = False

    if not all_passed:
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main()
