"""Parallelize only the pinned exporter's four independent OpenCV operations."""

import ast
from concurrent.futures import ThreadPoolExecutor
from functools import update_wrapper
import inspect
import textwrap


INPAINT_BLOCK = """
base_color = cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)
metallic = cv2.inpaint(metallic, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
roughness = cv2.inpaint(roughness, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
alpha = cv2.inpaint(alpha, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
"""


def inpaint_batch(base_color, metallic, roughness, alpha, mask, workers=1):
    import cv2

    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("inpaint workers must be an integer between 1 and 8")
    arguments = [(base_color, mask, 3, cv2.INPAINT_TELEA),
                 *[(channel, mask, 1, cv2.INPAINT_TELEA) for channel in (metallic, roughness, alpha)]]
    if workers == 1:
        results = [cv2.inpaint(*values) for values in arguments]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, 4)) as executor:
            results = list(executor.map(lambda values: cv2.inpaint(*values), arguments))
    return (results[0], *[result[..., None] for result in results[1:]])


def parallel_exporter(original, workers=1):
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("inpaint workers must be an integer between 1 and 8")
    if workers == 1:
        return original
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    expected = [ast.dump(node, include_attributes=False) for node in ast.parse(INPAINT_BLOCK).body]
    matches = [index for index in range(len(function.body) - 3)
               if [ast.dump(node, include_attributes=False) for node in function.body[index:index+4]] == expected]
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
             and node.func.value.id == "cv2" and node.func.attr == "inpaint"]
    if len(matches) != 1 or len(calls) != 4 or function.decorator_list:
        raise ValueError("Pinned native inpaint block changed; refusing exporter adaptation")
    replacement = ast.parse("base_color, metallic, roughness, alpha = _pixal_inpaint_batch(base_color, metallic, roughness, alpha, mask_inv)").body
    function.body[matches[0]:matches[0]+4] = replacement
    namespace = dict(original.__globals__)
    namespace["_pixal_inpaint_batch"] = lambda *arguments: inpaint_batch(*arguments, workers=workers)
    exec(compile(ast.fix_missing_locations(tree), inspect.getsourcefile(original), "exec"), namespace)
    result = namespace[function.name]
    result.__defaults__ = original.__defaults__
    return update_wrapper(result, original)
