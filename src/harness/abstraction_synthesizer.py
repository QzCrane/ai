"""
Tier-2 Reuse Engine: DreamCoder-Style AST Anti-Unification & Cognitive Primitive Synthesizer
============================================================================================
Implements the core philosophical breakthrough of SYSTEM_CONSTITUTION.md:
- Instead of merely storing solved tasks as passive text, extracts structural commonalities
  across multiple concrete solved Python programs via AST Anti-Unification (Plotkin's LGG:
  Least General Generalization) to synthesize higher-order reusable Cognitive Primitives!
- Dramatically shrinks future search spaces: a novel OOD task matching the abstracted
  higher-order schema can be solved by instantiating 1 parameter instead of searching over
  the entire primitive AST grammar.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class SynthesizedAbstraction:
    primitive_name: str
    parameter_names: List[str]
    template_source: str
    concrete_instantiations: List[Dict[str, str]]
    compression_ratio: float


class ASTAntiUnifier:
    """
    Computes the Least General Generalization (LGG / Anti-Unification) of two or more
    Python AST expressions/functions, replacing diverging subtrees with formal meta-parameters
    while preserving shared algorithmic control flow.
    """

    def __init__(self) -> None:
        self._var_counter = 0
        self._subst_map: Dict[Tuple[str, str], str] = {}

    def _fresh_param(self, left_dump: str, right_dump: str) -> str:
        key = (left_dump, right_dump)
        if key not in self._subst_map:
            self._var_counter += 1
            self._subst_map[key] = f"p{self._var_counter}"
        return self._subst_map[key]

    def anti_unify_expr(self, node_a: ast.AST, node_b: ast.AST) -> ast.AST:
        if type(node_a) is not type(node_b):
            pname = self._fresh_param(ast.unparse(node_a), ast.unparse(node_b))
            return ast.Name(id=pname, ctx=ast.Load())

        # Identical AST subtrees stay intact
        if ast.dump(node_a) == ast.dump(node_b):
            return node_a

        # Constants (numbers, strings) that differ -> parameterize!
        if isinstance(node_a, ast.Constant) and isinstance(node_b, ast.Constant):
            pname = self._fresh_param(ast.unparse(node_a), ast.unparse(node_b))
            return ast.Name(id=pname, ctx=ast.Load())

        # Binary operations with same operator -> recurse on operands
        if isinstance(node_a, ast.BinOp) and isinstance(node_b, ast.BinOp):
            if type(node_a.op) is type(node_b.op):
                return ast.BinOp(
                    left=self.anti_unify_expr(node_a.left, node_b.left),
                    op=node_a.op,
                    right=self.anti_unify_expr(node_a.right, node_b.right),
                )
            pname = self._fresh_param(ast.unparse(node_a), ast.unparse(node_b))
            return ast.Name(id=pname, ctx=ast.Load())

        # Compare operations with same ops
        if isinstance(node_a, ast.Compare) and isinstance(node_b, ast.Compare):
            if len(node_a.ops) == len(node_b.ops) and all(
                type(o1) is type(o2) for o1, o2 in zip(node_a.ops, node_b.ops)
            ):
                return ast.Compare(
                    left=self.anti_unify_expr(node_a.left, node_b.left),
                    ops=node_a.ops,
                    comparators=[
                        self.anti_unify_expr(c1, c2)
                        for c1, c2 in zip(node_a.comparators, node_b.comparators)
                    ],
                )

        # List comprehensions with same structure (e.g. filter-map-reduce algorithmic patterns)
        if isinstance(node_a, ast.ListComp) and isinstance(node_b, ast.ListComp):
            if len(node_a.generators) == 1 and len(node_b.generators) == 1:
                g1, g2 = node_a.generators[0], node_b.generators[0]
                if len(g1.ifs) == len(g2.ifs) and ast.dump(g1.target) == ast.dump(g2.target):
                    new_elt = self.anti_unify_expr(node_a.elt, node_b.elt)
                    new_iter = self.anti_unify_expr(g1.iter, g2.iter)
                    new_ifs = [self.anti_unify_expr(i1, i2) for i1, i2 in zip(g1.ifs, g2.ifs)]
                    new_gen = ast.comprehension(
                        target=g1.target,
                        iter=new_iter,
                        ifs=new_ifs,
                        is_async=0,
                    )
                    return ast.ListComp(
                        elt=new_elt,
                        generators=[new_gen],
                    )

        # Call expressions with same callee (e.g. sum(...), max(...), sorted(...))
        if isinstance(node_a, ast.Call) and isinstance(node_b, ast.Call):
            if ast.dump(node_a.func) == ast.dump(node_b.func) and len(node_a.args) == len(node_b.args):
                return ast.Call(
                    func=node_a.func,
                    args=[self.anti_unify_expr(a1, a2) for a1, a2 in zip(node_a.args, node_b.args)],
                    keywords=[],
                )

        # Fallback: parameterize the entire diverging subtree
        pname = self._fresh_param(ast.unparse(node_a), ast.unparse(node_b))
        return ast.Name(id=pname, ctx=ast.Load())

    def synthesize_from_functions(
        self, src_a: str, src_b: str, primitive_name: str
    ) -> Optional[SynthesizedAbstraction]:
        """Extract a reusable higher-order Python primitive from two concrete solved functions."""
        tree_a = ast.parse(src_a)
        tree_b = ast.parse(src_b)
        fn_a = tree_a.body[0]
        fn_b = tree_b.body[0]
        if not (isinstance(fn_a, ast.FunctionDef) and isinstance(fn_b, ast.FunctionDef)):
            return None
        if not (isinstance(fn_a.body[-1], ast.Return) and isinstance(fn_b.body[-1], ast.Return)):
            return None

        self._var_counter = 0
        self._subst_map = {}
        gen_ret = self.anti_unify_expr(fn_a.body[-1].value, fn_b.body[-1].value)
        orig_args = [arg.arg for arg in fn_a.args.args]
        meta_params = list(self._subst_map.values())
        all_params = orig_args + meta_params
        template_src = (
            f"def {primitive_name}({', '.join(all_params)}):\n"
            f"    return {ast.unparse(gen_ret)}\n"
        )
        inst_a = {pname: k[0] for k, pname in self._subst_map.items()}
        inst_b = {pname: k[1] for k, pname in self._subst_map.items()}
        orig_len = len(src_a) + len(src_b)
        comp_ratio = round(orig_len / max(len(template_src), 1), 3)
        return SynthesizedAbstraction(
            primitive_name=primitive_name,
            parameter_names=meta_params,
            template_source=template_src,
            concrete_instantiations=[inst_a, inst_b],
            compression_ratio=comp_ratio,
        )
