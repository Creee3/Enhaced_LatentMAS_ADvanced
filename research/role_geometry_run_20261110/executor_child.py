"""Pinned sandbox child with read-only interpreter runtime and preloaded NumPy."""
import ast,os,sys
from pathlib import Path
import numpy
import numpy.linalg
source=Path(__file__).resolve().parent/'dependencies/scoring/sandbox_child.py'
tree=ast.parse(source.read_text(encoding='utf-8'))
# Add only the isolated Python environment's read-only package/runtime directory.
# All original Landlock and seccomp syscall denial rules remain byte-for-byte
# AST identical except this explicit directory tuple.
class RuntimePath(ast.NodeTransformer):
 def visit_Tuple(self,node):
  if [getattr(x,'value',None) for x in node.elts]==['/usr','/lib','/lib64']:
   return ast.copy_location(ast.Tuple(elts=node.elts+[ast.Constant(value=sys.prefix)],ctx=ast.Load()),node)
  return self.generic_visit(node)
tree=RuntimePath().visit(tree);ns={'__name__':'frozen_sandbox_child','__file__':str(source)};exec(compile(ast.fix_missing_locations(tree),str(source),'exec'),ns)
ns['main']()
