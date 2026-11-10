"""Explicit HE/32 official-oracle parameter-order overlay; frozen HF stays intact."""
from __future__ import annotations
import ast
import hashlib
import json
import math
from pathlib import Path

TASK="HumanEval/32"
RAW_TEST_SHA="b4eccf838d1b0f79daeb40e130ae3e37ee9d75ca739be3dbe93b8caac886c760"
RENDERED_TEST_SHA="3e56952a3bc59fda0597ae7d8d33943ec72bc052a26a8fa8b3209c3712e81205"
COMMIT="26d6d00bb1fd0fa37f39c99d5290da67891d1c5e"
SOURCE_SHAS={"evalplus/eval/__init__.py":"3c833c39b842e33f251c83db4347e0a95191909f23b390c12d73ab29a28a4daf",
             "evalplus/eval/_special_oracle.py":"92b126b907ee493121b55de06f6a34058b6e18adc8cf1c48737eedcd83f24cdd"}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inputs_from_test(test):
    tree=ast.parse(test)
    check=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="check")
    assigned=[node for node in check.body if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=="inputs" for target in node.targets)]
    if len(assigned)!=1:raise ValueError("HE32 inputs assignment changed")
    inputs=ast.literal_eval(assigned[0].value)
    if len(inputs)!=886:raise ValueError("HE32 input population changed")
    return inputs


def overlay_test(test):
    digest=hashlib.sha256(test.encode("utf-8")).hexdigest()
    if digest!=RENDERED_TEST_SHA:return test,dict(applied=False,patch_count=0)
    tree=ast.parse(test);before_inputs=inputs_from_test(test);patches=0
    for node in ast.walk(tree):
        if not (isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=="_poly"):continue
        if len(node.args)!=2 or not isinstance(node.args[0],ast.Starred) or not isinstance(node.args[0].value,ast.Call):continue
        candidate=node.args[0].value
        if not (isinstance(candidate.func,ast.Name) and candidate.func.id=="find_zero" and len(candidate.args)==1 and isinstance(candidate.args[0],ast.Starred) and isinstance(candidate.args[0].value,ast.Name) and candidate.args[0].value.id=="inp"):continue
        if not isinstance(node.args[1],ast.Name) or node.args[1].id!="inp":raise ValueError("HE32 malformed oracle AST changed")
        node.args=[ast.Starred(value=ast.Name(id="inp",ctx=ast.Load()),ctx=ast.Load()),candidate];patches+=1
    if patches!=1:raise ValueError("HE32 explicit oracle overlay must change exactly one call")
    ast.fix_missing_locations(tree);updated=ast.unparse(tree)+"\n"
    if inputs_from_test(updated)!=before_inputs:raise ValueError("HE32 input corpus changed")
    assertions=[node for node in ast.walk(tree) if isinstance(node,ast.Assert)]
    if len(assertions)!=1 or not isinstance(assertions[0].test,ast.Compare) or not isinstance(assertions[0].test.ops[0],ast.LtE) or assertions[0].test.comparators[0].value!=0.0001:raise ValueError("HE32 official threshold changed")
    return updated,dict(applied=True,task_id=TASK,patch_count=patches,original_test_sha256=digest,
        overlay_test_sha256=hashlib.sha256(updated.encode()).hexdigest(),input_count=886,threshold=1e-4,
        official_predicate="_poly(*inp,out)<=atol",absolute_value_added=False,source_commit=COMMIT)


def verify_lock(root):
    root=Path(root);lock=json.loads((root/"evaluator_overlay_lock.json").read_text(encoding="utf-8"))
    if lock["rendered_test_sha256"]!=RENDERED_TEST_SHA or lock["raw_test_sha256"]!=RAW_TEST_SHA or lock["input_count"]!=886 or lock["threshold"]!=1e-4 or lock["expected_patch_count"]!=1 or lock["absolute_value_added"]:raise ValueError("overlay policy lock mismatch")
    for name,digest in SOURCE_SHAS.items():
        reference=lock["official_sources"][name]
        if reference["commit"]!=COMMIT or reference["sha256"]!=digest or sha(root/reference["path"])!=digest:raise ValueError("official predicate source SHA mismatch")
    witness=lock["witness"]
    if sha(root/witness["path"])!=witness["sha256"]:raise ValueError("independent witness SHA mismatch")
    return lock


def make_executor(executor,root):
    verify_lock(root)
    def wrapped(code,test="",timeout=10,**kwargs):
        effective,receipt=overlay_test(test)
        result=dict(executor(code,effective,timeout,**kwargs));result["evaluator_overlay"]=receipt
        return result
    return wrapped


def witness_code(root):
    root=Path(root);lock=verify_lock(root);witness=json.loads((root/lock["witness"]["path"]).read_text(encoding="utf-8"))
    if witness["task_id"]!=TASK or witness["inputs"]!=886 or witness["precision_digits"]!=100:raise ValueError("witness schema mismatch")
    values={tuple(row["coefficients"]):row["root_float"] for row in witness["records"]}
    if len(values)!=886:raise ValueError("witness input uniqueness mismatch")
    # This fixture is used only in evaluator preflight. Model scoring never sees it.
    return "def find_zero(xs):\n    values="+repr(values)+"\n    return values[tuple(xs)]\n"


def prepare_witness(private_root,code_root):
    """Offline trusted numerical fixture, never model generation or score repair."""
    import mpmath as mp
    import numpy as np
    mp.mp.dps=100
    private_root=Path(private_root);code_root=Path(code_root)
    reference=json.loads((private_root/"dependencies/humanevalplus_tests.json").read_text(encoding="utf-8"))[TASK]
    if reference["raw_test_sha256"]!=RAW_TEST_SHA or reference["rendered_test_sha256"]!=RENDERED_TEST_SHA:raise ValueError("HF test SHA changed")
    inputs=inputs_from_test(reference["raw_test"])
    def value(cs,x):
        y=mp.mpf(0)
        for c in reversed(cs):y=y*x+mp.mpf(c)
        return y
    records=[];metadata=[]
    for index,arguments in enumerate(inputs):
        cs=list(arguments[0])
        while len(cs)>1 and cs[-1]==0:cs=cs[:-1]
        if len(cs)==1 or cs[0]==0:
            if len(cs)==1 and cs[0]!=0:raise ValueError("constant nonzero polynomial has no witness")
            low=high=mp.mpf(0);algorithm="exact_zero_root"
        else:
            bound=mp.mpf(1)+max(abs(mp.mpf(c)/cs[-1]) for c in cs[:-1]);low=-bound;high=bound;flow=value(cs,low);fhigh=value(cs,high);algorithm="Cauchy_bracket_bisection400"
            if flow*fhigh>0:
                roots=np.polynomial.polynomial.polyroots(cs);bracket=None
                for guess in sorted((float(z.real) for z in roots if abs(z.imag)<=1e-6*max(1,abs(z.real))),key=abs):
                    center=mp.mpf(guess);delta=mp.mpf("1e-12")*max(1,abs(center))
                    for _ in range(50):
                        a=center-delta;b=center+delta;fa=value(cs,a);fb=value(cs,b)
                        if fa*fb<=0:bracket=(a,b,fa,fb);break
                        delta*=2
                    if bracket:break
                if bracket is None:
                    refined=[]
                    for estimate in sorted(roots,key=lambda z:(abs(z.imag),abs(z.real))):
                        center=mp.mpf(float(estimate.real))
                        try:
                            found=mp.findroot(lambda x:value(cs,x),(center-mp.mpf("1e-7"),center+mp.mpf("1e-7")),solver="secant",tol=mp.mpf("1e-80"),maxsteps=1000,verify=True)
                            if abs(mp.im(found))<mp.mpf("1e-70") and abs(value(cs,mp.re(found)))<mp.mpf("1e-70"):refined.append(mp.re(found))
                        except (ValueError,ZeroDivisionError):pass
                    if not refined:raise ValueError("no independent verified real root at input index="+str(index))
                    low=high=min(refined,key=lambda x:abs(value(cs,x)));flow=fhigh=value(cs,low);algorithm="independent_numpy_location_mpmath100_verified_multiple_root"
                else:
                    low,high,flow,fhigh=bracket;algorithm="independent_numpy_location_mpmath100_bisection400"
            for _ in range(400):
                middle=(low+high)/2;fm=value(cs,middle)
                if fm==0:low=high=middle;break
                if (flow<0)==(fm<0):low=middle;flow=fm
                else:high=middle
        root=(low+high)/2;candidate=float(root);options=[candidate];lower=candidate;upper=candidate
        for _ in range(64):
            lower=math.nextafter(lower,-math.inf);upper=math.nextafter(upper,math.inf);options.extend((lower,upper))
        good=[]
        for x in options:
            try:residual=sum(c*math.pow(x,i) for i,c in enumerate(cs))
            except OverflowError:continue
            if math.isfinite(residual) and residual<=1e-4:good.append((abs(mp.mpf(x)-root),abs(residual),x,residual))
        if not good:raise ValueError("no nearest float satisfying unchanged official predicate")
        chosen=min(good,key=lambda x:(x[0],x[1]));records.append(dict(coefficients=arguments[0],root_float=chosen[2]));metadata.append(dict(index=index,algorithm=algorithm,mp_root_abs_residual=str(abs(value(cs,root))),mp_bracket_width=str(high-low),float_residual=chosen[3]))
    witness=dict(task_id=TASK,algorithm="independent_mpmath100_real_root_nearest_float_official_predicate",precision_digits=100,inputs=886,records=records,metadata=metadata,model_scoring_modified=False)
    destination=private_root/"dependencies/evaluator_overlay";destination.mkdir(parents=True,exist_ok=True);path=destination/"humaneval32_root_witness.json";path.write_text(json.dumps(witness,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    refs={}
    for name,digest in SOURCE_SHAS.items():
        relative="dependencies/evaluator_overlay/"+Path(name).name;p=private_root/relative
        if sha(p)!=digest:raise ValueError("official source dependency missing")
        refs[name]=dict(commit=COMMIT,path=relative,sha256=digest,bytes=p.stat().st_size)
    lock=dict(task_id=TASK,raw_test_sha256=RAW_TEST_SHA,rendered_test_sha256=RENDERED_TEST_SHA,tests_file_sha256=sha(private_root/"dependencies/humanevalplus_tests.json"),source="official EvalPlus predicate parameter-order-only overlay",official_sources=refs,expected_patch_count=1,input_count=886,threshold=1e-4,absolute_value_added=False,original_source_modified=False,witness=dict(path="dependencies/evaluator_overlay/humaneval32_root_witness.json",sha256=sha(path),bytes=path.stat().st_size,precision_digits=100,inputs=886),known_original_canonical_numerical_failure=True)
    (code_root/"evaluator_overlay_lock.json").write_text(json.dumps(lock,indent=2)+"\n",encoding="utf-8")
    return lock
