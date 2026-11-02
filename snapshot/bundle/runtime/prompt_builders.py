from pathlib import Path
import runpy
SRC=Path(__file__).resolve().parent/'vendor'/'LatentMAS'/'prompts.py'
ns=runpy.run_path(str(SRC))
# Exact author prompts are loaded dynamically at runtime. These are only the two new repeated-role prompts.
C2 = """Question: {question}\n\nYou are a second Critic Agent. The preceding Planner, Critic, and Refiner have produced a current plan for solving the question. Evaluate the latest Refiner plan for remaining factual errors, unsupported steps, or omissions. Focus your feedback on the latest Refiner plan, and provide concrete corrections for the next Refiner.\n\nNow, output your critique below:\n"""
R2 = """Question: {question}\n\nYou are a second Refiner Agent. You are provided with the latest Refiner plan and the second Critic's feedback in latent-format information. Revise the latest plan into a correct, concise step-by-step plan for solving the question. Do not produce the final answer.\n\nNow, output your revised plan below:\n"""
(Path(__file__).resolve().parent/'new_round3_prompts.json').write_text(__import__('json').dumps({'C2':C2,'R2':R2},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('author_prompt_functions', [k for k in ns if k.startswith('build_agent')])
