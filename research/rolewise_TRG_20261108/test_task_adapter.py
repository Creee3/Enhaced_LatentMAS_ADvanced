import sys,unittest,runpy,os
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE=Path(os.environ.get('ROLEWISE_FROZEN_SOURCE',str(HERE.parents[2]/'advanced_MOC/experiments/latentmas_rolewise_TRG576_20261108/frozen_source_reference')))
sys.path.insert(0,str(HERE));from task_adapter import messages,args_for,parse_answer,canonical_number,normalized_gold
BUILD=runpy.run_path(str(SOURCE/'vendor_prompts.py'))['build_agent_message_sequential_latent_mas']
NATIVE=runpy.run_path(str(SOURCE/'author_parser.py'))['extract_gsm8k_answer']
class TaskAdapters(unittest.TestCase):
 def test_gpqa_and_gsm_templates_byte_exact_to_author(self):
  for ds in ['gpqa','gsm8k']:
   item=dict(dataset=ds,question='Synthetic fixture only.',n_options=4)
   for role in ['planner','critic','refiner','judger']:
    original=BUILD(role=role,question=item['question'],context='',method='latent_mas',args=args_for(ds,'Qwen3-8B'))
    self.assertEqual(messages(role,item,BUILD,'Qwen3-8B'),original)
 def test_mmlu_range_only_changes_boilerplate(self):
  item=dict(dataset='mmlu_pro',question='Literal Your final answer must be selected from A,B,C,D. in synthetic content.',n_options=10)
  for role in ['planner','critic','refiner','judger']:
   result=messages(role,item,BUILD,'Qwen3-8B');self.assertIn(item['question'],result[1]['content'])
   if role=='judger':self.assertEqual(result[1]['content'].count('selected from A,B,C,D,E,F,G,H,I,J.'),1)
 def test_choices_keep_frozen_parser_and_valid_range(self):
  for ds,n in [('gpqa',4),('mmlu_pro',10)]:
   item=dict(dataset=ds,n_options=n)
   for i in range(n):self.assertEqual(parse_answer(item,'Final '+r'\boxed{'+chr(65+i)+'}',NATIVE),chr(65+i))
   self.assertIsNone(parse_answer(item,r'\boxed{K}',NATIVE))
 def test_gsm_numeric_canonicalization_and_nested_box(self):
  item=dict(dataset='gsm8k',gold='1,200')
  fixtures={r'\boxed{1,200}':'1200',r'\boxed{12.0}':'12',r'\boxed{\frac{3}{4}}':'3/4',r'\boxed{-2.50}':'-5/2',r'work 17. final 1,200':'1200','#### 0.5':'1/2',r'\boxed{3 or 4}':None,r'\boxed{\frac{1}{0}}':None}
  for text,want in fixtures.items():self.assertEqual(parse_answer(item,text,NATIVE),want)
  self.assertEqual(normalized_gold(item),'1200');self.assertIsNone(canonical_number('not a number'))
if __name__=='__main__':unittest.main()
