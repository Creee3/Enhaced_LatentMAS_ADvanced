"""Frozen per-dataset prompt adapter and programmatic answer normalization."""
import re
from fractions import Fraction
from types import SimpleNamespace

def args_for(dataset,model_name,method='latent_mas'):
 return SimpleNamespace(task='gsm8k' if dataset=='gsm8k' else 'gpqa',model_name=model_name,prompt='sequential',think=False,method=method,text_mas_context_length=-1,latent_space_realign=False)

def messages(role,item,build,model_name):
 args=args_for(item['dataset'],model_name)
 if item['dataset']!='mmlu_pro':return build(role=role,question=item['question'],context='',method='latent_mas',args=args)
 sentinel='__ROLEWISE_EXACT_QUESTION_SLOT__';assert sentinel not in item['question']
 result=build(role=role,question=sentinel,context='',method='latent_mas',args=args)
 if role=='judger':
  old='Your final answer must be selected from A,B,C,D.';new='Your final answer must be selected from '+','.join(chr(65+i) for i in range(item['n_options']))+'.'
  assert sum(x['content'].count(old) for x in result)==1
  result=[dict(x,content=x['content'].replace(old,new)) for x in result]
 return [dict(x,content=x['content'].replace(sentinel,item['question'])) for x in result]

def last_box(text):
 matches=list(re.finditer(r'\\boxed\s*\{',text))
 if not matches:return None
 start=matches[-1].end();depth=1
 for i in range(start,len(text)):
  if text[i]=='{':depth+=1
  elif text[i]=='}':
   depth-=1
   if depth==0:return text[start:i]
 return ''

NUMBER=r'[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/[-+]?\d+(?:\.\d+)?)?'
def canonical_number(value):
 if value is None:return None
 s=str(value).strip().replace('−','-').replace('\\,','').replace('\\$','').strip('$£€ ')
 s=re.sub(r'\\(?:text|mathrm)\{[^{}]*\}','',s).strip().rstrip('%').strip()
 fraction=re.fullmatch(r'\\(?:dfrac|tfrac|frac)\{([-+]?\d+(?:\.\d+)?)\}\{([-+]?\d+(?:\.\d+)?)\}',s)
 if fraction:s=fraction[1]+'/'+fraction[2]
 if not re.fullmatch(NUMBER,s):return None
 try:
  n=Fraction(s.replace(',',''));return str(n.numerator) if n.denominator==1 else str(n.numerator)+'/'+str(n.denominator)
 except (ValueError,ZeroDivisionError):return None

def parse_answer(item,text,native_extract):
 if item['dataset'] in ['gpqa','mmlu_pro']:
  raw=native_extract(text);pred=None if raw is None else raw.strip().upper()
  return pred if pred in [chr(65+i) for i in range(item['n_options'])] else None
 assert item['dataset']=='gsm8k'
 boxed=last_box(text)
 if boxed is not None:return canonical_number(boxed)
 marked=re.findall(r'####\s*('+NUMBER+r')',text)
 if marked:return canonical_number(marked[-1])
 nums=re.findall(NUMBER,text)
 return canonical_number(nums[-1]) if nums else None

def normalized_gold(item):
 return canonical_number(item['gold']) if item['dataset']=='gsm8k' else str(item['gold']).strip().upper()
