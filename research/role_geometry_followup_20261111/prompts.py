"""Natural role-task interventions; no generated answer or gold dependence."""
ROLES=('planner','critic','refiner')
ACTIONS={
 'planner':('planning a solution','constructing a plan','developing a solution plan','formulating a solution plan'),
 'critic':('checking a solution','reviewing a plan','evaluating a solution plan','examining a solution plan'),
 'refiner':('refining a solution','improving a plan','revising a solution plan','correcting a solution plan'),
}
SCAFFOLD='Your assigned task is {action}. Use the available problem and prior reasoning to carry out this task. Do not provide the final answer.'
def intervention_messages(item,instruction):
 return [dict(role='system',content='You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'),dict(role='user',content=instruction+'\n\nQuestion:\n'+item['question']+'\n\nCarry out your assigned task in latent space.')]
def instruction_candidates():
 return {r:[SCAFFOLD.format(action=a) for a in ACTIONS[r]] for r in ROLES}
