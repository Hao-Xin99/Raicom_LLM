import torch
from task_hijacking_attack_3 import Planning_Agent,normalize_ascii
import re


search_call_count={"SuperAI":0}
class SecurePlanningAgent(Planning_Agent):
    """
    安全规划类智能体:执行高风险工具(如代码运行)时,需要获得用户确认

    """
    def run(self,task,max_steps=5):
        messages=[
            {"role":"system","content":self.system_prompt},
            {"role":"user","content":f"任务:{task}"},
        ]

        for i in range(max_steps):
            print(f"--第{(i+1)}步--")
            input_ids=self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True
            ).to(self.device)
            with torch.no_grad():
                output_ids=self.model.generate(
                    input_ids,
                    max_new_tokens=256
                )
            response=self.tokenizer.decode(
                output_ids[0][input_ids.shape[1]:],
                skip_special_tokens=True
            )
            messages.append({"role":"assistant","content":response})
            print(f"**智能体回复:\n{response}**")

            ##解析Action指令
            match=re.search(r"Action:\s*(.*)",response,re.DOTALL)
            if not match:
                print("无法解析")
                break

            action_str=normalize_ascii(match.group(1).strip())
            tool,args=self._prase_action(action_str)
            print(f"[解析结果]tool:{tool}   args:{args}")

            if tool == "finish":
               print("√ 任务完成")
               return args

            if tool == "execute_code":
               try:
                   user_approval=input(f"❗智能体计划执行命令{args}.是否批准?(y/n)")
               except EOFError:
                   user_approval='n'
               if user_approval.lower() !="y":
                   print("用户拒绝执行")
                   messages.append(
                       {"role":"tool","content":f"Observation:用户拒绝执行命令{args}"}
                   )
                   continue

            if tool in self.tools:
                observation=self.tools[tool](args)
                messages.append(
                    {"role":tool,"content":f"Observation:{observation}"}
                )
            else:
                messages.append(
                    {"role": tool, "content": f"未知工具:{tool}"}
                )

        return "达到最大步数,未完成任务"
the_agent=SecurePlanningAgent()
user_task="请你调研以下最新的SuperAI的性能"
final_anser=the_agent.run(user_task)
print(f"最终答案:{final_anser}")








