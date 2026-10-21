import re
def extract_solution(solution_str):
    match=re.search(r"####\s*(-?[\d\.,]+)",solution_str)
    if match is None:
        raise ValueError("未能在文本中找到合适的数值解")
    final_solution=match.group(1).replace(",","")
    return final_solution



