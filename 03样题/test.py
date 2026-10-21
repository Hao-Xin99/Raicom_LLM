
'''

######  1. input()函数  ######
the_get=input("请输入一个问题:")
print(f"用户输入:\n{the_get}")


######  2. argparse库(参数解析)的使用  ######
import argparse

# 1) 创建解析器；description 会出现在 -h 帮助的顶部
parser = argparse.ArgumentParser(description="一个演示 argparse 的小脚本")

# 2) 位置参数：不带横杠，按顺序(在代码里出现的顺寻)写在命令后面
parser.add_argument('name', help="你的名字")

# 3) 选项参数：带默认值，不传时用 default
parser.add_argument("--n", type=int, default=5, help="重复次数,默认5",metavar='cishu',dest='次数')   #type指定传入类型,默认str,metavar:占位符名称,dest:Namespace对象取n时的名字

# 4) 短、长两种写法都注册，指向同一个值
parser.add_argument("-i", '--id', type=int, default=0, help="你的编号")

# 5) 解析命令行，得到一个 Namespace 对象
args = parser.parse_args()

# 6) 用 args.字段 取值,值的名称默认等于-后的字母
for _ in range(args.次数):
    print(f"你好 {args.name}，你的编号是 {args.id}")


###### JSON文本字符串 ####
import json

# 1) 纯 JSON 对象 → 成功
print('1:', json.loads('{"steps": [1, 2]}'))

# 2) 纯 JSON 数组也算 JSON（不一定要是"字典",''里可以是列表,字符串）(里面字典的键,字符串都要用""包裹)
print('2.1:', json.loads('[1, 2, 3]'))
print('2.2', json.loads(' "test" '))
# 3) 前后带自然语言 → 直接报错（不是"含有 JSON 就行"）
for bad in ['答案是：{"a": 1}', '{"a": 1} 谢谢']:
    try:
        json.loads(bad)
    except Exception as e:
        print('3 报错:', type(e).__name__, '|', bad)

# 4) 若真有前后缀，需先把 JSON 部分截取出来，再 loads
raw = '好的，结果如下：{"a": 1}，谢谢'
s = raw[raw.index('{'): raw.rindex('}') + 1]
print('4 截取后 loads:', json.loads(s))

'''
import torch
print(torch.cuda.get_device_name())