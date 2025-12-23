from langchain_community.tools.shell import ShellTool

tool = ShellTool()
res = tool.invoke('ls -lR')
print(res)