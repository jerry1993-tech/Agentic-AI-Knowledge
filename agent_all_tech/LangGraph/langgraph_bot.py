import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# 1. 加载环境变量
load_dotenv()

# 2. 创建支持工具调用的 LLM
llm = ChatOpenAI(
    model="qwen-plus-latest",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    streaming=True,
)

# 3. 定义工具
@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气（模拟数据）"""
    # 模拟天气数据
    weather_data = {
        "北京": "晴，温度 25°C，湿度 40%",
        "上海": "多云，温度 28°C，湿度 65%",
        "广州": "小雨，温度 30°C，湿度 80%",
        "深圳": "晴，温度 32°C，湿度 70%",
        "杭州": "阴，温度 26°C，湿度 55%",
    }
    return weather_data.get(city, f"抱歉，未找到城市 '{city}' 的天气信息。支持的城市：{', '.join(weather_data.keys())}")

@tool
def calculator(expression: str) -> str:
    """计算数学表达式。例如：2 + 2, 10 * 5, 100 / 4"""
    try:
        # 简单的安全计算
        result = eval(expression, {"__builtins__": {}}, {})
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算错误：{str(e)}"

@tool
def search_web(query: str) -> str:
    """模拟联网搜索（返回示例结果）"""
    return f"搜索结果：关于 '{query}'，这是一个模拟的搜索结果。在实际应用中，这里会返回真实的网络搜索内容。"

# 工具列表
tools = [calculator, get_weather, search_web]
# 将工具绑定到 LLM
llm_with_tools = llm.bind_tools(tools)

# 4. 定义状态
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# 5. 定义节点
def chatbot(state: AgentState):
    """聊天机器人节点：根据消息决定是直接回复还是调用工具"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def tool_router(state: AgentState) -> Literal["tools", "end"]:
    """定义路由判断节点（too_router）"""
    # 获取最新一条消息（LLM的思考结果）
    last_message = state["messages"][-1]
    # 关键判断：如果最新消息包含tool_calls（工具调用指令），则去工具节点 tool_node 执行工具调用
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    # 否则结束
    return "end"

# 定义工具执行节点：接收工具调用指令，执行对应工具，并将结果返回给LLM
tool_node = ToolNode(tools)

# 6. 构建图
graph = StateGraph(AgentState)

# 添加节点
graph.add_node("chatbot", chatbot)
graph.add_node("tools", ToolNode(tools))

# 设置入口
graph.set_entry_point("chatbot")

# 添加条件边：根据是否需要调用工具进行路由
graph.add_conditional_edges(
    "chatbot",
    tool_router,
    {
        "tools": "tools",  # 如果需要调用工具，前往 tools 节点
        "end": END,        # 否则结束
    }
)
# tools 节点执行完后，回到 chatbot 节点（让 AI 总结工具返回的结果）
graph.add_edge("tools", "chatbot")

# ======================
# 编译图（启用检查点以支持记忆功能）
# ======================
memory = MemorySaver()
app = graph.compile(checkpointer=memory)


# 7. 运行函数
def run_chat(user_input: str, thread_id: str = "default"):
    """运行聊天机器人，支持工具调用和对话历史记忆"""
    print("\n" + "="*50)
    print(f"User: {user_input}")
    print("Assistant:", end=" ", flush=True)

    # 初始化消息历史（空列表，检查点会自动加载历史）
    messages = []
    # 通过添加 thread_id 参数配置对话ID
    config = {"configurable": {"thread_id": thread_id}}
    for msg, metadata in app.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
        stream_mode="messages",
    ):
        if isinstance(msg, AIMessage) and msg.content:
            print(msg.content, end="", flush=True)
        elif isinstance(msg, ToolMessage):
            print(f"\n[工具调用结果]: {msg.content[:100]}..." if len(msg.content) > 100 else f"\n[工具调用结果]: {msg.content}")

    print()

# 8. 测试
# 可视化显示 graph（需要在 Jupyter 环境中运行）
try:
    from IPython.display import Image, display
    print("\n=== Graph 可视化 ===")
    display(Image(app.get_graph().draw_mermaid_png()))
except ImportError:
    print("\n[提示] IPython 未安装，无法显示可视化图片。请在 Jupyter 环境中运行，或安装 ipython 包")


# 测试 多轮对话（测试记忆功能）
print("\n=== 测试: 多轮对话（记忆功能测试）===")
memory_thread = "memory_test_001"

print("\n--- 第 1 轮：自我介绍 ---")
run_chat("你好，我叫小明，我是一名程序员。", thread_id=memory_thread)

print("\n--- 第 2 轮：询问名字（测试是否记住）---")
run_chat("你还记得我叫什么名字吗？", thread_id=memory_thread)

print("\n--- 第 3 轮：询问职业（测试是否记住）---")
run_chat("我是做什么的？", thread_id=memory_thread)

print("\n--- 第 4 轮：新话题（不同 thread_id，无记忆）---")
run_chat("你还记得我叫什么名字吗？", thread_id="new_thread_002")
