import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# 1. 加载环境变量
load_dotenv()

# ======================
# 2. 构建 PostgreSQL DSN
# ======================
PG_DSN = os.getenv("PG_DSN")
if not PG_DSN:
    PG_DSN = (
        f"postgresql://{os.getenv('PG_USER', 'postgres')}:{os.getenv('PG_PASSWORD', '')}"
        f"@{os.getenv('PG_HOST', 'localhost')}:{os.getenv('PG_PORT', '5432')}"
        f"/{os.getenv('PG_DBNAME', 'langgraph_db')}"
    )

# 3. 创建支持工具调用的 LLM
llm = ChatOpenAI(
    model="qwen-plus-latest",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    streaming=True,
)

# 4. 定义工具
@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气（模拟数据）"""
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
        result = eval(expression, {"__builtins__": {}}, {})
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算错误：{str(e)}"

@tool
def search_web(query: str) -> str:
    """模拟联网搜索（返回示例结果）"""
    return f"搜索结果：关于 '{query}'，这是一个模拟的搜索结果。在实际应用中，这里会返回真实的网络搜索内容。"

tools = [calculator, get_weather, search_web]
llm_with_tools = llm.bind_tools(tools)

# 5. 定义状态、节点、图
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: AgentState):
    """聊天机器人节点"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def tool_router(state: AgentState) -> Literal["tools", "end"]:
    """路由判断节点"""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "end"

# 构建图
graph = StateGraph(AgentState)
graph.add_node("chatbot", chatbot)
graph.add_node("tools", ToolNode(tools))
graph.set_entry_point("chatbot")
graph.add_conditional_edges("chatbot", tool_router, {"tools": "tools", "end": END})
graph.add_edge("tools", "chatbot")

# 6. 运行函数
def _format_tool_content(content: str) -> str:
    """格式化工具调用结果的显示"""
    return content[:100] + "..." if len(content) > 100 else content

def run_chat(user_input: str, thread_id: str = "default", app=None):
    """运行聊天机器人，支持工具调用和 PostgreSQL 持久化记忆"""
    if app is None:
        raise ValueError("app 参数不能为 None，请先创建编译后的图")

    print("\n" + "="*50)
    print(f"User: {user_input}")
    print("Assistant:", end=" ", flush=True)

    config = {"configurable": {"thread_id": thread_id}}

    for msg, metadata in app.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
        stream_mode="messages",
    ):
        if isinstance(msg, AIMessage) and msg.content:
            print(msg.content, end="", flush=True)
        elif isinstance(msg, ToolMessage):
            print(f"\n[工具调用结果]: {_format_tool_content(msg.content)}")

    print()


# 7. 测试
try:
    from IPython.display import Image, display
    print("\n=== Graph 可视化 ===")
    display(Image(app.get_graph().draw_mermaid_png()))
except ImportError:
    print("\n[提示] IPython 未安装，无法显示可视化图片。请在 Jupyter 环境中运行，或安装 ipython 包")

# ======================
# 使用上下文管理器确保 PostgreSQL 连接正确打开和关闭
# ======================
with PostgresSaver.from_conn_string(PG_DSN) as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)

    print("\n=== 测试: 多轮对话（PostgreSQL 持久化记忆）===")
    memory_thread = "postgres_memory_001"

    print("\n--- 第 1 轮：自我介绍 ---")
    run_chat("你好，我叫小明，我是一名程序员。", thread_id=memory_thread, app=app)

    print("\n--- 第 2 轮：询问名字（测试是否记住）---")
    run_chat("你还记得我叫什么名字吗？", thread_id=memory_thread, app=app)

    print("\n--- 第 3 轮：询问职业（测试是否记住）---")
    run_chat("我是做什么的？", thread_id=memory_thread, app=app)

    print("\n--- 第 4 轮：新话题（不同 thread_id，无记忆）---")
    run_chat("你还记得我叫什么名字吗？", thread_id="new_thread_002", app=app)

    print("\n=== PostgreSQL 持久化记忆测试完成 ===")
