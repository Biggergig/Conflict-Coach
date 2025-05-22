import os
import sys
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    RemoveMessage,
)
from langchain_core.prompts import PromptTemplate

from langgraph.graph import START, StateGraph
from operator import add

from typing import Literal, List, Annotated

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "conflict-coach-vTool"

MAX_QUESTIONS = 1
SAVE_IMAGE = "--img" in sys.argv

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)


# State
class ContextState(MessagesState):
    questions: Annotated[List[str], add]
    answers: Annotated[List[str], add]
    context: str


# Tools
from langchain_core.tools import tool


@tool
def ask_user_question(question: str) -> str:
    """Ask the user a question for further context, and returns their response."""
    return input("QUESTION: " + question + "\n > ")


ask_user_question_node = ToolNode([ask_user_question])


# Nodes
def init(state: ContextState):
    starting_message = state["messages"][-1]
    message_transcript = starting_message.content

    sys_msg = SystemMessage(
        content='You are a personal coach tasked with helping the user handle their conflicts. Your goal is to ultimately help the user understand the other person\'s perspective, and to do that your current objective is to understand the situation. You will now recieve a message with the initial context of the conversation. You can either ask a question for clarification or more context, or send the word "DONE".'
    )
    user_msg = HumanMessage(content=message_transcript)

    return ContextState(
        messages=[RemoveMessage(id=starting_message.id), sys_msg, user_msg],  # type: ignore
        context="<EMPTY>",
        questions=[],
        answers=[],
    )


def gather_context(state: ContextState):
    print(state)
    tool_llm = llm.bind_tools([ask_user_question])
    response = tool_llm.invoke(state["messages"])

    return {"messages": state["messages"] + [response]}


def followup(state: ContextState):
    print("In followup", state)
    assert isinstance(
        state["messages"][-1], AIMessage
    ), "Expected last message to be an AIMessage"
    assert state["messages"][-1].tool_calls, "Expected last message to have tool calls"
    tool_call = state["messages"][-1].tool_calls[0]
    print(tool_call)

    # Should only be a call to ask_user_question
    question = tool_call["args"]["question"]
    response = ask_user_question_node.invoke(state)

    print("response to question:", response)
    return {
        "messages": response["messages"],
        "questions": [question],
        "answers": [response["messages"][-1].content],
    }


def summarize(state: ContextState):
    print("SUMMARIZE:\n", state)

    # Retrieve the initial user message (assumed second)
    initial_message = state["messages"][1].content
    followup_questions = state["questions"]
    followup_answers = state["answers"]

    prompt = f"""
You are a personal coach helping the user handle a conflict. Here is the relevant information from the conversation:

Original context given by the user:
{initial_message}

Follow-up questions and answers:
    """
    for question, answer in zip(followup_questions, followup_answers):
        prompt += f"""
Q: {question}
A: {answer}
"""
    prompt += """
Please output a concise but comprehensive summary of the conversation, with the goal of capturing the full context of the conversation.
"""
    prompt = prompt.strip()

    # print(prompt)
    response = llm.invoke(prompt)
    # print("LLM summary response:", response.content)
    # print("Initial message:", initial_message)
    # print("Questions:", followup_questions)
    # print("Follow-up answers:", followup_answers)
    return {"context": response.content}


def response(state: ContextState):
    context = state["context"]
    print("Context:", context)


# Conditional Edges
def has_question(state: ContextState) -> Literal["followup", "summarize"]:
    last_message = state["messages"][-1]
    assert isinstance(
        last_message, AIMessage
    ), "Expected last message to be an AIMessage"

    # No more than 3 questions allowed
    if not last_message.tool_calls or len(state["answers"]) >= MAX_QUESTIONS:
        return "summarize"
    else:
        return "followup"


def build_graph():
    builder = StateGraph(ContextState)

    # Add nodes to the graph
    builder.add_node("init", init)
    builder.add_node("gather_context", gather_context)
    builder.add_node("followup", followup)
    builder.add_node("summarize", summarize)
    builder.add_node("response", response)

    # Add edges to the graph
    builder.add_edge(START, "init")
    builder.add_edge("init", "gather_context")
    builder.add_conditional_edges("gather_context", has_question)
    builder.add_edge("followup", "gather_context")
    builder.add_edge("summarize", "response")

    return builder.compile()


if __name__ == "__main__":
    graph = build_graph()
    graph.get_graph(xray=True).print_ascii()
    # Save the Image to a file
    if SAVE_IMAGE:
        graph_image = graph.get_graph(xray=True).draw_mermaid_png()
        with open("graph.png", "wb") as f:
            f.write(graph_image)

    messages = [
        HumanMessage(
            content='My partner sent me the text "You didn\'t put the laundry away, you never do anything!", and they asked me to do that yesterday but I forgot because I was busy.'
        )
    ]

    response = graph.invoke({"messages": messages})
    print("\n\nResponse:\n", response)
