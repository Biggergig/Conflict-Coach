import os
import sys
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    AnyMessage,
    RemoveMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.graph import MermaidDrawMethod

from langgraph.graph import START, StateGraph, END
from operator import add

from typing import Literal, List, Annotated, TypedDict, Dict, Any

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "conflict-coach-vTool"

MAX_QUESTIONS = 3
SAVE_IMAGE = "--img" in sys.argv

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)


# State
class ContextState(MessagesState):
    questions: Annotated[List[str], add]
    answers: Annotated[List[str], add]
    context: str


class ResponseState(TypedDict):
    context: str
    response: str


# Tools
@tool
def ask_user_question(question: str) -> str:
    """Ask the user a question for further context, and returns their response."""
    return input("QUESTION: " + question + "\n > ")


ask_user_question_node = ToolNode([ask_user_question])


# Nodes
def init(state):
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
    if len(state["answers"]) >= MAX_QUESTIONS:
        return {}
    tool_llm = llm.bind_tools([ask_user_question])
    response = tool_llm.invoke(state["messages"])

    return {"messages": state["messages"] + [response]}


def followup(state: ContextState):
    assert isinstance(
        state["messages"][-1], AIMessage
    ), "Expected last message to be an AIMessage"
    assert state["messages"][-1].tool_calls, "Expected last message to have tool calls"
    tool_call = state["messages"][-1].tool_calls[0]

    # Should only be a call to ask_user_question
    question = tool_call["args"]["question"]
    response = ask_user_question_node.invoke(state)

    return {
        "messages": response["messages"],
        "questions": [question],
        "answers": [response["messages"][-1].content],
    }


def summarize(state: ContextState):
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


def respond(state: ContextState) -> ResponseState:
    context = state["context"]

    prompt_template = ChatPromptTemplate(
        [
            (
                "system",
                "You are a personal coach helping the user handle interpersonal conflicts. I will provide you with summaries of different contexts, and you will first process the situation. To process it, you will consider it from the perspective of the other person by writing down what you think are their thoughts and feelings. Finally, you will output a proper response for the user to reply with taking the perspective of the other person into account.",
            ),
            (
                "user",
                'The user is dealing with a conflict with their partner, who sent a text expressing frustration about the user forgetting to put the laundry away. The message included the statement, "You never do anything," which the user feels is an overreaction to the situation. The user acknowledges that they had intended to complete the task but forgot due to being busy.',
            ),
            (
                "assistant",
                """
SITUATION: Your partner asked you to put the laundry away yesterday and accused you of not contributing enough.

THOUGHTS: They are likely not upset about just this incident, but rather about this pattern and how they are tasked with too much without help.

FEELINGS: They are likely feeling frustrated, unsupported, and overworked.

RESPONSE: "I understand that you're feeling frustrated about the laundry, and I'm really sorry for forgetting. I can see how it might feel like I'm not contributing enough, and I want to make sure that isn't the case because it's not fair to you otherwise. Let's talk about how we can better share these tasks so it doesn't feel like a burden to you.
""".strip(),
            ),
            ("user", "{context}"),
        ]
    )

    filled_template = prompt_template.invoke(dict(state))
    output = llm.invoke(filled_template)
    # print("RESPONSE:\n", output.text())

    return ResponseState(context=context, response=output.text())


# Conditional Edges
def has_question(state: ContextState) -> Literal["followup", "summarize"]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        return "summarize"

    # No more than 3 questions allowed
    if not last_message.tool_calls or len(state["answers"]) >= MAX_QUESTIONS:
        return "summarize"
    else:
        return "followup"


if __name__ == "__main__":
    builder = StateGraph(ContextState, output=ResponseState)

    # Add nodes to the graph
    builder.add_node("init", init)
    builder.add_node("gather_context", gather_context)
    builder.add_node("followup", followup)
    builder.add_node("summarize", summarize)
    builder.add_node("respond", respond)

    # Add edges to the graph
    builder.add_edge(START, "init")
    builder.add_edge("init", "gather_context")
    builder.add_conditional_edges("gather_context", has_question)
    builder.add_edge("followup", "gather_context")
    builder.add_edge("summarize", "respond")
    builder.add_edge("respond", END)

    graph = builder.compile()

    graph.get_graph(xray=True).print_ascii()
    # Save the Image to a file
    if SAVE_IMAGE:
        graph_image = graph.get_graph(xray=True).draw_mermaid_png(
            draw_method=MermaidDrawMethod.PYPPETEER
        )
        with open("graph.png", "wb") as f:
            f.write(graph_image)

    user_context = input(
        "Please give me some context on the situation you are struggling with:\n"
    )

    messages: List[AnyMessage] = [HumanMessage(content=user_context)]

    response: Dict[str, Any] = graph.invoke({"messages": messages})
    # print(response)
    print("\n\nFinal Output:\n\nContext:\n", response["context"])
    print("\nResponse:\n", response["response"])
