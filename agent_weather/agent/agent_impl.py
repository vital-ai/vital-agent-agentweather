import asyncio
import json
import logging
from datetime import datetime, timedelta
import httpx
from ai_haley_kg_domain.model.KGChatBotMessage import KGChatBotMessage
from ai_haley_kg_domain.model.KGChatUserMessage import KGChatUserMessage
from ai_haley_kg_domain.model.KGToolRequest import KGToolRequest
from ai_haley_kg_domain.model.KGToolResult import KGToolResult
from com_vitalai_aimp_domain.model.AIMPIntent import AIMPIntent
from com_vitalai_aimp_domain.model.AIMPResponseMessage import AIMPResponseMessage
from com_vitalai_aimp_domain.model.AgentMessageContent import AgentMessageContent
from com_vitalai_aimp_domain.model.UserMessageContent import UserMessageContent
from com_vitalai_haleyai_question_domain.model.HaleyContainer import HaleyContainer
from com_vitalai_haleyai_question_domain.model.KGPropertyMap import KGPropertyMap
from kgraphplanner.agent.kg_planning_agent import KGPlanningAgent
from kgraphplanner.checkpointer.memory_checkpointer import MemoryCheckpointer
from kgraphplanner.tool_manager.tool_manager import ToolManager
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from starlette.websockets import WebSocket
from vital_agent_container.handler.aimp_message_handler_inf import AIMPMessageHandlerInf
from vital_agent_kg_utils.vitalsignsutils.vitalsignsutils import VitalSignsUtils
from vital_ai_vitalsigns.utils.uri_generator import URIGenerator
from vital_ai_vitalsigns.vitalsigns import VitalSigns
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from zoneinfo import ZoneInfo

from agent_weather.agent.agent_context import AgentContext
from agent_weather.config.local_config import LocalConfig
from agent_weather.tools.weather_info_tool import WeatherInfoTool


def print_stream(stream, messages_out: list = []):
    for s in stream:
        message = s["messages"][-1]
        messages_out.append(message)
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


def get_timestamp() -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return timestamp


class LoggingHandler(BaseCallbackHandler):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        self.logger.info(f"LLM Request: {prompts}")

    def on_llm_end(self, response, **kwargs):
        self.logger.info(f"LLM Response: {response.generations}")


class AgentImpl:
    def __init__(self):
        pass

    async def handle_chat_message(
            self,
            local_config: LocalConfig,
            config,
            client: httpx.AsyncClient,
            websocket: WebSocket, started_event: asyncio.Event,
            agent_context: AgentContext,
            agent_state,
            message_list
    ):

        logger = logging.getLogger(__name__)

        vs = VitalSigns()

        message_text = ""

        for go in message_list:
            if isinstance(go, UserMessageContent):
                user_text = go.text
                message_text = str(user_text)

        logger.info(f"Message Text: {message_text}")

        container = VitalSignsUtils.get_object_type(message_list, "http://vital.ai/ontology/haley-ai-question#HaleyContainer")

        # print(f"Container: {container}")

        history_list = []

        if container:

            container_list = VitalSignsUtils.unpack_container(container)

            # print(f"Container List: {container_list}")

            VitalSignsUtils.log_object_list("Container", container_list)

            # for now, add tool requests/responses from previous history as raw JSON
            # later, do so in a more clean way

            for c in container_list:
                if isinstance(c, KGChatUserMessage):
                    text = str(c.kGChatMessageText)
                    history_list.append(("human", text))
                if isinstance(c, KGChatBotMessage):
                    text = str(c.kGChatMessageText)
                    history_list.append(("ai", text))
                if isinstance(c, KGToolRequest):
                    tool_request_json = str(c.kGJSON)
                    tool_request_text = "** AI Prior Tool Request JSON: " + tool_request_json
                    history_list.append(("ai", tool_request_text))
                if isinstance(c, KGToolResult):
                    tool_result_json = str(c.kGJSON)
                    tool_result_text = "** AI Prior Tool Result JSON: " + tool_result_json
                    history_list.append(("ai", tool_result_text))

        logging_handler = LoggingHandler()

        llm = ChatOpenAI(model='gpt-4o', callbacks=[logging_handler], temperature=0)

        weather_info_tool = WeatherInfoTool({})

        tool_config = {}
        tool_manager = ToolManager(tool_config)

        # getting tools to use in agent into a function list

        tool_manager.add_tool(weather_info_tool)

        weather_info_tool_name = WeatherInfoTool.get_tool_cls_name()

        # function list

        tool_list = [

            tool_manager.get_tool(weather_info_tool_name).get_tool_function()

        ]

        # today = datetime.today()
        today = datetime.now(ZoneInfo('America/New_York'))


        # Format the dates as 'MM-DD-YYYY'
        today_str = today.strftime('%m-%d-%Y')

        memory = MemoryCheckpointer()

        agent = KGPlanningAgent(llm, checkpointer=memory, tools=tool_list)

        graph = agent.compile()

        config = {"configurable": {"thread_id": "urn:thread_1"}}

        system_prompt = f"""
        Today's date is: {today_str}
        You are a helpful assistant named Haley.
        You are chatting with: {agent_context.username}
        You may be given a history of recent chat messages between you and the person you are assisting.
        These messages may contain prior tool requests and results.
        These will be prefixed by "** AI Prior Tool" and contain JSON
        You may use this information to know previous tool requests and results that occurred before this current interaction.
        You may format chat messages using markdown, such as for tables.
        """

        chat_message_list = [
            ("system", system_prompt)
        ]

        for h in history_list:
            chat_message_list.append(h)

        chat_message_list.append(("human", message_text))

        logger.info(chat_message_list)

        inputs = {"messages": chat_message_list}

        messages_out = []

        print_stream(graph.stream(inputs, config, stream_mode="values"), messages_out)

        history_out_list = []

        for m in messages_out:
            t = type(m)
            logger.info(f"History ({t}): {m}")
            if isinstance(m, HumanMessage):
                user_message = KGChatUserMessage()
                user_message.URI = URIGenerator.generate_uri()
                user_message.kGChatMessageText = m.content
                history_out_list.append(user_message)

            if isinstance(m, AIMessage):
                if m.tool_calls:
                    tool_request = KGToolRequest()
                    tool_request.URI = URIGenerator.generate_uri()

                    tool_request.kGToolRequestType = "urn:langgraph_openai_tool_request"

                    tool_request_json = m.to_json()

                    tool_request.kGJSON = tool_request_json
                    history_out_list.append(tool_request)
                    logger.info(tool_request.to_json(pretty_print=False))
                else:
                    bot_message = KGChatBotMessage()
                    bot_message.URI = URIGenerator.generate_uri()
                    bot_message.kGChatMessageText = m.content
                    history_out_list.append(bot_message)

            if isinstance(m, ToolMessage):
                tool_result = KGToolResult()
                tool_result.URI = URIGenerator.generate_uri()
                tool_result.kGToolResultType = "urn:langgraph_openai_tool_result"

                tool_result_json = m.to_json()

                tool_result.kGJSON = tool_result_json
                history_out_list.append(tool_result)
                logger.info(tool_result.to_json(pretty_print=False))

        container = HaleyContainer()
        container.URI = URIGenerator.generate_uri()

        if len(history_out_list) > 0:
            logger.info(f"Outgoing container size is: {len(history_out_list)}")
            container = VitalSignsUtils.pack_container(container, history_out_list)

        last_message = messages_out[-1]
        response_text = last_message.content

        logger.info(f"Response Text: {response_text}")

        response_msg = AIMPResponseMessage()
        response_msg.URI = URIGenerator.generate_uri()
        response_msg.aIMPIntentType = "http://vital.ai/ontology/vital-aimp#AIMPIntentType_CHAT"

        agent_msg_content = AgentMessageContent()
        agent_msg_content.URI = URIGenerator.generate_uri()
        agent_msg_content.text = response_text  # "Hello from Agent."

        message = [response_msg, agent_msg_content, container]

        message_json = vs.to_json(message)

        await websocket.send_text(message_json)
        logger.info(f"Sent Message: {message_json}")

        # await websocket.close(1000, "Processing Complete")
        # print(f"Websocket closed.")
        started_event.set()
        logger.info(f"Completed Event.")
