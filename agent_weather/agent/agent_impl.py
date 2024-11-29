import asyncio
import logging
import pprint
from datetime import datetime
from typing import List, Any
import httpx
from ai_haley_kg_domain.model.KGChatBotMessage import KGChatBotMessage
from ai_haley_kg_domain.model.KGChatUserMessage import KGChatUserMessage
from ai_haley_kg_domain.model.KGToolRequest import KGToolRequest
from ai_haley_kg_domain.model.KGToolResult import KGToolResult
from com_vitalai_aimp_domain.model.AIMPResponseMessage import AIMPResponseMessage
from com_vitalai_aimp_domain.model.AgentMessageContent import AgentMessageContent
from com_vitalai_aimp_domain.model.UserMessageContent import UserMessageContent
from com_vitalai_haleyai_question_domain.model.HaleyContainer import HaleyContainer
from kgraphplanner.agent.kg_planning_structured_agent import KGPlanningStructuredAgent
from kgraphplanner.checkpointer.memory_checkpointer import MemoryCheckpointer
from kgraphplanner.tool_manager.tool_manager import ToolManager
from kgraphplanner.tools.place_search.place_search_tool import PlaceSearchTool
from kgraphplanner.tools.weather.weather_info_tool import WeatherInfoTool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from starlette.websockets import WebSocket
from typing_extensions import TypedDict
from vital_agent_kg_utils.vitalsignsutils.vitalsignsutils import VitalSignsUtils
from vital_ai_vitalsigns.utils.uri_generator import URIGenerator
from vital_ai_vitalsigns.vitalsigns import VitalSigns
from langchain.callbacks.base import BaseCallbackHandler
from zoneinfo import ZoneInfo
from agent_weather.agent.agent_context import AgentContext
from agent_weather.config.local_config import LocalConfig


async def process_stream(stream, messages_out: list) -> TypedDict:

    logger = logging.getLogger("HaleyAgentLogger")

    pp = pprint.PrettyPrinter(indent=4, width=40)

    final_result = None

    async for s in stream:
        message = s["messages"][-1]
        messages_out.append(message)
        if isinstance(message, tuple):
            logger.info(f"Stream Message: {message}")
        else:
            message.pretty_print()
        final_result = s

    logger.info(f"Final Result: {final_result}")

    response = final_result.get("final_response", None)

    logger.info("Final_Result:\n")
    logger.info("--------------------------------------")
    logger.info(pp.pformat(response))
    logger.info("--------------------------------------")
    return response


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

    async def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        self.logger.info(f"LLM Request: {prompts}")

    async def on_llm_end(self, response, **kwargs):
        self.logger.info(f"LLM Response: {response.generations}")


def extract_tool_response_data(tool_manager, messages_out: List[Any]) -> List[TypedDict]:

    logger = logging.getLogger("HaleyAgentLogger")

    capture_guids = []
    capture_classes = {}
    for message in messages_out:
        if hasattr(message, 'tool_calls'):
            for call in message.tool_calls:
                if call['name'] == 'capture_response':
                    args = call.get('args', {})
                    guid = args.get('tool_response_guid')
                    class_name = args.get('response_class_name')
                    if guid and class_name:
                        capture_guids.append(guid)
                        capture_classes[guid] = class_name

    logger.info(f"capture_guids: {capture_guids}")
    logger.info(f"capture_classes: {capture_classes}")

    captured_responses = []

    for guid in capture_guids:
        tool_data = tool_manager.get_tool_cache().get_from_cache(guid)
        if tool_data:
            captured_responses.append(tool_data)

    return captured_responses


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

        # logger.info(f"Container: {container}")

        history_list = []

        if container:

            container_list = VitalSignsUtils.unpack_container(container)

            # logger.info(f"Container List: {container_list}")

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

        model_tools = ChatOpenAI(model_name="gpt-4o", callbacks=[logging_handler], temperature=0)
        model_structured = ChatOpenAI(model_name="gpt-4o", callbacks=[logging_handler], temperature=0)

        tool_endpoint = config["agent_weather"]["tool_endpoint"]

        tool_config = {
            "tool_endpoint": tool_endpoint
        }

        tool_manager = ToolManager(tool_config)

        place_search_config = {
            "tool_endpoint": tool_endpoint,
            "place_search_tool": {}
        }
        weather_config = {
            "tool_endpoint": tool_endpoint,
            "weather_tool": {}
        }

        place_search_tool = PlaceSearchTool(place_search_config, tool_manager=tool_manager)

        weather_tool = WeatherInfoTool(weather_config, tool_manager=tool_manager)

        tool_function_list = []

        for t in tool_manager.get_tool_list():
            tool_function_list.append(t.get_tool_function())

        memory = MemoryCheckpointer()

        message_queue = asyncio.Queue()

        stop_event = asyncio.Event()

        agent = KGPlanningStructuredAgent(model_tools=model_tools,
                                          model_structured=model_structured,
                                          checkpointer=memory,
                                          tools=tool_function_list,
                                          reasoning_queue=message_queue)

        graph = agent.compile()

        pp = pprint.PrettyPrinter(indent=4, width=40)

        graph_config = {"configurable": {"thread_id": "urn:thread_1"}}

        today = datetime.now(ZoneInfo('America/New_York'))

        today_str = today.strftime('%m-%d-%Y')

        system_prompt = f"""
                Today's date is: {today_str}
                You are a helpful assistant named Haley.
                You are chatting with: {agent_context.username}
                You may be given a history of recent chat messages between you and the person you are assisting.
                These messages may contain prior tool requests and results.
                These will be prefixed by "** AI Prior Tool" and contain JSON
                You may use this information to know previous tool requests and results that occurred before this current interaction.
                If you are reporting a date, you should include the day of week if possible.
                You may format chat messages using markdown, such as for tables.
                """

        system_message = SystemMessage(content=system_prompt)

        content = message_text

        ###################################

        # TODO add in incoming history
        # chat_message_list = [
        #    ("system", system_prompt)
        # ]

        # for h in history_list:
        #    chat_message_list.append(h)

        # chat_message_list.append(("human", message_text))

        # logger.info(chat_message_list)

        # inputs = {"messages": chat_message_list}

        # messages_out = []

        message_input = [
            system_message,
            HumanMessage(content=content)
        ]

        inputs = {"messages": message_input}

        messages_out = []

        agent_status_response = await process_stream(graph.astream(inputs, graph_config, stream_mode="values"), messages_out)

        for m in messages_out:
            t = type(m)
            logger.info(f"History ({t}): {m}")

        # there may be minor variants for cases of capturing knowledge graph objects
        # such as objects for the UI, etc.

        tool_response_data_list = extract_tool_response_data(tool_manager, messages_out)

        # logger.info(tool_response_data_list)

        for tool_response_data in tool_response_data_list:
            logger.info(f"Response Data:\n{tool_response_data}\n")

        human_text_request = agent_status_response.get("human_text_request", None)
        agent_text_response = agent_status_response.get("agent_text_response", None)
        agent_request_status = agent_status_response.get("agent_request_status", None)
        agent_include_payload = agent_status_response.get("agent_include_payload", None)
        agent_payload_class_list = agent_status_response.get("agent_payload_class_list", None)
        agent_payload_guid_list = agent_status_response.get("agent_payload_guid_list", None)
        agent_request_status_message = agent_status_response.get("agent_request_status_message", None)
        missing_input = agent_status_response.get("missing_input", None)

        logger.info(f"Status: {agent_request_status}")

        logger.info(f"Human Text: {human_text_request}")
        logger.info(f"Agent Text: {agent_text_response}")

        if agent_include_payload:
            logger.info(f"Agent Payload ClassList: {agent_payload_class_list}")
            logger.info(f"Agent Payload GuidList: {agent_payload_guid_list}")

            for guid in agent_payload_guid_list:
                response_obj = tool_manager.get_tool_cache().get_from_cache(guid)
                tool_data_class = response_obj.get("tool_data_class", None)

                logger.info("--------------------------------------")
                logger.info(f"Tool Data Class: {tool_data_class}")
                pp.pprint(response_obj)
                logger.info("--------------------------------------")

        ###################################

        # TODO add in incoming history
        # chat_message_list = [
        #    ("system", system_prompt)
        # ]

        # for h in history_list:
        #    chat_message_list.append(h)

        # chat_message_list.append(("human", message_text))

        # logger.info(chat_message_list)

        # inputs = {"messages": chat_message_list}

        # messages_out = []

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

        response_text = agent_text_response

        logger.info(f"Response Text: {response_text}")

        response_msg = AIMPResponseMessage()
        response_msg.URI = URIGenerator.generate_uri()
        response_msg.aIMPIntentType = "http://vital.ai/ontology/vital-aimp#AIMPIntentType_CHAT"

        agent_msg_content = AgentMessageContent()
        agent_msg_content.URI = URIGenerator.generate_uri()
        agent_msg_content.text = response_text  # "Hello from Agent."

        # TODO add in AIMPTask

        # TODO add in weather card(s) based on selected weatherdata dicts
        # should these be in the container too?

        message = [response_msg, agent_msg_content, container]

        message_json = vs.to_json(message)

        await websocket.send_text(message_json)

        logger.info(f"Sent Message: {message_json}")

        # await websocket.close(1000, "Processing Complete")
        # print(f"Websocket closed.")
        started_event.set()
        logger.info(f"Completed Event.")
