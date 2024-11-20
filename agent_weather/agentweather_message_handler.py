import asyncio
import json
import logging
import httpx
from com_vitalai_aimp_domain.model.AIMPIntent import AIMPIntent
from com_vitalai_aimp_domain.model.AIMPMessage import AIMPMessage
from com_vitalai_aimp_domain.model.AIMPResponseMessage import AIMPResponseMessage
from com_vitalai_aimp_domain.model.AgentMessageContent import AgentMessageContent
from com_vitalai_aimp_domain.model.UserMessageContent import UserMessageContent
from starlette.websockets import WebSocket
from vital_agent_container.handler.aimp_message_handler_inf import AIMPMessageHandlerInf
from vital_ai_vitalsigns.utils.uri_generator import URIGenerator
from vital_ai_vitalsigns.vitalsigns import VitalSigns
from agent_weather.agent.agent_context import AgentContext
from agent_weather.agent.agent_impl import AgentImpl
from agent_weather.agent.agent_state_impl import AgentStateImpl
from agent_weather.config.local_config import LocalConfig


class AgentWeatherMessageHandler(AIMPMessageHandlerInf):

    def __init__(self, agent: AgentImpl, app_home: str):
        self.agent = agent
        self.app_home = app_home
        self.local_config = LocalConfig(app_home)

    async def process_message(self,
                              config,
                              client: httpx.AsyncClient,
                              websocket: WebSocket,
                              data: str,
                              started_event: asyncio.Event):

        logger = logging.getLogger(__name__)

        try:

            logger.info(f"Handler Received Message: {data}")

            vs = VitalSigns()

            message_list = []

            json_list = json.loads(data)

            try:
                for m in json_list:
                    logger.info(f"Object: {m}")
                    m_string = json.dumps(m)
                    go = vs.from_json(m_string)
                    logger.info(f"Graph Object: {go.to_json()}")
                    message_list.append(go)

            except Exception as e:
                logger.error(e)

            if len(message_list) > 0:

                aimp_message: AIMPMessage = message_list[0]

                session_id = str(aimp_message.sessionID)

                account_id = str(aimp_message.accountURI)
                login_id = str(aimp_message.userID)
                username = str(aimp_message.username)

                logger.info(f"Session ID: {session_id}")

                logger.info(f"Account ID: {account_id}")
                logger.info(f"Login ID: {login_id}")
                logger.info(f"Username: {username}")

                # extract state from message list:
                # interaction, container with message history

                agent_context = AgentContext(
                    session_id=session_id,
                    account_id=account_id,
                    login_id=login_id,
                    username=username
                )

                agent_state = AgentStateImpl(message_list)

                if isinstance(aimp_message, AIMPIntent):

                    intent_type = str(aimp_message.aIMPIntentType)

                    if intent_type == "http://vital.ai/ontology/vital-aimp#AIMPIntentType_CHAT":
                        await self.agent.handle_chat_message(
                            self.local_config,
                            config,
                            client,
                            websocket,
                            started_event,
                            agent_context,
                            agent_state,
                            message_list)

                        return

            # handle unknown type

        except asyncio.CancelledError:
            # log canceling
            raise
