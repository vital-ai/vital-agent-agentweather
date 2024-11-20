import json
import logging

from com_vitalai_aimp_domain.model.AIMPIntent import AIMPIntent
from com_vitalai_aimp_domain.model.AgentMessageContent import AgentMessageContent
from com_vitalai_aimp_domain.model.UserMessageContent import UserMessageContent
from vital_ai_vitalsigns.utils.uri_generator import URIGenerator
from vital_ai_vitalsigns.vitalsigns import VitalSigns


class HaleyAgentEvalManager:
    def __init__(self, handler, client):
        self.handler = handler
        self.client = client

    async def handle_chat_message(self, row, output_data):

        vs = VitalSigns()

        logger = logging.getLogger("HaleyAgentEvalLogger")

        action = row.get('Action', None)
        message_class = row.get('MessageClass', None)
        intent_type = row.get('IntentType', None)
        logger.info(f"Sending Message: {message_class}")

        message_uri = row.get('MessageUri', None)
        message_text = row.get('MessageText', None)

        logger.info(f"Message Text: {message_text}")

        aimp_intent = AIMPIntent()
        aimp_intent.URI = message_uri
        aimp_intent.aIMPIntentType = "http://vital.ai/ontology/vital-aimp#AIMPIntentType_CHAT"

        message_content = UserMessageContent()
        message_content.URI = URIGenerator.generate_uri()
        message_content.text = message_text

        message = [aimp_intent, message_content]

        # string
        message_json = vs.to_json(message)
        # list of dict
        message_list = json.loads(message_json)

        await self.client.send_message(message_list)
        await self.client.wait_for_close_or_timeout(60)

        response_list = self.handler.response_list

        message_text = 'Message Text'

        for response_message in response_list:
            # print(f"Response Message: {response_message}")
            go_list = []
            for message in response_message:
                message_string = json.dumps(message)
                go = vs.from_json(message_string)
                go_list.append(go)
            for go in go_list:
                if isinstance(go, AgentMessageContent):
                    agent_text = go.text
                    message_text = str(agent_text)

        logger.info(f"Agent Text: {message_text}")

        response_dict = {
            'MessageText': message_text
        }

        output_data.append(response_dict)


