import argparse
import asyncio
import json
import os
import sys
from agent_eval_utils.excel_reader import ExcelReader
from agent_eval_utils.excel_writer import ExcelWriter
from com_vitalai_aimp_domain.model.AIMPIntent import AIMPIntent
from com_vitalai_aimp_domain.model.AgentMessageContent import AgentMessageContent
from com_vitalai_aimp_domain.model.UserMessageContent import UserMessageContent
from vital_agent_container_client.aimp_message_handler_inf import AIMPMessageHandlerInf
from vital_agent_container_client.vital_agent_container_client import VitalAgentContainerClient
from vital_ai_vitalsigns.utils.uri_generator import URIGenerator
from vital_ai_vitalsigns.vitalsigns import VitalSigns

from agent_eval.agent_weather_eval.agent_weather_eval_manager import HaleyAgentEvalManager


class LocalMessageHandler(AIMPMessageHandlerInf):

    def __init__(self):
        self.response_list = []

    async def receive_message(self, message):
        print(f"Local Handler Received message: {message}")
        self.response_list.append(message)


class HaleyAgentEvalCommand:
    def __init__(self, args):
        self.parser = self.create_parser()
        self.args = self.parser.parse_args()
        self.vital_home = os.getenv('VITAL_HOME', '')

    def create_parser(self):

        parser = argparse.ArgumentParser(prog="haleyagenteval", description="HaleyAgentEval Command", add_help=True)

        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        help_parser = subparsers.add_parser('help', help="Display help information")

        info_parser = subparsers.add_parser('info', help="Display information about the system and environment")

        eval_parser = subparsers.add_parser('eval', help="Evaluate an Excel file")

        eval_parser.add_argument('-i', '--input-file', type=str, required=True, help="Excel input file path to process")
        eval_parser.add_argument('-o', '--output-file', type=str, required=True,
                                 help="Excel output file path to save the result")

        return parser

    def run(self):
        if self.args.command == 'help':
            self.parser.print_help()
        elif self.args.command == 'eval':
            if self.args.input_file and self.args.output_file:
                self.process_excel(self.args.input_file, self.args.output_file)
            else:
                self.parser.print_help()
        elif self.args.command == 'info':
            self.info()
        else:
            self.parser.print_help()

    def info(self):
        vital_home = self.vital_home
        print("HaleyAgentEval Info")
        print(f"Current VITAL_HOME: {vital_home}")

    async def process_excel_async(self, input_file_path, output_file_path):

        print("Initializing...")

        vs = VitalSigns()

        print("Initialized.")

        print(f"Processing Excel file: {input_file_path}")

        handler = LocalMessageHandler()

        client = VitalAgentContainerClient("http://localhost:7007", handler)

        manager = HaleyAgentEvalManager(handler=handler, client=client)

        health = await client.check_health()

        print("Health:", health)

        excel_reader = ExcelReader()

        rows_data = excel_reader.read_excel_to_dict(input_file_path)

        headers = [
            'Identifier',
            'AccountIdentifier',
            'LoginIdentifier',
            'SessionIdentifier',
            'Action',
            'MessageClass',
            'MessageText'
        ]

        output_data = []

        for row in rows_data:
            # print(row)

            action = row.get('Action', None)

            if action == 'SendMessage':

                await client.open_websocket()

                message_class = row.get('MessageClass', None)

                if message_class == 'AIMPIntent':

                    intent_type = row.get('IntentType', None)

                    if intent_type == 'CHAT':
                        await manager.handle_chat_message(row, output_data)

        if len(output_data) > 0:
            excel_writer = ExcelWriter()
            print(f"Writing Excel file: {output_file_path}")
            excel_writer.write_excel(output_file_path, headers, output_data)

        await client.close_websocket()

    def process_excel(self, input_file_path, output_file_path):

        if not os.path.exists(input_file_path):
            print(f"File {input_file_path} does not exist.")
            sys.exit(1)

        asyncio.run(self.process_excel_async(input_file_path, output_file_path))


def main():
    import sys
    command = HaleyAgentEvalCommand(sys.argv[1:])
    command.run()


if __name__ == "__main__":
    main()
