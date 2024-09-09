import logging
import os
import uvicorn
from vital_agent_container.agent_container_app import AgentContainerApp
from vital_ai_vitalsigns.vitalsigns import VitalSigns
from dotenv import load_dotenv

from agent_weather.agent.agent_impl import AgentImpl

# Load environment variables from .env file
load_dotenv()


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


def create_app():

    print('Hello Smartlinx Agent')

    vs = VitalSigns()

    print(f"VitalSigns Initialized.")

    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    app_home = current_file_directory

    agent = AgentImpl()

    handler = None

    return AgentContainerApp(handler, app_home)


app = create_app()


if __name__ == "__main__":
    uvicorn.run(host="0.0.0.0", port=7007, app=app)
