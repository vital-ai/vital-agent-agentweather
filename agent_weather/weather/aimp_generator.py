import datetime
from vital_agent_kg_utils.vital_agent_rest_resource_client.tools.weather.weather_response import WeatherData
from vital_agent_kg_utils.vital_agent_rest_resource_client.tools.weather.weather_tool_handler import WeatherToolHandler


class AIMPGenerator:
    def __init__(self):
        pass

    def convert_weather_data_to_renderer_format(self, weather_data: WeatherData) -> dict:
        """
        Converts WeatherData into a dictionary format suitable for a renderer.

        Args:
            weather_data (WeatherData): The weather data to convert.

        Returns:
            dict: A dictionary representation for rendering.
        """
        summary = ""  # summarize_weather(weather_data)

        main_icon = WeatherToolHandler.get_weather_code_id(weather_data['daily_predictions'][0]['weather_code'])

        days = []

        for prediction in weather_data['daily_predictions']:
            date_obj = datetime.datetime.strptime(prediction['date'], "%Y-%m-%d")
            day_of_week = date_obj.strftime("%a")
            days.append({
                "dow": day_of_week,
                "icon": WeatherToolHandler.get_weather_code_id(prediction['weather_code']),
                "maxTemp": round(prediction['temperature_max']),
                "minTemp": round(prediction['temperature_min'])
            })

        renderer_data = {
            "searchString": "New York City",
            "mainIcon": main_icon,
            "summary": summary,
            "temperature": round(weather_data['current_temperature']),

            # find parameter
            "precipitation": round(weather_data['daily_predictions'][0].get('rain_sum', 0)),

            "humidity": weather_data['current_humidity'],
            "wind": round(weather_data['wind_speed']),
            "days": days,
            "staticCard": False,
            "forecastIoLink": "https://www.agentweather.com"
        }

        return renderer_data
