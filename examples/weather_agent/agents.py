import json

from langgraph_swarm import Agent, tool

@tool
def get_weather(location, time="now"):
    """Get the current weather in a given location. Location MUST be a city.

    Args:
        location (str): The location to get the weather for
        time (str, optional): The time to get the weather for. Defaults to "now".

    Returns:
        str: The weather in the location
    """

    return json.dumps({"location": location, "temperature": "65", "time": time})

@tool
def send_email(recipient, subject, body):
    """Sends an email to a recipient with a subject and body.

    Args:
        recipient (str): The recipient's email address
        subject (str): The subject of the email
        body (str): The body of the email

    Returns:
        str: A success message
    """
    print("Sending email...")
    print(f"To: {recipient}")
    print(f"Subject: {subject}")
    print(f"Body: {body}")
    return "Sent!"


weather_agent = Agent(
    name="Weather Agent",
    instructions="You are a helpful agent.",
    functions=[get_weather, send_email],
)
