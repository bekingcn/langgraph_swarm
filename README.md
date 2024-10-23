## LangGraph Swarm

The LangGraph Swarm is a alternative implementation to the OpenAI's [Swarm](https://github.com/openai/swarm) framework with [langchain](https://github.com/langchain-ai/langchain) and [langgraph](https://github.com/langchain-ai/langgraph).


> **Coming soon...  with full configuration implementation in [LcStack](https://github.com/bekingcn/LcStack)**

## Overview

**Why LangGraph Swarm?**

LangGraph Swarm reimplements the [Swarm](https://github.com/openai/swarm) framework with [langchain](https://github.com/langchain-ai/langchain) and [langgraph](https://github.com/langchain-ai/langgraph). With LangGraph Swarm, you can create agents and integrate them with Langchain components. If you already have a langchain agent, you can benefit from Swarm's features without having to re-invent the wheel.

## Examples

- [`lg_swarm_demo`](notes/lg_swarm_demo.ipynb): A simple demo of how to set up an agent

We have implemented the following examples as OpenAI's Swarm. 
Check out `/examples` for inspiration! Learn more about each one in its README.

- [`basic`](examples/basic): Simple examples of fundamentals like setup, function calling, handoffs, and context variables
- [`triage_agent`](examples/triage_agent): Simple example of setting up a basic triage step to hand off to the right agent
- [`weather_agent`](examples/weather_agent): Simple example of function calling
- [`personal_shopper`](examples/personal_shopper): A personal shopping agent that can help with making sales and refunding orders
- TODO: [`airline`](examples/airline): A multi-agent setup for handling different customer service requests in an airline context.
- TODO: [`support_bot`](examples/support_bot): A customer service bot which includes a user interface agent and a help center agent with several tools

## Open issues

If you have any questions or feedback, please open an issue on [github](https://github.com/bekingcn/langgraph_swarm)

Or you can reach me on [email](mailto:beking_cn@hotmail.com)