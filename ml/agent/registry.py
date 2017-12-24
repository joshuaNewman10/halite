from ml.agent.base import Agent
from ml.agent.starter import StarterAgent

AGENT_REGISTRY = {
    Agent.name: Agent,
    StarterAgent.name: StarterAgent
}
