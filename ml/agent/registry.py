from ml.agent.base import Agent
from ml.agent.starter import StarterAgent
from ml.agent.qlearn import QLearnAgent

AGENT_REGISTRY = {
    Agent.name: Agent,
    StarterAgent.name: StarterAgent,
    QLearnAgent.name: QLearnAgent
}
