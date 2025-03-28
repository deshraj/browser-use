from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
	BaseMessage,
	HumanMessage,
	SystemMessage,
)
from pydantic import BaseModel

from browser_use.agent.message_manager.service import MessageManager
from browser_use.agent.message_manager.views import ManagedMessage, MessageMetadata
from browser_use.utils import time_execution_sync

logger = logging.getLogger(__name__)


class MemorySummarizerSettings(BaseModel):
	"""Settings for memory summarization."""

	enable_summarization: bool = True
	summarize_every_n_steps: int = 10


class MemorySummarizer:
	"""
	Manages memory summarization for agents.

	This class is responsible for summarizing agent memories to optimize
	context window usage for long-running tasks.
	"""

	def __init__(
		self,
		message_manager: MessageManager,
		llm: BaseChatModel,
		settings: MemorySummarizerSettings = MemorySummarizerSettings(),
	):
		self.message_manager = message_manager
		self.llm = llm
		self.settings = settings

	@time_execution_sync('--summarize_memories')
	def summarize_memories(self, current_step: int) -> None:
		"""
		Summarize memories if needed based on the current step.

		Args:
		    current_step: The current step number of the agent
		"""
		if not self.settings.enable_summarization:
			return

		# Check if current step is a multiple of summarize_every_n_steps
		if current_step % self.settings.summarize_every_n_steps != 0:
			return

		logger.info(f'Summarizing memories at step {current_step}')

		# Get all messages
		all_messages = self.message_manager.state.history.messages

		# Filter out messages that are marked as memory in metadata
		messages_to_process = []
		new_messages = []
		for msg in all_messages:
			# Don't process system message and summarized messages
			if isinstance(msg, ManagedMessage) and msg.metadata.message_type in set(['init', 'memory']):
				new_messages.append(msg)
			else:
				messages_to_process.append(msg)

		if len(messages_to_process) <= 1:
			logger.info('Not enough non-memory messages to summarize')
			return

		# Create a summary
		summary = self._create_summary([m.message for m in messages_to_process], current_step)

		if not summary:
			logger.warning('Failed to create summary')
			return

		# Replace the summarized messages with the summary
		summary_message = HumanMessage(content=summary)
		summary_tokens = self.message_manager._count_tokens(summary_message)
		summary_metadata = MessageMetadata(tokens=summary_tokens, message_type='memory')

		# Calculate the total tokens being removed
		removed_tokens = sum(m.metadata.tokens for m in messages_to_process)

		# Add the summary message
		new_messages.append(ManagedMessage(message=summary_message, metadata=summary_metadata))

		# Update the history
		self.message_manager.state.history.messages = new_messages
		self.message_manager.state.history.current_tokens -= removed_tokens
		self.message_manager.state.history.current_tokens += summary_tokens

		logger.info(f'Memories summarized: {len(messages_to_process)} messages replaced with summary')
		logger.info(f'Token reduction: {removed_tokens - summary_tokens} tokens')

	def _create_summary(self, messages: List[BaseMessage], current_step: int) -> Optional[str]:
		"""
		Create a summary of the given messages using the LLM.

		Args:
		    messages: List of messages to summarize

		Returns:
		    A string summary of the messages or None if summarization failed
		"""
		try:
			# Create system prompt for summarization
			system_prompt = SystemMessage(
				content=f"""
You are a memory summarization system that summarizes the interaction history between a human and a browser agent. You are provided with the agent execution history of agent for past 10 steps. Your task is to create a summary of the agent's output history making sure to preserve all the information and context needed to continue the task. Make sure to include the output of the agent in the summary since that is the most important part of the history.

For each step, make sure to include following information:
- Task objective: The overall goal the agent is trying to accomplish
- Progress status: Current completion percentage and specific steps completed
- Key findings: Important information discovered (URLs, data points, search results)
- Navigation history: Which pages visited and their relevance
- Errors & challenges: Problems encountered and solutions attempted
- Current context: What page/state the agent is in now and what it's trying to do next
- Agent actions: What actions the agent has taken and what the results were

Guidelines:
1. Make sure to include the output of the agent in the summary since that is the most important part of the history.
2. Bullet points should start with action number {current_step - 10} and end with {current_step} to make it easier to understand the progress since we are summarizing every {self.settings.summarize_every_n_steps} steps.
3. For each action:
- Be specific and include exact data (URLs, element indexes, error messages)
- Maintain chronological order of important actions
- Preserve numeric counts and metrics (e.g., '3 out of 5 items processed')
- Retain error messages and their causes for reference
4. Only output the summary and nothing else.

The summary must be comprehensive enough that if this were the agent's only memory, it could continue the task effectively.
"""
			)
			human_prompt = HumanMessage(content='Generate a summary of the above browser agent execution history.')
			response = self.llm.invoke([system_prompt, *messages, human_prompt])
			return str(response.content)
		except Exception as e:
			logger.error(f'Error creating summary: {e}')
			return None
