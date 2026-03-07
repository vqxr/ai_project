from typing import List, Optional
import queue
import time
from evo_swarm.core.scheduler.scheduler import Scheduler
from evo_swarm.core.interfaces.agent import Agent
from evo_swarm.core.events import Event
from rich.console import Console

console = Console()

class LocalEventScheduler(Scheduler):
    """
    A simple single-threaded loop that routes events between agents.
    """
    def __init__(self):
        self.agents: List[Agent] = []
        self.event_queue = queue.Queue()
        self.running = False

    def register_agent(self, agent: Agent):
        self.agents.append(agent)
        # Inject the publish method so the agent can send events back to the scheduler
        agent.set_event_bus(self.publish_event)
        console.print(f"[green]Registered agent:[/green] {agent.name}")

    def publish_event(self, event: Event):
        """Called by agents to broadcast an event."""
        self.event_queue.put(event)

    def process_next_event(self) -> bool:
        """Process a single event from the queue. Returns True if an event was processed."""
        if self.event_queue.empty():
            return False
            
        event = self.event_queue.get()
        console.print(f"[blue][Event][/blue] {event.event_type} from {event.sender}")
        
        # Route to specific receiver or broadcast
        for agent in self.agents:
            if event.receiver is None or event.receiver == agent.name:
                try:
                    agent.handle_event(event)
                except Exception as e:
                    console.print(f"[red]Error in agent {agent.name} handling {event.event_type}: {e}[/red]")
        return True

    def start(self):
        self.running = True
        console.print("[bold green]Scheduler started.[/bold green] Waiting for events...")
        while self.running:
            processed = self.process_next_event()
            if not processed:
                time.sleep(0.1) # Prevent hot-looping when idle

    def stop(self):
        self.running = False
        console.print("[bold yellow]Scheduler stopped.[/bold yellow]")
