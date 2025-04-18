"""
LLM controller for integration with language models.
"""

import json
import logging
import math
import os
import queue
import re
import threading
import time
import traceback  # Add explicit import
from datetime import datetime

import numpy as np
import requests

from swarm_squad.config import (
    JAMMING_RADIUS_MULTIPLIER,
    LLM_ENABLED,
    LLM_ENDPOINT,
    LLM_FEEDBACK_INTERVAL,
    LLM_MODEL,
    OBSTACLE_MODE,
    PT,
    ObstacleMode,
)
from swarm_squad.controllers.base_controller import BaseController
from swarm_squad.models.swarm_state import SwarmState
from swarm_squad.utils import format_llm_feedback, format_swarm_state_for_llm

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("LLMController")


class LLMController(BaseController):
    """
    Controller that integrates with Language Models for adaptive control.

    This controller interfaces with LLMs through Arch Gateway to provide
    periodic feedback and eventually enable dynamic decision-making based on
    high-level reasoning from the LLM.
    """

    def __init__(
        self, swarm_state: SwarmState, llm_model=None, llm_feedback_interval=None
    ):
        """
        Initialize the LLM controller.

        Args:
            swarm_state: Reference to the swarm state object
            llm_model: Custom LLM model to use (overrides config)
            llm_feedback_interval: Custom LLM feedback interval (overrides config)
        """
        print("### Initializing LLM controller")
        super().__init__(swarm_state)
        self.default_controller = None  # Will hold a reference to a backup controller
        self.last_llm_update_time = 0
        self.last_llm_update_step = 0
        self.feedback_history = []
        self.current_feedback = None
        self.enabled = LLM_ENABLED
        self.step_counter = 0

        # Store custom settings
        self.llm_model = llm_model if llm_model is not None else LLM_MODEL
        self.llm_feedback_interval = (
            llm_feedback_interval
            if llm_feedback_interval is not None
            else LLM_FEEDBACK_INTERVAL
        )

        # Store the last state description for UI display
        self.last_state_description = None

        # Thread management for async LLM calls
        self.feedback_thread = None
        self.feedback_queue = queue.Queue()
        self.is_llm_request_pending = False
        self.last_request_time = 0

        # Set up logging to file
        self._setup_file_logging()

    def _setup_file_logging(self):
        """Set up dedicated file logging for LLM responses"""
        # Create logs directory if it doesn't exist
        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)

        # Create a timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = os.path.join(log_dir, f"llm_responses_{timestamp}.log")

        # Set up a file handler specifically for LLM responses
        self.file_handler = logging.FileHandler(self.log_file_path)
        self.file_handler.setLevel(logging.INFO)
        self.file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

        # Create a separate logger for the file to avoid duplicate console output
        self.file_logger = logging.getLogger("LLMResponseFile")
        self.file_logger.setLevel(logging.INFO)
        self.file_logger.addHandler(self.file_handler)
        self.file_logger.propagate = False  # Don't propagate to root logger

        # Log initial message to the file
        self.log_to_file(
            f"LLM Response Log - Model: {self.llm_model} - Started at {timestamp}"
        )
        self.log_to_file("=" * 80)

        # Log to console about the file creation
        logger.info(f"LLM responses will be logged to: {self.log_file_path}")

    def log_to_file(self, message):
        """Log a message to the dedicated LLM response log file"""
        self.file_logger.info(message)

    def schedule_reconnect(self):
        """Schedule a background reconnection attempt after a delay"""
        logger.info("Scheduling background reconnection attempt in 15 seconds...")
        self.reconnect_time = time.time() + 15
        self.should_reconnect = True

    def try_reconnect(self):
        """Try to reconnect to the LLM service"""
        if not hasattr(self, "should_reconnect") or not self.should_reconnect:
            return

        current_time = time.time()
        if current_time >= self.reconnect_time:
            logger.info("Attempting background reconnection to LLM service...")
            try:
                self.test_llm_connection()
                logger.info("LLM connection re-established successfully")
                self.enabled = True
                self.should_reconnect = False
            except Exception as e:
                logger.warning(f"Background reconnection failed: {e}")
                # Schedule another attempt with longer delay
                self.reconnect_time = current_time + 30

    def set_default_controller(self, controller: BaseController):
        """
        Set a default controller to fall back on when LLM is not active.

        Args:
            controller: The controller to use as fallback
        """
        self.default_controller = controller

    def compute_control(self) -> np.ndarray:
        """
        Calculate control inputs using LLM-guided decisions.

        This method first gets control inputs from the default controller,
        then periodically updates LLM feedback, and in the future will
        modify control inputs based on LLM reasoning.

        Returns:
            A numpy array of shape (swarm_size, 2) containing the control
            inputs for each agent in the swarm.
        """
        # Increment step counter
        self.step_counter += 1

        # Log execution at different verbosity levels
        if self.step_counter % 10 == 0:  # Reduced frequency for higher visibility logs
            logger.info(
                f"LLMController.compute_control called at step {self.step_counter}"
            )
        else:
            logger.debug(f"LLMController.compute_control at step {self.step_counter}")

        # Try to reconnect if needed
        if (
            not self.enabled
            and hasattr(self, "should_reconnect")
            and self.should_reconnect
        ):
            self.try_reconnect()

        # Get base control inputs from default controller
        if self.default_controller:
            control_inputs = self.default_controller.compute_control()
        else:
            control_inputs = self._basic_destination_control()

        # Check for completed LLM request in queue
        self._check_feedback_queue()

        # Only proceed with LLM requests if enabled
        if not self.enabled:
            return control_inputs

        # Check if it's time to update LLM feedback and no request is pending
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        steps_since_update = self.step_counter - self.last_llm_update_step

        if (
            steps_since_update >= self.llm_feedback_interval
            and not self.is_llm_request_pending
            and time_since_last_request >= 2.0
        ):  # Minimum 2 seconds between requests
            logger.info(f"Requesting LLM feedback at step {self.step_counter}")

            # Start a new thread to get LLM feedback
            self._request_llm_feedback_async()

            self.last_llm_update_step = self.step_counter
            self.last_request_time = current_time
        elif self.is_llm_request_pending:
            logger.debug(
                f"Skipping LLM request at step {self.step_counter}: request already pending"
            )
        elif steps_since_update < self.llm_feedback_interval:
            logger.debug(
                f"Skipping LLM request at step {self.step_counter}: next at step {self.last_llm_update_step + self.llm_feedback_interval}"
            )
        else:
            logger.debug(
                f"Skipping LLM request at step {self.step_counter}: last request was {time_since_last_request:.1f}s ago"
            )

        # In the future, we'll modify control_inputs based on LLM reasoning here

        return control_inputs

    def _check_feedback_queue(self):
        """Check if any feedback is available in the queue and process it"""
        try:
            # Non-blocking check for feedback
            while not self.feedback_queue.empty():
                feedback = self.feedback_queue.get_nowait()
                if feedback:
                    logger.info(
                        f"SUCCESS: Received feedback from queue: {feedback[:50]}..."
                    )
                    self.current_feedback = feedback
                    self.feedback_history.append(feedback)
                    # Keep history manageable
                    if len(self.feedback_history) > 3:
                        self.feedback_history = self.feedback_history[-3:]

                    # Log the feedback to file
                    self.log_to_file(f"FEEDBACK (Step {self.step_counter}): {feedback}")
                self.feedback_queue.task_done()

            # Check if thread is done
            if self.feedback_thread and not self.feedback_thread.is_alive():
                self.is_llm_request_pending = False
                self.feedback_thread = None
        except queue.Empty:
            pass

    def _request_llm_feedback_async(self):
        """Start a thread to request LLM feedback asynchronously"""
        if self.is_llm_request_pending:
            logger.warning("LLM request already pending, not starting a new one")
            return

        try:
            # Format the current swarm state for LLM consumption
            state_description = format_swarm_state_for_llm(self.swarm_state)
            condensed_state = self._condense_state_description(state_description)

            # Store the state description for UI display
            self.last_state_description = condensed_state

            # Create a new thread for the LLM request
            self.feedback_thread = threading.Thread(
                target=self._llm_request_worker,
                args=(condensed_state, self.feedback_queue),
                daemon=True,
            )
            self.feedback_thread.start()
            self.is_llm_request_pending = True
            logger.info("Started background thread for LLM feedback")
        except Exception as e:
            logger.error(f"Error starting LLM feedback thread: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.is_llm_request_pending = False  # Make sure we don't get stuck

    def _llm_request_worker(self, state_description, result_queue):
        """Worker function that runs in a separate thread to make LLM requests"""
        try:
            # Check if there are any jamming fields before sending the request
            has_jamming = "jamming" in state_description.lower() and (
                "experiencing" in state_description.lower()
                or "affected by" in state_description.lower()
                or "degradation" in state_description.lower()
            )

            # Construct a clear prompt with system instructions and state information
            # Adjust the prompt based on whether jamming is present
            if has_jamming:
                # Jamming-focused prompt when jamming is detected
                prompt = f"""You are a tactical advisor for a swarm of autonomous vehicles specializing in ESCAPING JAMMING FIELDS.

IMPORTANT CONTEXT:
- Communication quality ranges from 0 (no connection) to 1 (perfect quality)
- Quality above 0.94 is considered "good" and below 0.94 is "poor"
- Low-power jamming causes gradual degradation of communication quality
- High-power jamming causes abrupt disconnection and agents return to base
- JAMMING FIELDS ARE YOUR PRIMARY CONCERN - they must be escaped immediately

YOUR TASK:
1. Identify agents inside jamming fields (showing communication degradation)
2. Provide PRECISE directional instructions to escape jamming
3. Specify EXACT directions using the following system:
   - Cardinal: north, east, south, west
   - Ordinal: northeast, southeast, southwest, northwest
   - Secondary: north-northeast, east-northeast, east-southeast, south-southeast, 
     south-southwest, west-southwest, west-northwest, north-northwest
   - Tertiary: north by northeast, northeast by north, northeast by east, east by northeast,
     east by southeast, southeast by east, southeast by south, south by southeast,
     south by southwest, southwest by south, southwest by west, west by southwest,
     west by northwest, northwest by west, northwest by north, north by northwest
4. Always prioritize MOVING AWAY FROM JAMMING FIELDS over formation integrity
5. Suggest distance values between 5-10 units for effective escape

FORMAT YOUR RESPONSE as a 30-word tactical instruction with specific agent numbers and EXACT directions:
Example: "Agent-1: Move 8 units north-northwest to escape jamming. Agent-2: Reposition 10 units west by southwest to exit interference field."

REMEMBER: Focus on EXTRACTING AGENTS FROM JAMMING FIELDS first, maintain formation second.

Current swarm state:
{state_description}

Provide tactical advice:"""
            else:
                # Standard mission-focused prompt when no jamming is present
                prompt = f"""You are a tactical advisor for a swarm of autonomous vehicles.

IMPORTANT CONTEXT:
- Communication quality ranges from 0 (no connection) to 1 (perfect quality)
- Quality above 0.94 is considered "good" and below 0.94 is "poor"
- The swarm needs to maintain good communication while reaching the destination
- Physical obstacles should be avoided by all agents

YOUR TASK:
1. Analyze the formation and communication quality between agents
2. Provide PRECISE directional instructions for effective movement
3. Specify EXACT directions using the following system:
   - Cardinal: north, east, south, west
   - Ordinal: northeast, southeast, southwest, northwest
   - Secondary: north-northeast, east-northeast, east-southeast, south-southeast, 
     south-southwest, west-southwest, west-northwest, north-northwest
   - Tertiary: north by northeast, northeast by north, northeast by east, east by northeast,
     east by southeast, southeast by east, southeast by south, south by southeast,
     south by southwest, southwest by south, southwest by west, west by southwest,
     west by northwest, northwest by west, northwest by north, north by northwest
4. Ensure all agents maintain good communication links
5. Focus on reaching the destination efficiently while maintaining formation

FORMAT YOUR RESPONSE as a 30-word tactical instruction with specific agent numbers and EXACT directions:
Example: "Agent-1: Move 5 units east by southeast to improve formation. Agent-2: Reposition 4 units north-northeast to maintain optimal spacing."

REMEMBER: Balance between maintaining formation and reaching the destination is key.

Current swarm state:
{state_description}

Provide tactical advice:"""

            # Create request for Ollama API format
            request_data = {"model": self.llm_model, "prompt": prompt, "stream": False}

            # Print debug info about the request
            logger.info(
                f"Worker thread request data: {json.dumps(request_data, indent=2)}"
            )

            # Log state description to file
            self.log_to_file(
                f"STATE DESCRIPTION (Step {self.step_counter}):\n{state_description}"
            )

            # Send request directly to Ollama with a longer timeout
            start_time = time.time()

            # Use the endpoint directly as configured in the settings
            response = requests.post(
                LLM_ENDPOINT,
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=180,  # Increased timeout for worker thread (3 minutes)
            )

            # Print raw response for debugging
            logger.info(
                f"Worker thread received response with status {response.status_code}"
            )
            logger.info(f"Response text: {response.text[:200]}...")

            # Parse response
            if response.status_code == 200:
                try:
                    result = response.json()

                    # Extract content based on the Ollama API format
                    if "response" in result:
                        content = result["response"]

                        end_time = time.time()
                        logger.info(
                            f"Worker thread received LLM response in {end_time - start_time:.2f}s: {content}"
                        )

                        # Log the response to file
                        self.log_to_file(
                            f"RESPONSE (Step {self.step_counter}, Time: {end_time - start_time:.2f}s):\n{content}"
                        )
                        self.log_to_file("-" * 80)

                        # Put the result in the queue
                        result_queue.put(content)
                        return
                    else:
                        error_msg = f"No 'response' field found in result: {result}"
                        logger.error(error_msg)
                        self.log_to_file(f"ERROR: {error_msg}")
                        print(
                            f"### Worker thread error: No 'response' field in {list(result.keys())}"
                        )
                except Exception as parse_error:
                    error_msg = f"Error parsing response: {parse_error}"
                    logger.error(error_msg)
                    self.log_to_file(f"ERROR: {error_msg}")
                    print(f"### Worker thread parse error: {parse_error}")
            else:
                error_msg = f"LLM request failed with status {response.status_code}"
                logger.error(error_msg)
                self.log_to_file(f"ERROR: {error_msg}")
                print(f"### Worker thread request failed: {response.status_code}")

            # If we get here, something went wrong
            result_queue.put(None)

        except Exception as e:
            error_msg = f"Worker thread error: {str(e)}"
            logger.error(error_msg)
            self.log_to_file(f"ERROR: {error_msg}")
            print(f"### Worker thread exception: {str(e)}")
            logger.error(f"Worker thread traceback: {traceback.format_exc()}")
            result_queue.put(None)
        finally:
            logger.info("Worker thread finished")
            print("### Worker thread finished")

    def _basic_destination_control(self) -> np.ndarray:
        """
        Basic control strategy for moving toward destination.

        Returns:
            Control inputs for all agents
        """
        control_inputs = np.zeros((self.swarm_state.swarm_size, 2))

        for i in range(self.swarm_state.swarm_size):
            # Simple vector toward destination
            direction = (
                self.swarm_state.swarm_destination - self.swarm_state.swarm_position[i]
            )
            distance = np.linalg.norm(direction)

            if distance > 0:
                # Normalize and scale the control input
                control_inputs[i] = direction / distance * 0.5

        return control_inputs

    def test_llm_connection(self):
        """Test connection to Ollama"""
        test_message = {
            "model": self.llm_model,
            "prompt": "Test connection. Reply with 'OK'.",
            "stream": False,
            "max_tokens": 10,  # Keep response very short for speed
        }

        try:
            logger.info(f"Testing connection to Ollama at {LLM_ENDPOINT}")
            start_time = time.time()

            # Print out the actual request for debugging
            logger.info(f"Sending test request to Ollama: {test_message}")

            response = requests.post(
                LLM_ENDPOINT,
                json=test_message,
                headers={"Content-Type": "application/json"},
                timeout=120,  # Increased timeout to 2 minutes for large model loading
            )

            # Log the raw response for debugging
            logger.info(f"Raw response status: {response.status_code}")
            logger.info(f"Raw response text: {response.text}")

            response.raise_for_status()

            # Parse the JSON response
            result = response.json()
            logger.info(f"Parsed response: {result}")

            # Check for 'response' field in Ollama API response
            if "response" in result:
                content = result["response"]
                logger.info(f"Content from Ollama: {content}")

                if "OK" in content:
                    end_time = time.time()
                    logger.info(
                        f"Connection to Ollama successful in {end_time - start_time:.2f}s"
                    )
                    return True
                else:
                    logger.warning(f"Unexpected response from Ollama: {content}")
            else:
                logger.warning("No 'response' field in Ollama API response")

            end_time = time.time()
            logger.info(
                f"Connection to Ollama successful in {end_time - start_time:.2f}s"
            )
            return True

        except Exception as e:
            logger.error(f"Ollama connection test failed: {str(e)}")
            # Print more detailed error information
            logger.error(f"Detailed error: {traceback.format_exc()}")
            raise

    def _condense_state_description(self, state_description):
        """
        Condense the state description to reduce tokens and speed up LLM processing
        while preserving important natural language details in a more conversational tone.

        Args:
            state_description: The full state description

        Returns:
            A condensed version focusing on the most important information in natural language
        """
        # Extract the most critical info from description
        lines = state_description.split("\n")

        # Get destination information
        destination_line = next((line for line in lines if "Destination" in line), "")
        destination_match = re.search(r"\[([\d\.\-]+), ([\d\.\-]+)\]", destination_line)
        dest_x = destination_match.group(1) if destination_match else "?"
        dest_y = destination_match.group(2) if destination_match else "?"

        # Build natural language state description
        natural_desc = []

        # Determine mission status and jamming information
        mission_status = "The mission is to reach the destination at coordinates "
        mission_status += f"[{dest_x}, {dest_y}] efficiently while maintaining communication between agents."

        # Check for jamming specifically
        jamming_detected = False
        if hasattr(self.swarm_state, "jamming_affected") and np.any(
            self.swarm_state.jamming_affected
        ):
            jamming_detected = True
            # Determine jamming type based on obstacle mode
            if OBSTACLE_MODE == ObstacleMode.HIGH_POWER_JAMMING:
                if not all(self.swarm_state.agent_status):
                    mission_status += " ALERT: High-power jamming detected! Affected agents are returning to base."
            elif OBSTACLE_MODE == ObstacleMode.LOW_POWER_JAMMING:
                mission_status += " ALERT: Low-power jamming detected affecting communication quality."

        natural_desc.append(f"{mission_status}\n")

        # Add destination information
        natural_desc.append(
            f"The swarm destination is at coordinates [{dest_x}, {dest_y}].\n"
        )

        # Add obstacle information if present and visible to agents
        # Only add physical obstacles or jamming that's actually been encountered
        if self.swarm_state.obstacles:
            # Get current active obstacle mode from swarm_state instead of static config
            current_obstacle_mode = getattr(
                self.swarm_state, "obstacle_mode", OBSTACLE_MODE
            )

            # Process different types of obstacles based on current mode
            if current_obstacle_mode == ObstacleMode.HARD:
                # For physical obstacles, only report them if agents are close enough to detect them
                obstacle_descriptions = []
                # Define a detection radius - how close an agent needs to be to "detect" a physical obstacle
                detection_radius = (
                    15.0  # Adjust this value based on desired detection range
                )

                for i, obstacle in enumerate(self.swarm_state.obstacles, 1):
                    obstacle_pos = np.array([obstacle[0], obstacle[1]])
                    # Check if any agent is close enough to detect this obstacle
                    obstacle_detected = False
                    closest_agent = -1
                    closest_distance = float("inf")

                    for agent_idx in range(self.swarm_state.swarm_size):
                        dist = np.linalg.norm(
                            self.swarm_state.swarm_position[agent_idx] - obstacle_pos
                        )

                        # Keep track of closest agent for logging
                        if dist < closest_distance:
                            closest_distance = dist
                            closest_agent = agent_idx

                        if (
                            dist < detection_radius + obstacle[2]
                        ):  # Within detection range plus obstacle radius
                            obstacle_detected = True
                            # Log first detection of an obstacle
                            logger.info(
                                f"Agent {agent_idx} detected physical obstacle {i} at distance {dist:.2f}"
                            )
                            break

                    if obstacle_detected:
                        obstacle_descriptions.append(
                            f"Obstacle {i}: Position [{obstacle[0]:.1f}, {obstacle[1]:.1f}], Radius {obstacle[2]:.1f}"
                        )
                    else:
                        # Debug log of closest agent to undetected obstacle
                        logger.debug(
                            f"Closest agent to obstacle {i} is Agent {closest_agent} at distance {closest_distance:.2f} (detection threshold: {detection_radius + obstacle[2]:.2f})"
                        )

                if obstacle_descriptions:
                    natural_desc.append(
                        f"Detected physical obstacles: {' | '.join(obstacle_descriptions)}\n"
                    )

            elif (
                current_obstacle_mode == ObstacleMode.LOW_POWER_JAMMING
                and jamming_detected
            ):
                # For low power jamming, ONLY report jamming fields that agents have actually entered
                # or are close to entering
                # Count how many agents are actually affected by jamming
                affected_agents = np.where(self.swarm_state.jamming_affected)[0]

                # Include agents near jamming fields for early warning
                detection_margin = (
                    10.0  # How far outside jamming field to start warning
                )
                near_jamming_agents = []

                if len(affected_agents) > 0 or len(self.swarm_state.obstacles) > 0:
                    # Report all jamming fields that have affected agents or are nearby
                    jamming_descriptions = []
                    for i, obstacle in enumerate(self.swarm_state.obstacles, 1):
                        # Find if any agents are within or near this specific jamming field's radius
                        jamming_radius = obstacle[2] * JAMMING_RADIUS_MULTIPLIER
                        obstacle_pos = np.array([obstacle[0], obstacle[1]])

                        # Track agents that are within or near this jamming field
                        agents_in_this_field = []
                        agents_near_this_field = []

                        for agent_idx in range(self.swarm_state.swarm_size):
                            dist = np.linalg.norm(
                                self.swarm_state.swarm_position[agent_idx]
                                - obstacle_pos
                            )
                            if dist < jamming_radius:
                                agents_in_this_field.append(agent_idx)
                            elif dist < jamming_radius + detection_margin:
                                agents_near_this_field.append(agent_idx)
                                if agent_idx not in near_jamming_agents:
                                    near_jamming_agents.append(agent_idx)

                        # Always report jamming fields whether agents are in them or not
                        jamming_descriptions.append(
                            f"Jamming Field {i}: Position [{obstacle[0]:.1f}, {obstacle[1]:.1f}], Radius {jamming_radius:.1f}"
                        )

                        # Add agent information for this field
                        if agents_in_this_field:
                            jamming_descriptions[-1] += (
                                f" (affecting Agents {', '.join(map(str, agents_in_this_field))})"
                            )
                        if agents_near_this_field:
                            jamming_descriptions[-1] += (
                                f" (approaching Agents {', '.join(map(str, agents_near_this_field))})"
                            )

                    if jamming_descriptions:
                        natural_desc.append(
                            f"Encountered low-power jamming: {' | '.join(jamming_descriptions)}\n"
                        )

            elif (
                current_obstacle_mode == ObstacleMode.HIGH_POWER_JAMMING
                and jamming_detected
            ):
                # For high power jamming, don't show exact obstacles but indicate jamming presence
                affected_count = np.sum(~self.swarm_state.agent_status)
                if affected_count > 0:
                    natural_desc.append(
                        f"WARNING: High-power jamming encountered! {affected_count} agents are returning to base.\n"
                    )

        # Process each agent directly using the swarm state data
        swarm_size = self.swarm_state.swarm_size
        positions = self.swarm_state.swarm_position
        comm_matrix = self.swarm_state.communication_qualities_matrix
        from swarm_squad.utils import get_direction

        for i in range(swarm_size):
            # Get agent name
            agent_name = f"Agent-{i}"

            # Get position
            pos = positions[i]

            # Calculate distance and direction to destination
            dest_vector = self.swarm_state.swarm_destination - pos
            dist_to_dest = np.linalg.norm(dest_vector)
            dir_to_dest = get_direction(pos, self.swarm_state.swarm_destination)

            # Start building agent description
            agent_desc = [f"{agent_name} is at position [{pos[0]:.1f}, {pos[1]:.1f}]."]

            # Get current active obstacle mode from swarm_state instead of static config
            current_obstacle_mode = getattr(
                self.swarm_state, "obstacle_mode", OBSTACLE_MODE
            )

            # Add jamming information if this agent is affected
            if (
                hasattr(self.swarm_state, "jamming_affected")
                and self.swarm_state.jamming_affected[i]
            ):
                if current_obstacle_mode == ObstacleMode.HIGH_POWER_JAMMING:
                    agent_desc.append(
                        f"{agent_name} is affected by high-power jamming and returning to base."
                    )
                elif current_obstacle_mode == ObstacleMode.LOW_POWER_JAMMING:
                    # Get jamming depth if available
                    jamming_depth = getattr(
                        self.swarm_state, "jamming_depth", np.zeros(swarm_size)
                    )[i]
                    severity = (
                        "severe"
                        if jamming_depth > 0.7
                        else "moderate"
                        if jamming_depth > 0.3
                        else "mild"
                    )
                    agent_desc.append(
                        f"{agent_name} is experiencing {severity} communication degradation due to low-power jamming."
                    )

                    # Add specific advice for jamming evasion
                    # Calculate direction vectors from agent to all obstacles
                    evasion_directions = []
                    for obstacle in self.swarm_state.obstacles:
                        obstacle_pos = np.array([obstacle[0], obstacle[1]])
                        jamming_radius = obstacle[2] * JAMMING_RADIUS_MULTIPLIER
                        vector_to_obstacle = obstacle_pos - pos
                        dist_to_obstacle = np.linalg.norm(vector_to_obstacle)

                        # Only add evasion directions for nearby obstacles
                        if dist_to_obstacle < jamming_radius * 1.5:
                            # Direction AWAY from obstacle
                            evasion_vector = -vector_to_obstacle
                            # Normalize
                            if np.linalg.norm(evasion_vector) > 0:
                                evasion_vector = evasion_vector / np.linalg.norm(
                                    evasion_vector
                                )

                            # Get precise direction using helper function
                            def get_precise_direction(vector):
                                """Convert a vector to a precise cardinal direction string"""
                                # Calculate angle in degrees (0 = east, 90 = north, etc.)
                                angle_rad = math.atan2(vector[1], vector[0])
                                angle_deg = math.degrees(angle_rad)
                                if angle_deg < 0:
                                    angle_deg += 360

                                # Map angles to cardinal directions with finer granularity
                                directions = {
                                    # Cardinal directions (4)
                                    "north": 90,
                                    "east": 0,
                                    "south": 270,
                                    "west": 180,
                                    # Ordinal directions (4)
                                    "northeast": 45,
                                    "southeast": 315,
                                    "southwest": 225,
                                    "northwest": 135,
                                    # Secondary intercardinal directions (8)
                                    "north-northeast": 67.5,
                                    "east-northeast": 22.5,
                                    "east-southeast": 337.5,
                                    "south-southeast": 292.5,
                                    "south-southwest": 247.5,
                                    "west-southwest": 202.5,
                                    "west-northwest": 157.5,
                                    "north-northwest": 112.5,
                                    # Tertiary intercardinal directions (16)
                                    "north by northeast": 78.75,
                                    "northeast by north": 56.25,
                                    "northeast by east": 33.75,
                                    "east by northeast": 11.25,
                                    "east by southeast": 348.75,
                                    "southeast by east": 326.25,
                                    "southeast by south": 303.75,
                                    "south by southeast": 281.25,
                                    "south by southwest": 258.75,
                                    "southwest by south": 236.25,
                                    "southwest by west": 213.75,
                                    "west by southwest": 191.25,
                                    "west by northwest": 168.75,
                                    "northwest by west": 146.25,
                                    "northwest by north": 123.75,
                                    "north by northwest": 101.25,
                                }

                                # Find the closest direction by minimizing the angular difference
                                closest_dir = min(
                                    directions.items(),
                                    key=lambda x: min(
                                        abs(angle_deg - x[1]),
                                        abs(angle_deg - (x[1] + 360)),
                                        abs((angle_deg + 360) - x[1]),
                                    ),
                                )
                                return closest_dir[0]

                            # Get precise directional recommendation
                            precise_direction = get_precise_direction(evasion_vector)

                            # Calculate recommended distance based on depth in jamming field
                            # More severe jamming = move further
                            distance = min(10, max(5, int(jamming_depth * 15)))

                            # Skip if it would give confusing advice (too deep inside field)
                            penetration_distance = jamming_radius - dist_to_obstacle
                            if penetration_distance > jamming_radius * 0.8:
                                continue

                            evasion_directions.append((precise_direction, distance))

                    if evasion_directions:
                        # Take the first (closest) evasion direction
                        direction, distance = evasion_directions[0]
                        agent_desc.append(
                            f"RECOMMENDED EVASION: {agent_name} should move {distance} units {direction} to exit the jamming field."
                        )
                elif (
                    i in near_jamming_agents
                    if "near_jamming_agents" in locals()
                    else []
                ):
                    # Add warning for agents that are near jamming fields
                    agent_desc.append(
                        f"{agent_name} is approaching a jamming field and should change course."
                    )

                    # Find closest jamming field to this agent
                    closest_obstacle = None
                    closest_distance = float("inf")

                    for obstacle in self.swarm_state.obstacles:
                        obstacle_pos = np.array([obstacle[0], obstacle[1]])
                        jamming_radius = obstacle[2] * JAMMING_RADIUS_MULTIPLIER
                        dist_to_obstacle = np.linalg.norm(obstacle_pos - pos)

                        if dist_to_obstacle < closest_distance:
                            closest_distance = dist_to_obstacle
                            closest_obstacle = obstacle

                    if closest_obstacle:
                        # Calculate avoidance direction
                        obstacle_pos = np.array(
                            [closest_obstacle[0], closest_obstacle[1]]
                        )
                        vector_to_obstacle = obstacle_pos - pos
                        # Calculate perpendicular vector (to go around)
                        perp_vector = np.array(
                            [-vector_to_obstacle[1], vector_to_obstacle[0]]
                        )
                        # Normalize
                        if np.linalg.norm(perp_vector) > 0:
                            perp_vector = perp_vector / np.linalg.norm(perp_vector)

                        # Get precise direction using helper function
                        def get_precise_direction(vector):
                            """Convert a vector to a precise cardinal direction string"""
                            # Calculate angle in degrees (0 = east, 90 = north, etc.)
                            angle_rad = math.atan2(vector[1], vector[0])
                            angle_deg = math.degrees(angle_rad)
                            if angle_deg < 0:
                                angle_deg += 360

                            # Map angles to cardinal directions with finer granularity
                            directions = {
                                # Cardinal directions (4)
                                "north": 90,
                                "east": 0,
                                "south": 270,
                                "west": 180,
                                # Ordinal directions (4)
                                "northeast": 45,
                                "southeast": 315,
                                "southwest": 225,
                                "northwest": 135,
                                # Secondary intercardinal directions (8)
                                "north-northeast": 67.5,
                                "east-northeast": 22.5,
                                "east-southeast": 337.5,
                                "south-southeast": 292.5,
                                "south-southwest": 247.5,
                                "west-southwest": 202.5,
                                "west-northwest": 157.5,
                                "north-northwest": 112.5,
                                # Tertiary intercardinal directions (16)
                                "north by northeast": 78.75,
                                "northeast by north": 56.25,
                                "northeast by east": 33.75,
                                "east by northeast": 11.25,
                                "east by southeast": 348.75,
                                "southeast by east": 326.25,
                                "southeast by south": 303.75,
                                "south by southeast": 281.25,
                                "south by southwest": 258.75,
                                "southwest by south": 236.25,
                                "southwest by west": 213.75,
                                "west by southwest": 191.25,
                                "west by northwest": 168.75,
                                "northwest by west": 146.25,
                                "northwest by north": 123.75,
                                "north by northwest": 101.25,
                            }

                            # Find the closest direction by minimizing the angular difference
                            closest_dir = min(
                                directions.items(),
                                key=lambda x: min(
                                    abs(angle_deg - x[1]),
                                    abs(angle_deg - (x[1] + 360)),
                                    abs((angle_deg + 360) - x[1]),
                                ),
                            )
                            return closest_dir[0]

                        # Get precise avoidance direction
                        precise_direction = get_precise_direction(perp_vector)

                        # Calculate distance based on proximity to field
                        avoidance_distance = int(
                            min(
                                10,
                                max(
                                    6,
                                    detection_margin
                                    - (
                                        closest_distance
                                        - closest_obstacle[2]
                                        * JAMMING_RADIUS_MULTIPLIER
                                    ),
                                ),
                            )
                        )

                        agent_desc.append(
                            f"RECOMMENDED AVOIDANCE: {agent_name} should move {avoidance_distance} units {precise_direction} to avoid the jamming field."
                        )
                elif (
                    hasattr(self.swarm_state, "jamming_affected")
                    and not self.swarm_state.jamming_affected[i]
                ):
                    # Only add this message if jamming mode is active but agent is not affected
                    if (
                        current_obstacle_mode == ObstacleMode.LOW_POWER_JAMMING
                        or current_obstacle_mode == ObstacleMode.HIGH_POWER_JAMMING
                    ) and jamming_detected:
                        agent_desc.append(
                            f"{agent_name} is currently outside jamming fields and has normal communications."
                        )

            if not (
                current_obstacle_mode == ObstacleMode.HIGH_POWER_JAMMING
                and not self.swarm_state.agent_status[i]
            ):
                agent_desc.append(
                    f"{agent_name} is {dist_to_dest:.1f} units away from the destination and needs to travel in the {dir_to_dest} direction to reach it."
                )

            # Add communication links with all other agents
            comm_links = []
            for j in range(swarm_size):
                if i != j:  # Don't include self-connection
                    other_agent = f"Agent-{j}"
                    quality = comm_matrix[i, j]

                    # Skip connection info for high-power jamming affected agents
                    if current_obstacle_mode == ObstacleMode.HIGH_POWER_JAMMING:
                        # Skip if either agent is inactive
                        if (
                            not self.swarm_state.agent_status[i]
                            or not self.swarm_state.agent_status[j]
                        ):
                            continue

                    # Calculate distance and direction
                    other_pos = positions[j]
                    distance = np.linalg.norm(other_pos - pos)
                    direction = get_direction(pos, other_pos)

                    # Convert direction code to natural language
                    direction_text = {
                        "N": "north",
                        "NE": "northeast",
                        "E": "east",
                        "SE": "southeast",
                        "S": "south",
                        "SW": "southwest",
                        "W": "west",
                        "NW": "northwest",
                    }.get(direction, direction)

                    # Determine link quality description
                    quality_desc = "poor" if quality < PT else "good"
                    link_status = "connected" if quality > PT else "disconnected"

                    # Add jamming indication if we're in jamming mode
                    jamming_indication = ""
                    if (
                        current_obstacle_mode == ObstacleMode.LOW_POWER_JAMMING
                        and self.swarm_state.jamming_affected[i]
                        and self.swarm_state.jamming_affected[j]
                    ):
                        jamming_indication = ", affected by jamming"

                    # Format the communication info in natural language
                    comm_links.append(
                        f"{other_agent} ({distance:.1f} units away to the {direction_text}, {quality:.2f} {quality_desc} quality{jamming_indication}, {link_status})"
                    )

            if comm_links:
                agent_desc.append(f"{agent_name} has communication with:")
                for link in comm_links:
                    agent_desc.append(f"  - {link}")
            else:
                agent_desc.append(
                    f"{agent_name} has no communication links with other agents."
                )

            # Add a blank line after each agent description
            natural_desc.append("\n".join(agent_desc) + "\n")

        condensed_state = "\n".join(natural_desc)
        logger.info(f"Condensed state:\n{condensed_state}")
        return condensed_state

    def get_last_feedback(self):
        """Return the most recent LLM feedback"""
        # Check for new feedback before returning
        self._check_feedback_queue()
        return self.current_feedback

    def get_feedback_history(self, limit=3):
        """
        Return the feedback history with newest first

        Args:
            limit: Maximum number of history items to return

        Returns:
            List of feedback strings, newest first
        """
        return self.feedback_history[-limit:]

    def format_feedback_for_display(self):
        """
        Format the current feedback and history for display in GUI

        Returns:
            Formatted string with current feedback highlighted and history
        """
        if not self.current_feedback:
            if self.is_llm_request_pending:
                return "Waiting for tactical advice..."
            return "No tactical advice available"

        # Use the utility function to format the current feedback
        current_time = time.strftime("%H:%M:%S", time.localtime())
        result = [format_llm_feedback(self.current_feedback, current_time)]

        # Add history if available
        history = (
            self.get_feedback_history(limit=2)[1:]
            if len(self.feedback_history) > 1
            else []
        )
        if history:
            result.append("\nPREVIOUS ADVICE:")
            for i, feedback in enumerate(history):
                result.append(f"{i + 1}. {feedback}")

        return "\n".join(result)

    # Methods for future LLM-based control implementation
    def analyze_situation(self):
        """
        Analyze the current swarm state and formulate a query for the LLM.
        This method extracts relevant information from the swarm state
        to create context for the LLM decision.
        """
        # This will be implemented when we expand LLM control capabilities
        pass

    def interpret_llm_response(self, response):
        """
        Interpret the LLM's response and convert it to control actions.
        This parses text or structured output from the LLM into
        specific control parameters.
        """
        # This will be implemented when we expand LLM control capabilities
        pass
