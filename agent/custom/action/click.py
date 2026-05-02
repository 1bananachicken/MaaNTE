import json
from .utils import click_rect

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

@AgentServer.custom_action("click_override")
class ClickOverride(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> CustomAction.RunResult:
        print("=== Click Action Started ===")
        controller = context.tasker.controller
        
        if argv.custom_action_param is not None:
            try:
                params = json.loads(argv.custom_action_param)
                target = params.get("target")
                if not target or len(target) != 4:
                    print("Invalid rect parameter.")
                    return CustomAction.RunResult(success=False)
            except Exception as e:
                print(f"Error parsing parameters: {e}")
                return CustomAction.RunResult(success=False)

            click_rect(controller, target, 0.005)
            print(f"Clicked at rect: {target}")
            return CustomAction.RunResult(success=True)
        
        elif argv.reco_detail is not None:
            click_rect(controller, argv.box, 0.005)
            return CustomAction.RunResult(success=True)
        
        else:
            print("No valid parameters provided for click action.")
            return CustomAction.RunResult(success=False)
