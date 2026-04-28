import sys

BOLD_YELLOW = "\033[1;33m"
COLOR_RESET = "\033[0m"


def print_alert(message, width=60):
    line = BOLD_YELLOW + "=" * width + COLOR_RESET
    sys.stderr.write(f"\n{line}\n")
    for msg_line in message.split("\n"):
        sys.stderr.write(f"{BOLD_YELLOW}{msg_line}{COLOR_RESET}\n")
    sys.stderr.write(f"{line}\n\n")
    sys.stderr.flush()
