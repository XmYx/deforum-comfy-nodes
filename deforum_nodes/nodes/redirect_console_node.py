import sys
import asyncio

console_redirected = None
# stdout_backup = sys.stdout
# stderr_backup = sys.stderr
class StreamToWebSocket:
    def __init__(self, original_stream, server, stream_type='stdout'):
        self.original_stream = original_stream
        self.server = server
        self.stream_type = stream_type

    def write(self, message):
        # Write to the original stdout or stderr
        self.original_stream.write(message)

        # Asynchronously send to the frontend via WebSocket
        if message.strip():  # Avoid sending empty messages
            asyncio.run_coroutine_threadsafe(
                self.server.send('console_output', {'message': message, 'stream': self.stream_type}),
                self.server.loop
            )

    def flush(self):
        self.original_stream.flush()

    def __getattr__(self, attr):
        # Delegate attribute access to the original stream
        return getattr(self.original_stream, attr)


class DeforumRedirectConsole:

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"redirect_console":  ("BOOLEAN", {"default": False},),}
                }

    RETURN_TYPES = ("BOOLEAN",)
    OUTPUT_NODE = True

    FUNCTION = "fn"
    display_name = "Redirect Console"
    CATEGORY = "deforum"

    def fn(self, redirect_console):
        global console_redirected

        if redirect_console:
            if not console_redirected:
                try:
                    import server
                    server_instance = server.PromptServer.instance
                    sys.stdout = StreamToWebSocket(sys.stdout, server_instance, 'stdout')
                    sys.stderr = StreamToWebSocket(sys.stderr, server_instance, 'stderr')
                    console_redirected = True
                except:
                    pass
            # else:
            #     sys.stdout = stdout_backup
            #     sys.stderr = stderr_backup
        else:
            # if console_redirected:
            #     sys.stdout = stdout_backup
            #     sys.stderr = stderr_backup
            console_redirected = False
        return (console_redirected,)


