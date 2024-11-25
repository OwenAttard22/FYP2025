from bluesky.network.client import Client
import threading
import time

bsclient = None

class TextClient(Client):
    def __init__(self):
        super().__init__()
        self.running = True
        self.thread = threading.Thread(target=self.update_loop)
        self.thread.start()

    def update_loop(self):
        while self.running:
            self.update()
            time.sleep(0.02)

    def event(self, name, data, sender_id):
        if name == b'ECHO':
            print(f"Received ECHO: {data.get('text')}")

    def stack(self, text):
        self.send_event(b'STACK', text)

    def stop(self):
        self.running = False
        self.thread.join()

def main():
    global bsclient
    bsclient = TextClient()
    bsclient.connect(event_port=11000, stream_port=11001)

    try:
        commands = [
            "ECHO Hello, BlueSky!",
            "PLUGIN TESTPLUGIN",
        ]

        for command in commands:
            print(f"Sending command: {command}")
            bsclient.stack(command)
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")

if __name__ == '__main__':
    main()
