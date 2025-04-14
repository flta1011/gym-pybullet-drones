class DroneController:
    def __init__(self, uri):
        self.uri = uri
        self.latest_position = None
        self.latest_measurement = None

    def connect(self):
        # Connect to the drone
        pass

    def send_hover_command(self, hover):
        # Send hover command
        pass

    def get_position(self):
        return self.latest_position

    def get_measurements(self):
        return self.latest_measurement
