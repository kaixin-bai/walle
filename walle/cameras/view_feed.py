from walle.cameras import RealSenseD415


if __name__ == "__main__":
    cam = RealSenseD415(resolution="low")
    cam.view_feed()