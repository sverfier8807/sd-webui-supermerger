class UserSettings:
    def __init__(self):
        self.value = []
    
    def __getitem__(self, key):
        return self.value[key]

    def __setitem__(self, key, value):
        self.value[key] = value
        
USER_SETTINGS = UserSettings()