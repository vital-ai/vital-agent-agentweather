
class AgentContext:
    def __init__(self, *,
                 session_id: str = None,
                 account_id: str = None,
                 login_id: str = None,
                 username: str = None):
        self.session_id = session_id
        self.account_id = account_id
        self.login_id = login_id
        self.username = username
