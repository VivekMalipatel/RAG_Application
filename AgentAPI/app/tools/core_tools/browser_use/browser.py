def mb_start_browser(state: dict) -> dict:
    """Start browser session for LangGraph workflow"""
    try:
        from marinabox import MarinaboxSDK
        sdk = MarinaboxSDK()
        session = sdk.create_session(
            env_type="browser",
            tag="LangGraph-Browser",
            initial_url="https://google.com"
        )
        print(f"[LangGraph] Browser session started: {session.session_id}")
        return {
            **state,
            "session_id": session.session_id,
            "vnc_port": session.vnc_port,
            "status": session.status,
            "browser_active": True
        }
    except ImportError:
        print("[LangGraph] Marinabox not available, using fallback session")
        import uuid
        return {
            **state,
            "session_id": str(uuid.uuid4()),
            "browser_active": True
        }
session_info = mb_start_browser({})
existing_session_id = session_info['session_id']
print("Your browser session ID:", existing_session_id)