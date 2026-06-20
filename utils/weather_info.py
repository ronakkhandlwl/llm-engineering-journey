def get_current_temperature(location: str) -> dict:
    """Returns current temperature for few cities"""

    if location == "Tokyo":
        return {"temperature": 20}
    elif location == "London":
        return {"temperature": 10}
    elif location == "New York":
        return {"temperature": 15}
    elif location == "Mumbai":
        return {"temperature": 25}
    else:
        return {"temperature": "Not found"}
