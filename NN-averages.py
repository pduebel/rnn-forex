df[f"{ratio}_close_20avg"] = df[[f"{ratio}_close"]].ewm(span=20).mean()
    df[f"{ratio}_close_40avg"] = df[[f"{ratio}_close"]].ewm(span=40).mean()