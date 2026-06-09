import pandas as pd

def velocity(df, window_seconds):
    """Calculate the velocity of transactions for each card1 within a specified time window.using sliding window approach."""
    df = df.sort_values(["card1", "TransactionDT"])

    velocities = []
    for card, group in df.groupby("card1"):

        times = group["TransactionDT"].values

        i = 0
        count = [0] * len(times)

        for j in range(len(times)):

            while times[j] - times[i] > window_seconds:
                i += 1

            count[j] = j - i

        results = pd.Series(count, index=group.index)

        velocities.append(results)
    

        

    return pd.concat(velocities).sort_index()