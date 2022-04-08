def calculate_mean_time(time_list: list[float]):
    # Remove min and max value
    time_list.remove(min(time_list))
    time_list.remove(max(time_list))

    # Calculate mean
    mean_time = sum(time_list) / len(time_list)
    return mean_time
