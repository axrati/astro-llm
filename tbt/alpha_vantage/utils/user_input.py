def get_boolean_from_user():
    user_input = input("# (y|n): ").lower()
    valid_map =  {
        "y":True,
        "n":False,
        "yes":True,
        "no":False,
    }
    if user_input not in list(valid_map.keys()):
        print("Invalid input. Try again.")
        return get_boolean_from_user()
    return valid_map[user_input]