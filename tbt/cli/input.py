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


def get_int_from_user():
    user_input = input("# (int): ")
    try:
        return int(user_input)
    except:
        print("Invalid input. Try again.")
        return get_int_from_user()
    

def get_option_from_user(options:list[str]):
    user_input = input(f"# ({', '.join(options)}): ")
    if user_input not in options:
        print("Invalid input. Try again.")
        return get_option_from_user(options)
    return user_input